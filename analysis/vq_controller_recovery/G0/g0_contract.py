"""G0: freeze the recovery-experiment contract before any diagnostic or training run.

Read-only. Verifies and records, per docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md section 3:

  1. git HEAD / branch / tracked-dirty / tracked-diff SHA
  2. the active pairing: VQ best.pt (epoch 90) + Controller best.pt + norm + DB + latent
     stats + VQ target cache + style prior, with recomputed fingerprints cross-checked
     against the ones recorded inside the controller checkpoint
  3. precision/device/dependency environment
  4. test availability: CPU and GPU regression suites actually run
  5. wrapper DRY_RUN expansions parse with the trainers' own argparse
  6. inventory of already-fixed issues (must not be re-reported) and open observations

Writes G0/contract.json and G0/AUDIT.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from Utils.reproducibility import artifact_fingerprints, file_sha256, state_fingerprint
from Utils.runtime import load_norm_stats, sanitize_state_dict_keys, set_seed, resolve_device

VQ = ROOT / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
CONTROLLER = ROOT / "checkpoints" / "flowmatch_label_history_obsnoise_tf100"
DB = ROOT / "database_100sty_375_memmap_nomirror"
LATENT_STATS = ROOT / "checkpoints" / "flowmatch_label_nomirror_150ep" / "latent_stats.npz"
VQ_CACHE = ROOT / "cache" / "vq_targets_vq150ep_nomirror_repair_v2"
STYLE_PRIOR = ROOT / "cache" / "style_code_prior_vq150ep_nomirror_repair_v2.npz"

ALREADY_FIXED = [
    "S4 g0 acceptance gate is now mandatory/fail-closed (12861af) — do not re-file as a bug",
    "S4 best.pt epochs now read from history.jsonl (12861af) — B0 ep1 / B1 ep3 is correct",
    "S4 continuous-AE causal wording corrected (12861af) — c.jitter +2.0%, not 'smoothed'",
    "VQ entry wrapper --no-decoder-only argparse bug fixed (28ac8a0) + regression test",
    "S2/S3/S4 contract fixes: norm pinned, loader generators, teacher fingerprint, resume guards",
    "codebook_boundary_distance exact Voronoi plane + margin/no_grad search (0a11707 era)",
]
OPEN_OBSERVATIONS = [
    "controller wrapper expands duplicated flags (--epochs/--flow-layers twice); argparse last-wins "
    "makes it benign, recorded here as cosmetic only",
    "flowmatch_label_history_obsnoise_tf100 contains an extra best_loss.pt from an older artifact "
    "convention; not used by any runner",
    "historical 'shell tail' anomaly in an old pilot log (AUDIT section 6.1) was never reproduced; "
    "wrapper syntax + DRY_RUN verified in G0",
    "controller trainer resume guard covers objective version; C2 must bump it if semantics change",
]


def run(*args, **kw) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, **kw).stdout.strip()


def git_state() -> dict:
    tracked_dirty = bool(run("git", "status", "--porcelain", "--untracked-files=no"))
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    return dict(
        commit=run("git", "rev-parse", "HEAD"),
        branch=run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        tracked_dirty=tracked_dirty,
        tracked_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        untracked_files=len([l for l in run("git", "ls-files", "--others", "--exclude-standard").splitlines() if l]),
    )


def parse_with_trainer(command_line: str, trainer_module: str, strip_prefix: str) -> dict:
    """Parse an expanded DRY_RUN argv with the trainer's own argparse (fail-closed check)."""
    import contextlib
    import importlib
    import io
    import shlex

    body = command_line[len(strip_prefix):] if command_line.startswith(strip_prefix) else command_line
    argv = shlex.split(body)[2:]  # drop python + script path
    module = importlib.import_module(trainer_module)
    old = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.parse_args()
        return dict(ok=True, error=None)
    except SystemExit as exc:
        return dict(ok=False, error=f"SystemExit({exc.code})")
    finally:
        sys.argv = old


def dry_run_wrapper(script: str, env_extra: dict) -> dict:
    env = {**os.environ, "DRY_RUN": "1", "PYTHON_BIN": sys.executable, **env_extra}
    proc = subprocess.run(["bash", str(ROOT / "scripts" / script)], env=env, cwd=ROOT,
                          capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("[Run]") or l.startswith("[Dry run]")), None)
    return dict(script=script, returncode=proc.returncode, line=line)


def test_suites() -> dict:
    result = {}
    for suite, timeout in (("cpu", 900), ("gpu", 900)):
        proc = subprocess.run(["bash", str(ROOT / "scripts" / "run_tests.sh"), suite],
                              cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        tail = [l for l in proc.stdout.strip().splitlines() if "passed" in l or "failed" in l]
        result[suite] = dict(returncode=proc.returncode, summary=tail[-1] if tail else None,
                             log=f"G0/tests_{suite}.log")
        (HERE / f"tests_{suite}.log").write_text(proc.stdout[-20000:] + "\n" + proc.stderr[-20000:])
    return result


def cudnn_version() -> str | None:
    # The runtime cuDNN here is older than the one torch was built against; querying raises
    # even though CUDA kernels work (GPU training/eval already ran on this box). Record it.
    try:
        return str(torch.backends.cudnn.version())
    except RuntimeError as exc:
        return f"unavailable ({exc.__class__.__name__}: {str(exc)[:60]}...)"


def main() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    contract: dict = {"protocol": "docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md section 3 (G0)"}

    contract["git"] = git_state()
    contract["environment"] = dict(
        python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        cudnn=cudnn_version(),
        vector_quantize_pytorch=__import__("importlib.metadata", fromlist=["version"]).version("vector-quantize-pytorch"),
        numpy=np.__version__, fp32_forced=True, tf32_off=True, matmul_precision=torch.get_float32_matmul_precision(),
        device=str(resolve_device("auto")),
    )

    vq_ckpt = torch.load(VQ / "best.pt", map_location="cpu", weights_only=False)
    vq_state = sanitize_state_dict_keys(vq_ckpt["model"])
    vq_state_fp = state_fingerprint(vq_state)
    mean, std = load_norm_stats(VQ / "norm_stats.npz")
    artifacts = artifact_fingerprints(DB, VQ / "best.pt", VQ / "norm_stats.npz")
    contract["vq"] = dict(
        path=str(VQ / "best.pt"), sha256=file_sha256(VQ / "best.pt"),
        epoch=vq_ckpt.get("epoch"), state_sha256=vq_state_fp,
        norm_sha256=file_sha256(VQ / "norm_stats.npz"),
        db_fingerprints={k: v for k, v in artifacts.items() if k != "norm_fingerprint"},
    )

    ckpt = torch.load(CONTROLLER / "best.pt", map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    recorded = args.get("fingerprints") or {}
    cross_checks = dict(
        vq_state_matches=recorded.get("vq_state_sha256") == vq_state_fp,
        db_matches=recorded.get("db_sha256") == artifacts.get("db_sha256"),
        split_matches=recorded.get("split_sha256") == artifacts.get("split_sha256"),
        norm_matches=recorded.get("norm_fingerprint") == artifacts.get("norm_fingerprint"),
    )
    contract["controller"] = dict(
        path=str(CONTROLLER / "best.pt"), sha256=file_sha256(CONTROLLER / "best.pt"),
        epoch=ckpt.get("epoch"), best_metric=ckpt.get("best_metric"),
        training_objective_version=ckpt.get("training_objective_version"),
        recorded_fingerprints=recorded, cross_checks=cross_checks,
        architecture=dict(flow_layers=args.get("flow_layers"), flow_hidden_dim=args.get("flow_hidden_dim"),
                          flow_heads=args.get("flow_heads"), seq_len=args.get("seq_len"),
                          unroll_steps=args.get("unroll_steps"), style_film=args.get("style_film"),
                          history_conditioning=args.get("history_conditioning"),
                          prediction_mode=args.get("prediction_mode"),
                          solver_steps=args.get("solver_steps"), noise_std=args.get("noise_std"),
                          observation_noise_std=args.get("observation_noise_std"),
                          inference_requantize=args.get("inference_requantize"),
                          inference_latent_clip=args.get("inference_latent_clip"),
                          feedback_probability=args.get("feedback_probability"),
                          conditioning_mode=args.get("conditioning_mode")),
    )
    contract["aux_assets"] = dict(
        latent_stats=dict(path=str(LATENT_STATS), sha256=file_sha256(LATENT_STATS)),
        vq_target_cache=dict(path=str(VQ_CACHE), meta=json.loads((VQ_CACHE / "meta.json").read_text()),
                             indices_sha256=file_sha256(VQ_CACHE / "indices.u16")),
        style_prior=dict(path=str(STYLE_PRIOR), sha256=file_sha256(STYLE_PRIOR)),
    )

    contract["test_suites"] = test_suites()

    contract["wrappers"] = {}
    for name, script, env_extra, trainer, prefix in (
        ("vq_entry", "run_train_basic_vqvae_375to291.sh", {},
         "training.train_basic_vqvae_375to291", "[Dry run]"),
        ("controller_obsnoise_tf100", "run_train_flowmatch_history_obsnoise_tf100.sh", {},
         "training.train_flowmatch_vqvae_controller", "[Run]"),
    ):
        run_info = dry_run_wrapper(script, env_extra)
        if run_info["line"] is None:
            contract["wrappers"][name] = {**run_info, "parse": dict(ok=False, error="no command line printed")}
            continue
        contract["wrappers"][name] = {**run_info, "parse": parse_with_trainer(run_info["line"], trainer, prefix)}

    blockers = []
    if contract["git"]["tracked_dirty"]:
        blockers.append("tracked files dirty at G0 time")
    if contract["vq"]["epoch"] != 90:
        blockers.append(f"VQ checkpoint epoch != 90 ({contract['vq']['epoch']})")
    if not all(cross_checks.values()):
        blockers.append(f"controller fingerprints disagree: {cross_checks}")
    if not torch.cuda.is_available():
        blockers.append("no CUDA device")
    for name, info in contract["wrappers"].items():
        if info["returncode"] != 0 or not info["parse"]["ok"]:
            blockers.append(f"wrapper {name} dry-run/parse failed: {info}")
    for suite, info in contract["test_suites"].items():
        if info["returncode"] != 0:
            blockers.append(f"{suite} test suite failed: {info['summary']}")
    contract["already_fixed"] = ALREADY_FIXED
    contract["open_observations"] = OPEN_OBSERVATIONS
    contract["verdict"] = "PASS" if not blockers else "BLOCKED"
    contract["blockers"] = blockers

    (HERE / "contract.json").write_text(json.dumps(contract, indent=1, ensure_ascii=False))

    lines = [
        "# G0 契约核验（只读，不更新任何模型）",
        "",
        f"- 结论：**{contract['verdict']}**" + (f"；blockers：{blockers}" if blockers else "（全部条件满足）"),
        f"- git：`{contract['git']['commit']}`（{contract['git']['branch']}，tracked_dirty={contract['git']['tracked_dirty']}，"
        f"tracked diff sha={contract['git']['tracked_diff_sha256'][:16]}…）",
        f"- VQ：`{VQ.name}/best.pt` epoch={vq_ckpt.get('epoch')}，state sha={vq_state_fp[:16]}…",
        f"- Controller：`{CONTROLLER.name}/best.pt` epoch={ckpt.get('epoch')}，objective v{ckpt.get('training_objective_version')}，"
        f"2L/256dim，solver 4，obs noise 0.5，pure TF（p=0）",
        f"- 指纹交叉核验：{cross_checks}",
        f"- 辅助资产：latent stats sha={contract['aux_assets']['latent_stats']['sha256'][:16]}…，"
        f"VQ target cache indices sha={contract['aux_assets']['vq_target_cache']['indices_sha256'][:16]}…，"
        f"style prior sha={contract['aux_assets']['style_prior']['sha256'][:16]}…",
        f"- 环境：torch {contract['environment']['torch']} / cuda {contract['environment']['cuda']} / "
        f"{contract['environment']['gpu']} / vq-lib {contract['environment']['vector_quantize_pytorch']}；FP32 强制、TF32 off",
        f"- 测试：CPU {contract['test_suites']['cpu']['summary']}；GPU {contract['test_suites']['gpu']['summary']}",
        "",
        "## wrapper DRY_RUN 核验",
        "",
    ]
    for name, info in contract["wrappers"].items():
        lines.append(f"- `{name}`：returncode={info['returncode']}，argparse parse ok={info['parse']['ok']}")
    lines += [
        "",
        "## 已修复（不得重复上报）",
        "",
    ]
    lines += [f"- {item}" for item in ALREADY_FIXED]
    lines += ["", "## 未决观察（不阻塞 G0）", ""]
    lines += [f"- {item}" for item in OPEN_OBSERVATIONS]
    lines += ["", "本文件与 contract.json 在 C0 开跑前冻结。"]
    (HERE / "AUDIT.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in contract.items()
                      if k not in ("vq", "controller")}, indent=1, ensure_ascii=False)[:3500])
    print("[Done]", HERE / "contract.json")


if __name__ == "__main__":
    main()
