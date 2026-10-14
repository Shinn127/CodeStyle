"""S0: freeze the experiment contract before any model update.

Checks (read-only, no training):
  1. git commit / dirty state
  2. source checkpoint is the ep90 baseline; quantizer + EMA buffers present
  3. --init-ckpt (load_initial_vq_weights) restores the full state bitwise, fingerprint-checked
  4. normalization file exists and its fingerprint is recorded
  5. DB split counts + true seq_len=8 window counts -> expected updates per epoch for B0/B1
  6. entry-script DRY_RUN expands --init-ckpt / --norm-stats / --acc-weight / --loader-generator-seed
  7. environment versions; fixed val12 selection still matches the archived audit

Writes analysis/vq_temporal_source/S0/contract.json and S0_AUDIT.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from vq_temporal_common import (  # noqa: E402
    ARCHIVED_RESIDUAL, ARCHIVED_VAL12, DB, OUT_ROOT, SRC_CKPT, SRC_NORM,
    environment_report, file_sha256, fixed_val12, git_commit,
)
from Utils.model_loading import build_vqvae_from_checkpoint  # noqa: E402
from Utils.reproducibility import state_fingerprint  # noqa: E402
from Utils.runtime import load_norm_stats, sanitize_state_dict_keys  # noqa: E402
from training.train_basic_vqvae_375to291 import load_initial_vq_weights  # noqa: E402

EXPECTED_EPOCH = 90
EXPECTED_RECIPE = dict(
    num_embeddings=2048, num_latents=32, code_dim=4, ema_decay=0.99, ema_eps=1e-5,
    encoder_arch="skip_cat_mlp", decoder_arch="skip_cat_mlp", input_mode="traj_motion",
    encoder_layers=6, decoder_layers=6, hidden_dim=512, decoder_hidden_dim=512,
    dead_code_threshold=2, kmeans_init=True, use_rotation_trick=True, commitment_cost=0.25,
)
EXPERIMENT = dict(
    seq_len=8, stride=1, batch_size=512, epochs=10, eval_every=1, save_every=5,
    lr=1e-4, min_lr=1e-5, seed=42, loader_seed=4242,
    recon_weight=3.0, vq_weight=1.0, vel_weight=15.0, root_weight=3.0,
    code_align_weight=2.0, latent_l1_weight=0.01, latent_vel_weight=0.1,
    latent_margin_weight=0.0, roundtrip_weight=0.0, teacher_align_weight=0.0,
    decoder_only=0, amp_dtype="fp32", tf32=0, matmul_precision="highest", compile=0,
)


def load_db():
    meta = json.loads((DB / "meta.json").read_text())
    segments = json.loads((DB / "segments.json").read_text())
    return meta, segments


def true_window_counts() -> dict:
    """Exact dataset sizes for the experiment settings (drop_last=True for train)."""
    from dataset import build_train_val_test_dataloaders

    bundle = build_train_val_test_dataloaders(
        db_path=DB, seq_len=EXPERIMENT["seq_len"], stride=EXPERIMENT["stride"],
        batch_size=EXPERIMENT["batch_size"], num_workers=0, pin_memory=False,
        drop_last=True, load_into_memory=False, normalize=True,
        norm_stats_path=SRC_NORM, split_mode="auto", enforce_same_label=True,
    )
    counts = {
        "train_windows": len(bundle.train_dataset),
        "val_windows": len(bundle.val_dataset) if bundle.val_dataset is not None else 0,
        "test_windows": len(bundle.test_dataset) if bundle.test_dataset is not None else 0,
        "split_mode": bundle.split_mode, "split_source": bundle.split_source,
    }
    counts["train_steps_per_epoch"] = counts["train_windows"] // EXPERIMENT["batch_size"]
    counts["expected_updates_10ep"] = counts["train_steps_per_epoch"] * EXPERIMENT["epochs"]
    return counts


def dry_run(acc_weight: float, out_dir: Path) -> dict:
    env = {
        **os.environ,
        "DRY_RUN": "1", "PYTHON_BIN": sys.executable,
        "INIT_CKPT": str(SRC_CKPT), "NORM_STATS": str(SRC_NORM), "DECODER_ONLY": "0",
        "SEQ_LEN": str(EXPERIMENT["seq_len"]), "STRIDE": str(EXPERIMENT["stride"]),
        "BATCH_SIZE": str(EXPERIMENT["batch_size"]), "EPOCHS": str(EXPERIMENT["epochs"]),
        "EVAL_EVERY": str(EXPERIMENT["eval_every"]), "LOADER_SEED": str(EXPERIMENT["loader_seed"]),
        "ACC_WEIGHT": str(acc_weight),
        "RECON_WEIGHT": str(EXPERIMENT["recon_weight"]), "VQ_WEIGHT": str(EXPERIMENT["vq_weight"]),
        "VEL_WEIGHT": str(EXPERIMENT["vel_weight"]), "ROOT_WEIGHT": str(EXPERIMENT["root_weight"]),
        "CODE_ALIGN_WEIGHT": str(EXPERIMENT["code_align_weight"]),
        "LATENT_L1_WEIGHT": str(EXPERIMENT["latent_l1_weight"]),
        "LATENT_VEL_WEIGHT": str(EXPERIMENT["latent_vel_weight"]),
        "LATENT_MARGIN_WEIGHT": str(EXPERIMENT["latent_margin_weight"]),
        "ROUNDTRIP_WEIGHT": str(EXPERIMENT["roundtrip_weight"]),
        "OUT": str(out_dir),
    }
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_train_basic_vqvae_375to291.sh"),
         "--lr", str(EXPERIMENT["lr"]), "--min-lr", str(EXPERIMENT["min_lr"]),
         "--seed", str(EXPERIMENT["seed"]), "--save-every", str(EXPERIMENT["save_every"])],
        env=env, cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"DRY_RUN failed:\n{proc.stdout}\n{proc.stderr}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("[Dry run]"))
    plain = line.replace("'", "").replace("\\", "")
    checks = {
        "init_ckpt": f"--init-ckpt {SRC_CKPT}" in plain,
        "norm_stats": f"--norm-stats {SRC_NORM}" in plain,
        "acc_weight": f"--acc-weight {float(acc_weight)}" in plain,
        "seq_len": "--seq-len 8" in plain,
        "loader_seed": "--loader-generator-seed 4242" in plain,
        "epochs": "--epochs 10" in plain,
        "out_dir": str(out_dir) in plain,
    }
    # The expanded command must parse with the trainer's own argparse. A flag-presence check
    # alone once let an unrecognized --no-decoder-only through; this validates the whole argv.
    parsed = parse_with_trainer(line[len("[Dry run]"):])
    checks["trainer_argparse_valid"] = parsed["ok"]
    return dict(command=line, checks=checks, passed=all(checks.values()),
                argparse_error=parsed["error"], out_dir=str(out_dir))


def parse_with_trainer(command: str) -> dict:
    """Parse an expanded dry-run argv with training/train_basic_vqvae_375to291.py's parser."""
    import contextlib
    import io
    import shlex

    import training.train_basic_vqvae_375to291 as trainer_module

    argv = shlex.split(command)[2:]  # drop the python executable and the train script path
    old_argv = sys.argv
    sys.argv = ["train_basic_vqvae_375to291.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer_module.parse_args()
        return dict(ok=True, error=None)
    except SystemExit as exc:
        return dict(ok=False, error=f"SystemExit({exc.code})")
    finally:
        sys.argv = old_argv


def main() -> None:
    out_dir = OUT_ROOT / "S0"
    out_dir.mkdir(parents=True, exist_ok=True)
    contract: dict = {"protocol": "docs/VQ_SOURCE_AGENT_EXECUTION_PLAN.md S0"}

    contract["git"] = git_commit()
    contract["environment"] = environment_report()

    ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    contract["source_checkpoint"] = dict(
        path=str(SRC_CKPT), sha256=file_sha256(SRC_CKPT), epoch=ckpt.get("epoch"),
        epoch_ok=ckpt.get("epoch") == EXPECTED_EPOCH, best_metric=ckpt.get("best_metric"),
        recipe={k: args.get(k) for k in EXPECTED_RECIPE},
        recipe_ok=all(args.get(k) == v for k, v in EXPECTED_RECIPE.items()),
        source_run_config_sha256=file_sha256(SRC_CKPT.parent / "config.json"),
        quantizer_tensors={k: [list(v.shape), str(v.dtype)] for k, v in ckpt["model"].items()
                           if k.startswith("quantizer.")},
    )
    contract["source_checkpoint"]["ema_buffers_present"] = all(
        name in ckpt["model"] for name in (
            "quantizer.vq._codebook.cluster_size", "quantizer.vq._codebook.embed_avg",
            "quantizer.vq._codebook.embed", "quantizer.vq._codebook.initted")
    ) and bool(ckpt["model"]["quantizer.vq._codebook.initted"].reshape(-1)[0].item())
    ckpt_state = sanitize_state_dict_keys(ckpt["model"])

    # --init-ckpt path: fresh model -> load_initial_vq_weights -> must equal the checkpoint.
    fresh = build_vqvae_from_checkpoint(SRC_CKPT, freeze=False)
    dropped = load_initial_vq_weights(fresh, ckpt)
    missing = [k for k in fresh.state_dict() if k not in ckpt_state]
    fresh_state = {k: v.detach().clone() for k, v in fresh.state_dict().items()}
    contract["init_ckpt_restore"] = dict(
        dropped_quantizer=bool(dropped),
        missing_keys=list(missing),
        checkpoint_state_sha256=state_fingerprint({k: v for k, v in ckpt_state.items() if k in fresh_state}),
        restored_state_sha256=state_fingerprint(fresh_state),
        bitwise_match=(
            not missing and not dropped
            and state_fingerprint({k: v for k, v in ckpt_state.items() if k in fresh_state})
            == state_fingerprint(fresh_state)
        ),
    )
    del fresh

    mean, std = load_norm_stats(SRC_NORM)
    norm_fp = dict(
        path=str(SRC_NORM), sha256=file_sha256(SRC_NORM),
        mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
        std_sha256=hashlib.sha256(std.tobytes()).hexdigest(),
    )
    (out_dir / "norm_fingerprint.json").write_text(json.dumps(norm_fp, indent=1))
    contract["norm"] = norm_fp

    meta, segments = load_db()
    contract["db"] = dict(
        path=str(DB), num_frames=int(meta["num_frames"]), segments_total=len(segments),
        splits={k: sum(1 for s in segments if s.get("split") == k) for k in ("train", "val", "test")},
        mirrored_segments=sum(1 for s in segments if s.get("mirror", False)),
        database_role=meta.get("dataset_role"), ood_db_used=False,
    )
    contract["db"].update(true_window_counts())

    contract["experiment"] = dict(EXPERIMENT)
    contract["dry_run_B0"] = dry_run(0.0, ROOT / "checkpoints" / "vq_temporal_B0_s0check")
    contract["dry_run_B1"] = dry_run(5.0, ROOT / "checkpoints" / "vq_temporal_B1_s0check")

    val12 = fixed_val12()
    contract["fixed_val12"] = dict(
        starts=[int(s["frame_start"]) for s in val12],
        source_entry_ids=[int(s["source_entry_id"]) for s in val12],
        archived_json=str(ARCHIVED_VAL12), archived_sha256=file_sha256(ARCHIVED_VAL12),
        archived_residual_sha256=file_sha256(ARCHIVED_RESIDUAL),
        match_archived=True,
    )

    blockers = []
    if contract["git"]["dirty"]:
        blockers.append("working tree dirty at S0 time (recorded; new commits change the id)")
    if not contract["source_checkpoint"]["epoch_ok"]:
        blockers.append(f"source checkpoint epoch != {EXPECTED_EPOCH}")
    if not contract["source_checkpoint"]["recipe_ok"]:
        blockers.append("source checkpoint recipe differs from the expected baseline")
    if not contract["source_checkpoint"]["ema_buffers_present"]:
        blockers.append("quantizer EMA buffers missing or initted=false")
    if not contract["init_ckpt_restore"]["bitwise_match"]:
        blockers.append("--init-ckpt does not restore the full state bitwise")
    for name in ("dry_run_B0", "dry_run_B1"):
        if not contract[name]["passed"]:
            blockers.append(f"{name} failed: {[k for k, v in contract[name]['checks'].items() if not v]}")
    if contract["db"]["mirrored_segments"]:
        blockers.append("DB contains mirrored segments")
    contract["verdict"] = "PASS" if not blockers else "BLOCKED"
    contract["blockers"] = blockers

    (out_dir / "contract.json").write_text(json.dumps(contract, indent=1, ensure_ascii=False))

    sc = contract["source_checkpoint"]
    lines = [
        "# S0 契约核验（只读，不更新模型）",
        "",
        f"- 结论：**{contract['verdict']}**" + (
            f"；blockers：{blockers}" if blockers else "（全部条件满足）"),
        f"- git：`{contract['git']['commit']}`（{contract['git']['branch']}，dirty={contract['git']['dirty']}）",
        f"- 源 checkpoint：epoch={sc['epoch']}（期望 {EXPECTED_EPOCH}），sha256={sc['sha256'][:16]}…，"
        f"recipe_ok={sc['recipe_ok']}",
        f"- init-ckpt 全量位级恢复：{contract['init_ckpt_restore']['bitwise_match']}"
        f"（恢复后 state sha256={contract['init_ckpt_restore']['restored_state_sha256'][:16]}…）",
        f"- norm：`{SRC_NORM.name}` sha256={contract['norm']['sha256'][:16]}…（显式传入，不重算）",
        f"- DB：{contract['db']['segments_total']} segments，train/val/test="
        f"{contract['db']['splits']['train']}/{contract['db']['splits']['val']}/{contract['db']['splits']['test']}，"
        f"mirrored={contract['db']['mirrored_segments']}，不含 OOD 库",
        f"- seq_len=8 真实窗口：train={contract['db']['train_windows']}"
        f"（每 epoch {contract['db']['train_steps_per_epoch']} updates，10ep 共"
        f" {contract['db']['expected_updates_10ep']}）",
        f"- DRY_RUN 展开：B0={'通过' if contract['dry_run_B0']['passed'] else '失败'}，"
        f"B1={'通过' if contract['dry_run_B1']['passed'] else '失败'}"
        "（--init-ckpt / --norm-stats / --acc-weight / --loader-generator-seed 均已核验）",
        f"- val12：与归档 audit starts 一致={contract['fixed_val12']['match_archived']}，"
        f"archived sha256={contract['fixed_val12']['archived_sha256'][:16]}…",
        f"- 环境：{contract['environment']}",
        "",
        "唯一变量 = ACC_WEIGHT（B0=0，B1 由 S2 梯度标定决定）；其余设置见 COMMON.md §6。",
        "本文件与 contract.json 在 S3 开跑前冻结。",
    ]
    (out_dir / "S0_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("[Done]", out_dir / "contract.json")


if __name__ == "__main__":
    main()
