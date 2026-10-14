"""Write manifest.json for one B0/B1 run after training (S3 DoD evidence).

Reads the run's own config.json / history.jsonl and computes content hashes for the
dataset, source checkpoint, normalization and the final trained state, so S4 can compare
the two arms without trusting terminal output. Run by scripts/run_train_vq_temporal_ab.sh;
can also be re-run by hand on a finished run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from vq_temporal_common import (  # noqa: E402
    DB, OUT_ROOT, SRC_CKPT, SRC_NORM, environment_report, file_sha256, git_commit, json_dump,
)
from Utils.reproducibility import artifact_fingerprints, state_fingerprint  # noqa: E402
from Utils.runtime import load_checkpoint, sanitize_state_dict_keys  # noqa: E402

PINNED = dict(
    seq_len=8, stride=1, batch_size=512, epochs=10, eval_every=1, save_every=5,
    lr=1e-4, min_lr=1e-5, seed=42, loader_generator_seed=4242,
    recon_weight=3.0, vq_weight=1.0, vel_weight=15.0, root_weight=3.0,
    code_align_weight=2.0, latent_l1_weight=0.01, latent_vel_weight=0.1,
    latent_margin_weight=0.0, roundtrip_weight=0.0, teacher_align_weight=0.0,
    decoder_only=False, amp_dtype="fp32", tf32=False, matmul_precision="highest",
)
EXPECTED_ACC = {"B0": 0.0, "B1": 4.9842}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--arm", choices=("B0", "B1"), required=True)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line]

    mismatches = {k: (config.get(k), v) for k, v in PINNED.items() if config.get(k) != v}
    if mismatches:
        raise RuntimeError(f"run config deviates from the pinned protocol: {mismatches}")
    if float(config.get("acc_weight", -1)) != EXPECTED_ACC[args.arm]:
        raise RuntimeError(
            f"acc_weight={config.get('acc_weight')} does not match arm {args.arm} "
            f"(expected {EXPECTED_ACC[args.arm]})"
        )

    final_path = run_dir / "epoch_0010.pt"
    if not final_path.exists():
        raise FileNotFoundError(f"expected the S3 endpoint checkpoint at {final_path}")
    final_ckpt = load_checkpoint(final_path)
    final_state = sanitize_state_dict_keys(final_ckpt["model"])

    init_ckpt = Path(str(config["init_ckpt"]))
    init_state = sanitize_state_dict_keys(load_checkpoint(init_ckpt)["model"])
    shared_keys = sorted(set(init_state) & set(final_state))

    manifest = dict(
        arm=args.arm, run_dir=str(run_dir), log=None if args.log is None else str(Path(args.log).resolve()),
        git=git_commit(), environment=environment_report(),
        contract=dict(pinned=PINNED, acc_weight_expected=EXPECTED_ACC[args.arm],
                      acc_weight_actual=float(config["acc_weight"])),
        init_checkpoint=dict(path=str(init_ckpt), sha256=file_sha256(init_ckpt),
                             state_sha256=state_fingerprint(init_state)),
        final_checkpoint=dict(path=str(final_path), sha256=file_sha256(final_path),
                              epoch=final_ckpt.get("epoch"),
                              state_sha256=state_fingerprint(final_state)),
        state_change=dict(
            shared_tensor_count=len(shared_keys),
            tensors_changed=sum(
                not torch_equal(init_state[k], final_state[k]) for k in shared_keys
            ),
            quantizer_changed=any(
                not torch_equal(init_state[k], final_state[k])
                for k in shared_keys if k.startswith("quantizer.")
            ),
        ),
        dataset=artifact_fingerprints(DB, init_ckpt, config.get("norm_stats", str(SRC_NORM))),
        norm_stats=dict(path=str(config.get("norm_stats")), sha256=file_sha256(config["norm_stats"])),
        epochs=[
            dict(
                epoch=int(row["epoch"]), time_sec=float(row.get("time_sec", 0.0)),
                num_updates=float(row["train"].get("num_updates", 0.0)),
                num_samples=float(row["train"].get("num_samples", 0.0)),
                train_loss=float(row["train"]["loss"]),
                train_acc=float(row["train"].get("acc_loss", float("nan"))),
                dead_codes_reset=float(row["train"].get("dead_codes_reset", float("nan"))),
                perplexity=float(row["train"].get("perplexity", float("nan"))),
                eval=None if not row.get("eval") else {
                    key: float(row["eval"][key]) for key in (
                        "mpjpe_cm", "local_mpjpe_cm", "jitter_cm", "foot_sliding_cm",
                        "root_ade_cm", "codebook_usage",
                    )},
                best_gate=None if not row.get("best_gate") else bool(row["best_gate"]["accept"]),
            )
            for row in history
        ],
        integrity=dict(
            epochs_recorded=len(history),
            total_updates=sum(float(row["train"].get("num_updates", 0.0)) for row in history),
            final_epoch=int(history[-1]["epoch"]) if history else None,
        ),
    )
    json_dump(run_dir / "manifest.json", manifest)
    print(f"[Manifest {args.arm}] updates={manifest['integrity']['total_updates']:.0f} "
          f"epochs={manifest['integrity']['epochs_recorded']} "
          f"final_state={manifest['final_checkpoint']['state_sha256'][:16]}…")
    print("[Done]", run_dir / "manifest.json")


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


if __name__ == "__main__":
    main()
