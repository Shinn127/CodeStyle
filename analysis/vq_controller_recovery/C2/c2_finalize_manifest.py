"""C2 arm manifest: contract evidence for the pure-TF observation-noise A/B.

  python analysis/vq_controller_recovery/C2/c2_finalize_manifest.py \
      --run-dir checkpoints/c2_obsnoise_A025_10ep --arm C2-A025 --obs-noise-std 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "C0"))

import loop_lib as lib  # noqa: E402
from Utils.reproducibility import file_sha256, state_fingerprint  # noqa: E402
from Utils.runtime import load_checkpoint, sanitize_state_dict_keys  # noqa: E402

INIT_CONTROLLER = lib.ROOT / "checkpoints" / "flowmatch_label_history_obsnoise_tf100" / "best.pt"
PINNED = dict(
    seq_len=6, epochs=10, batch_size=512, lr=5e-4, seed=42,
    feedback_probability=0.0, history_conditioning=True,
    prediction_mode="target", solver_steps=4, flow_layers=2,
)
# Keys allowed to differ between arms (the single experimental variable).
CONFIG_DIFF_ALLOWED = frozenset({"out_dir", "observation_noise_std"})


def _epoch_record(row: dict) -> dict:
    """history.jsonl mixes training rows and rollout-validation rows (same epoch key)."""
    if "train" in row:
        return dict(kind="train", epoch=int(row["epoch"]), time_sec=float(row.get("time_sec", 0.0)),
                    num_batches=float(row.get("train_num_batches", 0.0)),
                    train_loss=float(row["train"]["loss"]),
                    val_loss=float(row["val"]["loss"]) if row.get("val") else None,
                    noise_coeff=float(row["train"].get("observation_noise_coeff_mean", float("nan"))))
    rollout = row.get("rollout_style_val") or {}
    keep = ("matched_accuracy", "cross_accuracy", "matched_root_ade_cm", "cross_root_ade_cm",
            "jitter_cm", "foot_sliding_local_cm", "foot_sliding_world_cm")
    return dict(kind="rollout_val", epoch=int(row["epoch"]),
                selection_key=row.get("selection_key"),
                **{k: (float(rollout[k]) if rollout.get(k) is not None else None) for k in keep})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--obs-noise-std", type=float, required=True)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line]

    mismatches = {k: (config.get(k), v) for k, v in PINNED.items() if config.get(k) != v}
    if mismatches:
        raise RuntimeError(f"C2 arm deviates from the pinned protocol: {mismatches}")
    if float(config.get("observation_noise_std", -1)) != float(args.obs_noise_std):
        raise RuntimeError(f"observation_noise_std={config.get('observation_noise_std')} "
                           f"!= expected {args.obs_noise_std}")

    endpoint = run_dir / "epoch_0010.pt"
    if not endpoint.exists():
        raise FileNotFoundError(f"C2 endpoint missing: {endpoint}")
    final_state = sanitize_state_dict_keys(load_checkpoint(endpoint)["model"])
    init_state = sanitize_state_dict_keys(load_checkpoint(Path(str(config["init_ckpt"])))["model"])
    init_sha = state_fingerprint({k: v for k, v in init_state.items() if k in final_state})

    vq_state = state_fingerprint(load_checkpoint(lib.VQ / "best.pt")["model"])

    manifest = dict(
        stage="C2a", arm=args.arm, run_dir=str(run_dir),
        git=lib.vt.git_commit(), environment=lib.vt.environment_report(),
        contract=dict(pinned=PINNED, observation_noise_std=float(args.obs_noise_std),
                      allowed_config_diff=sorted(CONFIG_DIFF_ALLOWED)),
        init_controller=dict(path=str(INIT_CONTROLLER), sha256=file_sha256(INIT_CONTROLLER),
                             state_sha256=init_sha),
        vq_state_sha256=vq_state,
        final_checkpoint=dict(path=str(endpoint), sha256=file_sha256(endpoint),
                              epoch=load_checkpoint(endpoint).get("epoch"),
                              state_sha256=state_fingerprint(final_state)),
        epochs=[_epoch_record(r) for r in history],
        integrity=dict(epochs_recorded=sum(1 for r in history if "train" in r),
                       # the controller trainer reports batches; updates == batches at bs512
                       total_updates=sum(float(r.get("train_num_batches", 0.0))
                                         for r in history if "train" in r),
                       rollout_validations=sum(1 for r in history if "rollout_style_val" in r),
                       final_epoch=int(history[-1]["epoch"]) if history else None),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[C2 manifest {args.arm}] obs_noise={args.obs_noise_std} "
          f"updates={manifest['integrity']['total_updates']:.0f} "
          f"init_sha={init_sha[:12]}…")
    print("[Done]", run_dir / "manifest.json")


if __name__ == "__main__":
    main()
