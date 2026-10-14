"""V1 manifest: per-arm evidence with the E/Q/EMA-frozen contract verified tensor by tensor.

Run by scripts/run_train_vq_posacc_ab.sh after training; also re-runnable by hand.

  python analysis/vq_controller_recovery/V1/v1_finalize_manifest.py \
      --run-dir checkpoints/vq_posacc_V1-B0_ep90seq8_5ep --arm V1-B0 --posacc-weight 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "C0"))

import loop_lib as lib  # noqa: E402
from Utils.reproducibility import file_sha256, state_fingerprint  # noqa: E402
from Utils.runtime import load_checkpoint, sanitize_state_dict_keys  # noqa: E402

# The trainer's decoder-only path zeroes vq/code_align/latent_l1/latent_vel before writing
# config.json (it prints "[Info] decoder-only zeroed ..."), so the pinned values are the
# POST-zeroing ones — verified against a real smoke run's config.json.
PINNED = dict(
    seq_len=8, stride=1, batch_size=512, epochs=5, eval_every=1, save_every=5,
    lr=1e-4, min_lr=1e-5, seed=42, loader_generator_seed=4242,
    recon_weight=3.0, vq_weight=0.0, vel_weight=15.0, root_weight=3.0,
    code_align_weight=0.0, latent_l1_weight=0.0, latent_vel_weight=0.0,
    latent_margin_weight=0.0, roundtrip_weight=0.0, teacher_align_weight=0.0, acc_weight=0.0,
    decoder_only=True, amp_dtype="fp32", tf32=False, matmul_precision="highest",
)
FROZEN_PREFIXES = ("encoder.", "quantizer.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--arm", type=str, required=True)
    ap.add_argument("--posacc-weight", type=float, required=True)
    ap.add_argument("--norm-stats", type=Path, default=lib.VQ / "norm_stats.npz")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line]

    mismatches = {k: (config.get(k), v) for k, v in PINNED.items() if config.get(k) != v}
    if mismatches:
        raise RuntimeError(f"run config deviates from the V1 protocol: {mismatches}")
    if float(config.get("posacc_weight", -1)) != float(args.posacc_weight):
        raise RuntimeError(f"posacc_weight={config.get('posacc_weight')} != expected {args.posacc_weight}")
    if float(config.get("posacc_beta_cm", -1)) != 0.5:
        raise RuntimeError(f"posacc_beta_cm={config.get('posacc_beta_cm')} != 0.5")

    endpoint = run_dir / "epoch_0005.pt"
    if not endpoint.exists():
        raise FileNotFoundError(f"V1 endpoint missing: {endpoint}")
    final_state = sanitize_state_dict_keys(load_checkpoint(endpoint)["model"])
    init_ckpt = Path(str(config["init_ckpt"]))
    init_state = sanitize_state_dict_keys(load_checkpoint(init_ckpt)["model"])

    frozen_report = {}
    for prefix in FROZEN_PREFIXES:
        keys = [k for k in final_state if k.startswith(prefix)]
        unchanged = all(torch.equal(final_state[k], init_state[k]) for k in keys)
        frozen_report[prefix] = dict(tensors=len(keys), unchanged=unchanged)
    if not all(v["unchanged"] for v in frozen_report.values()):
        raise RuntimeError(f"E/Q changed during a decoder-only run: {frozen_report}")
    decoder_changed = sum(
        not torch.equal(final_state[k], init_state[k]) for k in final_state if k.startswith("decoder."))

    manifest = dict(
        stage="V1", arm=args.arm, run_dir=str(run_dir),
        git=lib.vt.git_commit(), environment=lib.vt.environment_report(),
        contract=dict(pinned=PINNED, posacc_weight=float(args.posacc_weight),
                      posacc_beta_cm=0.5),
        init_checkpoint=dict(path=str(init_ckpt), sha256=file_sha256(init_ckpt),
                             state_sha256=state_fingerprint(init_state)),
        final_checkpoint=dict(path=str(endpoint), sha256=file_sha256(endpoint),
                              epoch=load_checkpoint(endpoint).get("epoch"),
                              state_sha256=state_fingerprint(final_state)),
        frozen_groups=frozen_report,
        decoder_tensors_changed=int(decoder_changed),
        norm_stats=dict(path=str(args.norm_stats), sha256=file_sha256(args.norm_stats)),
        epochs=[dict(epoch=int(r["epoch"]), time_sec=float(r.get("time_sec", 0.0)),
                     num_updates=float(r["train"].get("num_updates", 0.0)),
                     train_loss=float(r["train"]["loss"]),
                     train_posacc=float(r["train"].get("posacc_loss", float("nan"))),
                     train_posacc_abs_cm=float(r["train"].get("posacc_abs_cm", float("nan"))),
                     eval=None if not r.get("eval") else {k: float(r["eval"][k]) for k in (
                         "mpjpe_cm", "local_mpjpe_cm", "jitter_cm", "foot_sliding_cm",
                         "root_ade_cm", "codebook_usage")})
                for r in history],
        integrity=dict(epochs_recorded=len(history),
                       total_updates=sum(float(r["train"].get("num_updates", 0.0)) for r in history),
                       final_epoch=int(history[-1]["epoch"]) if history else None),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[V1 manifest {args.arm}] frozen={ {k: v['unchanged'] for k, v in frozen_report.items()} } "
          f"decoder_changed={decoder_changed} updates={manifest['integrity']['total_updates']:.0f}")
    print("[Done]", run_dir / "manifest.json")


if __name__ == "__main__":
    main()
