"""Efficiency check for the V1 posacc term: forward+backward cost and peak memory.

Bounded timing (10 measured iterations after warmup), no optimizer steps, EMA frozen, state
fingerprints verified. Compares posacc_weight=0 (B0 recipe) against the calibrated lambda.

  python analysis/vq_controller_recovery/V1/v1_posacc_benchmark.py [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "C0"))

import loop_lib as lib  # noqa: E402
from Utils.reproducibility import state_fingerprint  # noqa: E402
from Utils.model_loading import build_vqvae_from_checkpoint  # noqa: E402
from Utils.runtime import load_norm_stats  # noqa: E402

WEIGHTS = dict(recon_type="l1", temporal_loss_type="l1", recon_weight=3.0, vq_weight=1.0,
               vel_weight=15.0, root_weight=3.0, code_align_weight=2.0,
               latent_l1_weight=0.01, latent_vel_weight=0.1,
               latent_margin_weight=0.0, roundtrip_weight=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--posacc-weight", type=float, default=1.6278)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_vqvae_from_checkpoint(lib.VQ / "best.pt", freeze=False).to(device).train()
    model.set_representation_frozen(True)
    forward = model.quantizer.vq.forward
    model.quantizer.vq.forward = lambda *a, **kw: forward(*a, freeze_codebook=True, **kw)
    mean, std = load_norm_stats(lib.VQ / "norm_stats.npz")
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)

    before = state_fingerprint(model.state_dict())
    meta = json.loads((lib.vt.DB / "meta.json").read_text())
    segments = json.loads((lib.vt.DB / "segments.json").read_text())
    x_source = np.memmap(lib.vt.DB / "X.f32", mode="r", dtype=np.float32,
                         shape=(int(meta["num_frames"]), 375))
    train = [s for s in segments if s.get("split") == "train" and not s.get("mirror", False)
             and int(s["num_frames"]) >= args.seq_len]
    rng = np.random.default_rng(4242)
    rows = np.empty((args.batch_size, args.seq_len, 375), dtype=np.float32)
    for j in range(args.batch_size):
        s = train[int(rng.integers(len(train)))]
        start = int(s["frame_start"]) + int(rng.integers(int(s["num_frames"]) - args.seq_len + 1))
        rows[j] = x_source[start:start + args.seq_len]
    x = torch.from_numpy((rows - mean) / std).to(device)

    results = {}
    for label, weight in (("posacc_0", 0.0), ("posacc_lambda", args.posacc_weight)):
        durations = []
        torch.cuda.reset_peak_memory_stats(device)
        for it in range(args.iterations + 3):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model.compute_loss(
                x=x, target_motion=x[..., 84:], posacc_weight=weight, posacc_beta_cm=0.5,
                motion_mean=mean_t, motion_std=std_t, **WEIGHTS)
            out["loss"].backward()
            torch.cuda.synchronize(device)
            if it >= 3:
                durations.append((time.perf_counter() - t0) * 1000.0)
        results[label] = dict(
            mean_ms=float(np.mean(durations)), p95_ms=float(np.percentile(durations, 95)),
            peak_allocated_gib=float(torch.cuda.max_memory_allocated(device) / 2**30),
            posacc_value=float(out["posacc_loss"].detach()),
            posacc_abs_cm=float(out["posacc_abs_cm"].detach()),
        )

    after = state_fingerprint(model.state_dict())
    assert before == after, "benchmark mutated model state"
    overhead = results["posacc_lambda"]["mean_ms"] - results["posacc_0"]["mean_ms"]
    payload = dict(
        protocol="frozen-EMA bounded fwd/bwd timing, no optimizer step, decoder-only freeze",
        batch_size=args.batch_size, seq_len=args.seq_len, iterations=args.iterations,
        posacc_weight=args.posacc_weight, results=results,
        overhead_ms=overhead,
        overhead_ratio=overhead / results["posacc_0"]["mean_ms"],
        state_unchanged=True, device=lib.vt.environment_report(),
    )
    out_path = HERE / "posacc_benchmark.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"[V1 bench] posacc=0: {results['posacc_0']['mean_ms']:.1f}ms  "
          f"posacc={args.posacc_weight}: {results['posacc_lambda']['mean_ms']:.1f}ms  "
          f"overhead={overhead:+.1f}ms ({100*payload['overhead_ratio']:+.1f}%)")
    print("[Done]", out_path)


if __name__ == "__main__":
    main()
