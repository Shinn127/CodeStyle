"""Run the fixed-val12 temporal probe for one VQ checkpoint (S1 baseline / S4 candidates).

Read-only: loads the checkpoint next to which norm_stats.npz must exist, and writes one
JSON. The model state fingerprint is checked before/after; the probe never trains.

  python analysis/vq_temporal_source/run_probe.py \
      --ckpt checkpoints/<run>/best.pt --tag <name> --out analysis/vq_temporal_source/S1/<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vq_temporal_common import full_probe, json_dump  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-style", action="store_true", help="skip the frozen-classifier style proxy")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing probe result: {args.out}")
    payload = full_probe(args.ckpt, device=args.device, style=not args.no_style, tag=args.tag)
    json_dump(args.out, payload)
    agg = payload["aggregate"]
    print(
        f"[Probe {args.tag}] d2r_local per-source mean={agg['d2r_local']['per_source_mean']:.4f} "
        f"p95={agg['d2r_local']['per_source_p95']:.4f} | "
        f"jitter q/c/gt={agg['official_q']['jitter']:.4f}/{agg['official_c']['jitter']:.4f}/"
        f"{agg['official_gt']['jitter']:.4f} | "
        f"fs world={agg['official_q']['fs_world']:.4f} local={agg['official_q']['fs_local']:.4f} | "
        f"mpjpe={agg['official_q']['mpjpe']:.4f} local={agg['official_q']['local_mpjpe']:.4f} "
        f"rootADE={agg['official_q']['root_ade']:.4f} | "
        f"flip={agg['codec_flip_rate']['mean']:.3f} | "
        f"style_acc={payload['style_proxy']['accuracy'] if payload['style_proxy'] else 'n/a'}"
    )
    print("[Done]", args.out)


if __name__ == "__main__":
    main()
