"""Measure VQ code-switch behavior inside the realtime interactive rollout.

Runs the frozen label controller exactly like run_interactive_label_control.sh --headless --demo
(240-frame autoregressive rollout, Euler8, requantize feedback) and records the per-head code
indices of the feedback latent each step. Reports switch rates and the per-frame decoded
motion jump. Optionally reruns with requantize=False (AE feedback) for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Utils.interactive_control import LabelControlSession
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import load_checkpoint, load_norm_stats, resolve_device, configure_torch_runtime, set_seed
from evaluation.simulate_realtime_flowmatch_control import _resolve_base_ckpt_path, _resolve_norm_stats_path, _build_flow_controller

SCRIPT_DIR = ROOT


def build_session(args, device):
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision="highest")
    checkpoint = load_checkpoint(args.flow_ckpt, map_location="cpu")
    config = checkpoint.get("args", {})
    kind, base_path = _resolve_base_ckpt_path(args.vqvae_ckpt, None, flow_args=config, flow_ckpt_path=args.flow_ckpt)
    norm_path = _resolve_norm_stats_path(args.norm_stats, flow_args=config, flow_ckpt_path=args.flow_ckpt, base_ckpt_path=base_path)
    mean, std = load_norm_stats(norm_path)
    names = json.loads((args.db / "StyleNames.json").read_text())
    meta = json.loads((args.db / "meta.json").read_text())
    segments = json.loads((args.db / "segments.json").read_text())
    source = names.index(args.source_style)
    candidates = [s for s in segments if s["label"] == source and s["split"] == "train" and not s.get("mirror", False)]
    seg = min(candidates, key=lambda s: ({"ID": 0, "FW": 1}.get(s["suffix"], 2), s["frame_start"]))
    start = seg["frame_start"] + min(60, seg["num_frames"] - 1)
    data = np.memmap(args.db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))
    initial = np.array(data[start], copy=True)
    del data
    vq = build_vqvae_from_checkpoint(base_path).to(device)
    model = _build_flow_controller(flow_args=config, flow_ckpt=checkpoint, base_model=vq, base_model_type=kind,
                                   num_styles=int(config.get("num_styles", len(names))), device=device)
    steps = int(config.get('solver_steps', 8)) if args.ode_steps is None else int(args.ode_steps)
    if steps < 1:
        raise ValueError('ode_steps must be >= 1')
    session = LabelControlSession(model, initial, mean, std, names, style=source, seed=args.seed,
                                  steps=steps, style_blend=0.25, requantize=args.requantize)
    session.coordinate_convert = False
    return session


def run(args):
    device = resolve_device(args.device)
    set_seed(args.seed)
    session = build_session(args, device)
    quantizer = session.model.vqvae.quantizer
    codes, jumps, per_path = [], [], []
    with torch.no_grad():
        for frame in range(args.max_frames):
            controls = ((1.0, 0.0, 0.0) if frame < 60 else (1.0, 0.0, 1.0) if frame < 120
                        else (0.0, 1.0, 0.0) if frame < 180 else (-1.0, 0.0, -1.0))
            if frame == 80:
                session.set_style((session.style + 1) % len(session.names))
            session.step(*controls, speed=1.2)
            indices = quantizer(session.latent.float()).indices[0].cpu().numpy()
            codes.append(indices)
            per_path.append(np.asarray(session.raw_motion, np.float32))
    codes = np.stack(codes)
    motions = np.stack(per_path)
    changed = codes[1:] != codes[:-1]
    delta = np.linalg.norm(np.diff(motions[:, 3:75].reshape(-1, 24, 3), axis=0), axis=-1).mean(-1) * 100
    idle = ~changed.any(-1)
    local = motions[:, 3:75].reshape(-1, 24, 3)
    if len(local) >= 3:
        second = np.linalg.norm(local[2:] - 2 * local[1:-1] + local[:-2], axis=-1).mean() * 100
    else:
        second = float("nan")
    result = dict(
        checkpoint=str(args.flow_ckpt), frames=int(args.max_frames), requantize=bool(args.requantize), seed=args.seed,
        per_head_switch_rate=float(changed.mean()),
        any_head_switch_rate=float(changed.any(-1).mean()),
        all_head_switch_rate=float(changed.all(-1).mean()),
        distinct_codes_used=int(len(np.unique(codes, axis=0))),
        jump_at_switch_cm=float(delta[~idle].mean()) if (~idle).any() else None,
        jump_steady_cm=float(delta[idle].mean()) if idle.any() else None,
        fraction_steady_frames=float(idle.mean()),
        jitter_proxy_cm=float(second),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    p.add_argument("--flow-ckpt", type=Path, default=SCRIPT_DIR / "checkpoints" / "flowmatch_label_nomirror_150ep" / "best.pt")
    p.add_argument("--vqvae-ckpt", type=Path, default=None)
    p.add_argument("--norm-stats", type=Path, default=None)
    p.add_argument("--source-style", default="Neutral")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ode-steps", type=int, default=None,
                   help="Euler steps; default inherits solver_steps from the checkpoint args")
    p.add_argument("--max-frames", type=int, default=240)
    p.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()
    args.db = args.db.expanduser().resolve()
    args.flow_ckpt = args.flow_ckpt.expanduser().resolve()
    result = run(args)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
        print(f"[Done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
