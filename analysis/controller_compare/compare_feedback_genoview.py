"""Side-by-side GenoView demo: clip->quantize->decode vs continuous AE feedback.

Runs the scripted keyboard sequence (idle / walk / walk+turn / walk / strafe) twice through
the exact production label-control session. Left character = formal quantized feedback
(clip -> snap to VQ codebook -> decode), right character = --no-requantize continuous
feedback. Same seed, same controls, same Euler steps and the same initial frame, so the
ONLY difference in the loop is the requantization step.

Usage:
  python analysis/controller_compare/compare_feedback_genoview.py \
      --flow-ckpt checkpoints/flowmatch_label_history_ode4_100ep/latest.pt \
      --noise-std 0.25 --frames 300
  # re-show a saved pair without re-running inference:
  python analysis/controller_compare/compare_feedback_genoview.py --load-npz <path.npz>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB = ROOT / "database_100sty_375_memmap_nomirror"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-ckpt", type=Path, default=ROOT / "checkpoints/flowmatch_label_history_ode4_100ep/latest.pt")
    p.add_argument("--style", default="Neutral")
    p.add_argument("--frames", type=int, default=300, help="generated frames per character (plus 1 seed frame)")
    p.add_argument("--noise-std", type=float, default=0.25,
                   help="sampling noise; 0.25 shows the quantized-vs-continuous difference most clearly")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--ode-steps", type=int, default=None, help="default inherits the checkpoint solver_steps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gap", type=float, default=1.8, help="world-space distance between the two characters")
    p.add_argument("--save-npz", type=Path, default=None, help="default: analysis/controller_compare/feedback_compare/<tag>.npz")
    p.add_argument("--load-npz", type=Path, default=None, help="re-show a saved pair; skips inference")
    p.add_argument("--no-show", action="store_true", help="record/save only, do not open the window")
    # Captured stills are more useful than transient motion for spotted comparisons.
    p.add_argument("--screenshot-dir", type=Path, default=None,
                   help="write PNG stills at fixed frames (idle/walk/turn) while the window plays")
    p.add_argument("--screenshot-frames", type=int, nargs="+", default=[40, 90, 150, 210, 270])
    return p.parse_args()


def build_session(args, *, requantize: bool):
    from Visualization.cli.interactive_label_control import load_session
    ns = SimpleNamespace(
        db=DB, flow_ckpt=args.flow_ckpt, vqvae_ckpt=None, norm_stats=None,
        device=args.device, style=args.style, source_style=None, seed=args.seed,
        ode_steps=args.ode_steps, noise_std=args.noise_std, style_blend=0.0,
        requantize=requantize, turn_speed=90.0, smooth_plan=False)
    return load_session(ns)


def record(args, *, requantize: bool):
    """Play the scripted demo and collect the rendered 375 sequence + world poses."""
    from analysis.motion_quality_diagnosis.record_control_video import control_for
    session, info = build_session(args, requantize=requantize)
    seq, positions, rotations = [], [], []
    t0 = time.perf_counter()
    for t in range(args.frames + 1):
        if t > 0:
            session.step(*control_for(t - 1), speed=args.speed)
        seq.append(np.concatenate((session.trajectory, session.raw_motion)).astype(np.float32))
        pos, rot = session.selected_pose()
        positions.append(np.asarray(pos[0], np.float32))
        rotations.append(np.asarray(rot[0], np.float32))
    wall = time.perf_counter() - t0
    return dict(seq=np.stack(seq), positions=np.stack(positions), rotations=np.stack(rotations),
                info=info, wall=wall)


def local_jitter_cm(positions: np.ndarray, lo: int = 31, hi: int = 151) -> float:
    hips = positions[:, 0, :]
    local = positions - hips[:, None, :]
    jit = np.linalg.norm(np.diff(local, n=2, axis=0), axis=-1).mean(-1) * 100.0
    return float(jit[lo:hi].mean())


def report(name, rec):
    from analysis.motion_quality_diagnosis.record_control_video import analyze
    m = analyze(rec["positions"], name=name)["summary"]
    print(f"[{name}] steps={rec['info']['steps']} feedback={rec['info']['feedback']} "
          f"wall={rec['wall']:.1f}s glide={m['glide_fraction_locomotion']:.4f} "
          f"steps L/R={m['left_steps']}/{m['right_steps']} "
          f"walk-jitter(2nd diff)={local_jitter_cm(rec['positions']):.3f}cm", flush=True)
    return m


def main():
    args = parse_args()
    if args.load_npz is None:
        args.flow_ckpt = args.flow_ckpt.expanduser().resolve()
        if not args.flow_ckpt.is_file():
            raise FileNotFoundError(f"Flow controller checkpoint not found: {args.flow_ckpt}")
        print(f"[Record] quantized (formal) run ...", flush=True)
        rq = record(args, requantize=True)
        print(f"[Record] continuous AE run ...", flush=True)
        ae = record(args, requantize=False)
        report("RQ", rq)
        report("AE", ae)
        save = args.save_npz or (ROOT / "analysis/controller_compare/feedback_compare" /
                                 f"{args.flow_ckpt.parent.name}_ns{args.noise_std:g}.npz")
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, rq_seq=rq["seq"], ae_seq=ae["seq"],
                            rq_pos=rq["positions"], ae_pos=ae["positions"],
                            rq_rot=rq["rotations"], ae_rot=ae["rotations"],
                            noise_std=args.noise_std, seed=args.seed, frames=args.frames,
                            steps=rq["info"]["steps"], speed=args.speed, style=args.style)
        print(f"[Saved] {save}", flush=True)
        left_seq, right_seq, steps = rq["seq"], ae["seq"], int(rq["info"]["steps"])
    else:
        data = np.load(args.load_npz, allow_pickle=False)
        left_seq, right_seq = data["rq_seq"], data["ae_seq"]
        args.noise_std = float(data["noise_std"]); args.frames = int(data["frames"])
        steps = int(data["steps"]) if "steps" in data else 4
        print(f"[Loaded] {args.load_npz} ({left_seq.shape[0]} frames, std={args.noise_std:g}, Euler{steps})", flush=True)

    if args.no_show:
        return

    from Visualization.backends.genoview_backend import visualize_dual_motion375_genoview_from_375
    model_name = args.flow_ckpt.parent.name.removeprefix("flowmatch_label_")
    shots = None
    if args.screenshot_dir is not None:
        n = int(left_seq.shape[0])
        shots = {int(f) % n: (args.screenshot_dir / f"frame_{int(f):03d}.png").resolve()
                 for f in args.screenshot_frames}
    visualize_dual_motion375_genoview_from_375(
        ref_features_375=left_seq, pred_features_375=right_seq, db_path=DB,
        fps=60, dt=1 / 60, use_global=True,
        title=f"{model_name} | LEFT: clip->quantize->decode (formal) | RIGHT: continuous AE feedback "
              f"| Euler{steps} | noise std {args.noise_std:g} | same seed & controls",
        model_gap=args.gap, screenshots=shots)


if __name__ == "__main__":
    main()
