"""Offline record + analyze keyboard-control locomotion: gliding vs proper stepping.

Plays a scripted control sequence (idle / walk / walk+turn / walk / strafe) through the exact
production label-controller session (same code path as the live window, no rendering needed),
records per-frame world joint positions, then:

  * renders an MP4: left = raw model motion, right = display post-processing (smoothing+foot IK)
  * quantifies "static-pose translation" (root moves while the local pose is frozen) and
    foot behaviour (contact fraction, stance sliding speed, swing lift height, step count)
  * compares a checkpoint against a GT reference walk from the database (same fold of metrics)

Examples:
  python analysis/motion_quality_diagnosis/record_control_video.py \
      --flow-ckpt checkpoints/<your_run>/best.pt --tag myrun \
      --style Neutral --frames 300 --noise-std 0.25 --render dual
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

FPS = 60
FOOT_IDS = {"right": (18, 19), "left": (22, 23)}  # RightFoot/ToeBase, LeftFoot/ToeBase
PHASES = (("idle", 0, 30), ("walk", 30, 150), ("turn", 150, 210), ("walk2", 210, 270), ("strafe", 270, 300))


def control_for(t: int) -> tuple[float, float, float]:
    """Scripted keyboard intent: (forward, side, turn). speed is fixed by the planner."""
    if t < 30:
        return 0.0, 0.0, 0.0
    if t < 150:
        return 1.0, 0.0, 0.0
    if t < 210:
        return 1.0, 0.0, 1.0
    if t < 270:
        return 1.0, 0.0, 0.0
    return 0.0, 1.0, 0.0


def record_session(flow_ckpt: Path, *, style: str | None, frames: int, noise_std: float,
                   seed: int, speed: float, device: str, ode_steps: int | None = None,
                   requantize: bool = True):
    from Visualization.cli.interactive_label_control import load_session
    ns = SimpleNamespace(
        db=ROOT / "database_100sty_375_memmap_nomirror", flow_ckpt=flow_ckpt,
        vqvae_ckpt=None, norm_stats=None, device=device, style=style, source_style=None,
        seed=seed, ode_steps=ode_steps, noise_std=noise_std, style_blend=0.25, requantize=requantize,
        turn_speed=90.0)
    session, info = load_session(ns)
    positions, rotations, raw, roots = [], [], [], []
    t0 = time.perf_counter()
    for t in range(frames):
        forward, side, turn = control_for(t)
        session.step(forward, side, turn, speed=speed)
        pos, rot = session.selected_pose()
        positions.append(np.asarray(pos[0], np.float32))
        rotations.append(np.asarray(rot[0], np.float32))
        raw.append(np.asarray(session.raw_motion, np.float32))
        roots.append(np.asarray(session.planner.position, np.float32))
    wall = time.perf_counter() - t0
    return dict(positions=np.stack(positions), rotations=np.stack(rotations),
                raw=np.stack(raw), roots=np.stack(roots), info=info, wall=wall)


def gt_reference(style: str, frames: int, seed: int):
    from Utils.motion375 import reconstruct_selected_globals_from_375
    db = ROOT / "database_100sty_375_memmap_nomirror"
    names = json.loads((db / "StyleNames.json").read_text())
    meta = json.loads((db / "meta.json").read_text())
    segments = json.loads((db / "segments.json").read_text())
    label = names.index(style)
    pool = [s for s in segments if s["label"] == label and s["num_frames"] >= frames + 1]
    pool = sorted(pool, key=lambda s: ({"FW": 0, "ID": 1}.get(s["suffix"], 2), -s["num_frames"]))
    seg = pool[0]
    start = seg["frame_start"] + max(0, (seg["num_frames"] - frames) // 2)
    x = np.memmap(db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))
    raw = np.asarray(x[start : start + frames], np.float32)
    pos, rot = reconstruct_selected_globals_from_375(raw, dt=1 / FPS, root_wy_in_degrees=True, use_global=True)
    return dict(positions=np.asarray(pos, np.float32), rotations=np.asarray(rot, np.float32),
                raw=raw, suffix=seg["suffix"], segment=seg["source_entry_id"])


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------

def _foot_series(positions: np.ndarray, side: str) -> np.ndarray:
    a, b = FOOT_IDS[side]
    return positions[:, [a, b], :].mean(axis=1)  # (T, 3)


def analyze(positions: np.ndarray, *, name: str, phases=PHASES) -> dict:
    """Gliding + foot-contact metrics on world joint positions [T,24,3]."""
    T = positions.shape[0]
    hips = positions[:, 0, :]
    local = positions - hips[:, None, :]
    pose_delta = np.linalg.norm(np.diff(local, axis=0), axis=-1)[:, 1:].mean(axis=1) * 100.0  # cm/frame
    root_speed = np.linalg.norm(np.diff(hips[:, [0, 2]], axis=0), axis=-1) * FPS  # m/s

    feet, out_phases = {}, {}
    for side in FOOT_IDS:
        f = _foot_series(positions, side)
        height = f[:, 1]
        base = float(np.percentile(height, 5))
        contact = height < base + 0.02
        horiz = np.linalg.norm(np.diff(f[:, [0, 2]], axis=0), axis=-1) * 100.0 * FPS  # cm/s
        feet[side] = dict(height=height, base=base, contact=contact, horiz=horiz,
                          lift=height - base)

    def steps_of(lift, contact):
        count = 0
        armed = False
        for i, h in enumerate(lift):
            if not contact[i]:
                if not armed and h > 0.04:
                    count += 1
                    armed = True
            else:
                armed = False
        return count

    for pname, a, b in phases:
        b = min(b, T)
        if b - a < 2:
            continue
        rs = root_speed[a : b - 1]
        pd = pose_delta[a : b - 1]
        phase = dict(
            frames=int(b - a), root_speed_mean=float(rs.mean()),
            pose_delta_mean=float(pd.mean()),
            glide_fraction=float(((rs > 0.4) & (pd < 0.3)).mean()),
        )
        for side in FOOT_IDS:
            f = feet[side]
            c = f["contact"][a:b]
            hz = f["horiz"][a : b - 1]
            stance = c[:-1] & c[1:]
            phase[f"{side}_contact_fraction"] = float(c.mean())
            phase[f"{side}_stance_speed_cm_s"] = float(hz[stance].mean()) if stance.any() else float("nan")
            phase[f"{side}_swing_lift_cm"] = float(np.percentile(f["lift"][a:b], 95) * 100.0)
            phase[f"{side}_steps"] = int(steps_of(f["lift"][a:b], c))
        out_phases[pname] = phase

    # whole-sequence summary + longest glide streak over locomotion phases
    loc = np.zeros(T - 1, bool)
    for pname, a, b in phases:
        if pname in ("walk", "turn", "walk2"):
            loc[a : min(b - 1, T - 1)] = True
    glide = (root_speed > 0.4) & (pose_delta < 0.3)
    streak = best = 0
    for i in range(len(glide)):
        if glide[i]:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    summary = dict(
        name=name, frames=int(T),
        root_speed_mean=float(root_speed.mean()),
        pose_delta_mean=float(pose_delta.mean()),
        glide_fraction_locomotion=float(glide[loc].mean()) if loc.any() else float("nan"),
        glide_longest_streak_frames=int(best),
        right_steps=int(sum(p["right_steps"] for p in out_phases.values())),
        left_steps=int(sum(p["left_steps"] for p in out_phases.values())),
    )
    return dict(summary=summary, phases=out_phases)


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------

def _prepare_template():
    from Visualization.io.genoview_bvh import (
        _load_template, _template_local_transforms, _expand_selected_to_full, GENOVIEW_RESOURCES_DIR)
    template = _load_template(GENOVIEW_RESOURCES_DIR / "Geno_bind.bvh")
    parents = np.asarray(template["parents"], np.int32)
    local_pos, local_rot = _template_local_transforms(template, 0)
    return template, parents, local_pos, local_rot


def expand_sequence(positions24, rotations24):
    """Deterministic selected-24 -> full-Geno expansion; no display post-processing."""
    from Visualization.io.genoview_bvh import _expand_selected_to_full
    template, parents, local_pos, local_rot = _prepare_template()
    out = []
    for t in range(positions24.shape[0]):
        pos, _ = _expand_selected_to_full(
            positions24[t : t + 1], rotations24[t : t + 1], full_names=template["names"],
            full_parents=parents, template_local_pos=local_pos, template_local_rot=local_rot)
        out.append(np.asarray(pos[0], np.float32))
    return np.stack(out)


def postprocess_sequence(positions24, rotations24, *, cutoff=6.0, foot_ik=True):
    from Visualization.pose_postprocess import PosePostprocessor
    from Visualization.io.genoview_bvh import _expand_selected_to_full
    template, parents, local_pos, local_rot = _prepare_template()
    processor = PosePostprocessor(template["names"], parents, local_pos, cutoff=cutoff, foot_ik=foot_ik)
    out = []
    for t in range(positions24.shape[0]):
        pos, rot = _expand_selected_to_full(
            positions24[t : t + 1], rotations24[t : t + 1], full_names=template["names"],
            full_parents=parents, template_local_pos=local_pos, template_local_rot=local_rot)
        p, r = processor.process(np.asarray(pos[0], np.float32), np.asarray(rot[0], np.float32), dt=1 / FPS)
        out.append(np.asarray(p, np.float32))
    return np.stack(out)


def render_video(sets, save_path: Path, *, title: str, fps: int = FPS):
    """sets: list of (label, positions[T,J,3]) panels, each rendered as a skeleton."""
    template, parents, local_pos, local_rot = _prepare_template()
    n = len(sets)
    fig = plt.figure(figsize=(6.4 * n, 6.4), dpi=100)
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
    allp = np.concatenate([p.reshape(-1, 3) for _, p in sets])
    lo, hi = allp.min(0), allp.max(0)
    ctr = (lo + hi) / 2
    rad = max((hi - lo).max() / 2, 1.0) * 1.1
    bones = [(j, int(parents[j])) for j in range(1, len(parents)) if parents[j] >= 0]
    artists = []
    for ax, (label, positions) in zip(axes, sets):
        lines = [ax.plot([], [], [], color="#3E7CB1", lw=2.0)[0] for _ in bones]
        (hips_dot,) = ax.plot([], [], [], "o", color="#E08E45", ms=4)
        foot_lines = {}
        for side, (fa, fb) in FOOT_IDS.items():
            foot_lines[side] = ax.plot([], [], [], "o", color="#C94F4F", ms=3)[0]
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[2] - rad, ctr[2] + rad)
        ax.set_zlim(max(0, lo[1] - .2), lo[1] + rad)
        ax.set_box_aspect((1, 1, 0.8))
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        artists.append((lines, hips_dot, foot_lines))
    fig.suptitle(title, fontsize=12)
    T = min(p.shape[0] for _, p in sets)

    def update(t):
        for (lines, hips_dot, foot_lines), (_, positions) in zip(artists, sets):
            for line, (j, pj) in zip(lines, bones):
                line.set_data_3d([positions[t, j, 0], positions[t, pj, 0]],
                                 [positions[t, j, 2], positions[t, pj, 2]],
                                 [positions[t, j, 1], positions[t, pj, 1]])
            hips_dot.set_data_3d([positions[t, 0, 0]], [positions[t, 0, 2]], [positions[t, 0, 1]])
            for side, (fa, fb) in FOOT_IDS.items():
                foot_lines[side].set_data_3d([positions[t, fa, 0], positions[t, fb, 0]],
                                             [positions[t, fa, 2], positions[t, fb, 2]],
                                             [positions[t, fa, 1], positions[t, fb, 1]])
        return []

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    ani.save(str(save_path), writer=writer)
    plt.close(fig)
    del ani
    return save_path


def contact_sheet(positions, rotations, save_path: Path, frames=(30, 90, 150, 210, 270, 299)):
    template, parents, local_pos, local_rot = _prepare_template()
    from Visualization.io.genoview_bvh import _expand_selected_to_full
    fig = plt.figure(figsize=(18, 6.4), dpi=90)
    bones = [(j, int(parents[j])) for j in range(1, len(parents)) if parents[j] >= 0]
    for i, t in enumerate(frames):
        pos, _ = _expand_selected_to_full(
            positions[t : t + 1], rotations[t : t + 1], full_names=template["names"],
            full_parents=parents, template_local_pos=local_pos, template_local_rot=local_rot)
        p = np.asarray(pos[0], np.float32)
        ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
        for j, pj in bones:
            ax.plot([p[j, 0], p[pj, 0]], [p[j, 2], p[pj, 2]], [p[j, 1], p[pj, 1]], color="#3E7CB1", lw=1.6)
        ax.set_title(f"t={t} ({t / FPS:.1f}s)", fontsize=9)
        ax.set_xlim(p[:, 0].min() - .5, p[:, 0].max() + .5)
        ax.set_ylim(p[:, 2].min() - .5, p[:, 2].max() + .5)
        ax.set_zlim(0, p[:, 1].max() + .2)
        ax.set_box_aspect((1, 1, 1)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-ckpt", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--style", type=str, default="Neutral")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--noise-std", type=float, default=0.25)
    ap.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True,
                    help="Feedback snapping to the VQ codebook (formal). --no-requantize uses continuous AE feedback")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render", choices=("none", "single", "dual"), default="dual")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "motion_quality_diagnosis" / "videos")
    args = ap.parse_args()
    args.flow_ckpt = args.flow_ckpt.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve() / args.tag
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Record] {args.tag}: session on {args.flow_ckpt.name}, style={args.style}, "
          f"noise_std={args.noise_std}", flush=True)
    rec = record_session(args.flow_ckpt, style=args.style, frames=args.frames,
                         noise_std=args.noise_std, seed=args.seed, speed=args.speed, device=args.device,
                         requantize=args.requantize)
    print(f"[Record] {args.frames} frames in {rec['wall']:.1f}s", flush=True)
    model_m = analyze(rec["positions"], name=args.tag)
    np.savez_compressed(args.out_dir / "session.npz", positions=rec["positions"],
                        rotations=rec["rotations"], raw=rec["raw"], roots=rec["roots"])

    reference = None
    ref_metrics = None
    try:
        reference = gt_reference(args.style, args.frames, args.seed)
        ref_metrics = analyze(reference["positions"], name=f"GT_{args.style}_{reference['suffix']}")
    except Exception as exc:  # reference is best-effort
        print(f"[Warn] GT reference skipped: {exc}", flush=True)

    post = None
    if args.render in ("single", "dual"):
        try:
            post = postprocess_sequence(rec["positions"], rec["rotations"])
        except Exception as exc:
            print(f"[Warn] postprocess skipped: {exc}", flush=True)

    report = dict(
        tag=args.tag, checkpoint=str(args.flow_ckpt), style=args.style,
        noise_std=args.noise_std, frames=args.frames,
        model=model_m["summary"], model_phases=model_m["phases"],
        reference=None if ref_metrics is None else ref_metrics["summary"],
        reference_phases=None if ref_metrics is None else ref_metrics["phases"],
        reference_clip=None if reference is None else reference["suffix"],
    )

    if args.render != "none":
        raw_full = expand_sequence(rec["positions"], rec["rotations"])
        panels = [("generated (raw)", raw_full)]
        if args.render == "dual" and post is not None:
            panels.append(("display post-processed", post))
        vid = render_video(panels, args.out_dir / f"{args.tag}_control.mp4",
                           title=f"{args.tag} | scripted walk/turn/strafe | noise_std={args.noise_std}")
        report["video"] = str(vid)
        sheet = contact_sheet(rec["positions"], rec["rotations"], args.out_dir / f"{args.tag}_contact.png")
        report["contact_sheet"] = str(sheet)
        if reference is not None:
            ref_full = expand_sequence(reference["positions"], reference["rotations"])
            render_video([(f"GT reference ({reference['suffix']})", ref_full)],
                         args.out_dir / f"{args.tag}_reference.mp4",
                         title=f"GT {args.style}/{reference['suffix']} | same metrics fold")

    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(report["model"], indent=2, ensure_ascii=False, default=float), flush=True)
    if report["reference"]:
        print("[GT]", json.dumps(report["reference"], ensure_ascii=False, default=float), flush=True)
    print(f"[Done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
