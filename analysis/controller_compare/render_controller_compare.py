"""Reference vs label-controller generation, side by side in a live GenoView window (or headless metrics).

Mirrors analysis/vq_reconstruction_check/render_vq_compare.py, but the right panel is the frozen
label controller's autoregressive rollout (official protocol: Euler8, noise std 1,
clip->quantize->decode), driven by the reference clip's 84-dim trajectory plan.

  - matched : target style == source style (pass no --target-style)
  - cross   : target style differs (pass --target-style)

Per clip it prints RootADE (cm, integrated root tracks) and the frozen 240f style classifier's
verdict on the generated motion (classification input = generated frames only, GT seed excluded,
same as evaluate_minor_revision_generalization.py).

Examples:
  # one random 240-frame clip per source style, live window per clip: left=reference right=generated
  python analysis/controller_compare/render_controller_compare.py \
      --styles March Roadrunner --frames 240

  # cross-style: source March, target Zombie (same trajectory plan)
  python analysis/controller_compare/render_controller_compare.py \
      --styles March --target-style Zombie --frames 240

  # headless metrics only (no window), fixed seed
  python analysis/controller_compare/render_controller_compare.py \
      --styles Neutral --backend none --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Utils.motion375 import reconstruct_root_track_from_375
from Utils.motion_eval import MotionEvalConfig, compute_sequence_metric_sums
from Utils.runtime import load_checkpoint, load_norm_stats, resolve_device, configure_torch_runtime, set_seed
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.fid_feature_extractor import build_motion_feature_classifier_from_checkpoint, motion375_clip_to_classifier_input
from evaluation.simulate_realtime_flowmatch_control import (
    _resolve_base_ckpt_path,
    _resolve_norm_stats_path,
    _build_flow_controller,
    rollout_sequence,
    reconstruct_sequence,
)

SCRIPT_DIR = ROOT
DEFAULT_FLOW = SCRIPT_DIR / "checkpoints" / "flowmatch_label_nomirror_150ep" / "best.pt"
DEFAULT_CLASSIFIER = SCRIPT_DIR / "checkpoints" / "fid_feature_extractor_camdm100_60fps_240f_nomirror" / "100style_position_classifier.pth"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    p.add_argument("--flow-ckpt", type=Path, default=DEFAULT_FLOW)
    p.add_argument("--vqvae-ckpt", type=Path, default=None)
    p.add_argument("--norm-stats", type=Path, default=None)
    p.add_argument("--classifier-ckpt", type=Path, default=DEFAULT_CLASSIFIER)
    p.add_argument("--no-classifier", action="store_true", help="Skip style classification (ADE only)")
    p.add_argument("--split", default="any", choices=("train", "val", "test", "any"))
    p.add_argument("--styles", nargs="*", default=None, help="Source style names or label ids; default: random styles")
    p.add_argument("--target-style", default=None, help="Target style for cross generation; default = matched (same as source)")
    p.add_argument("--clips-per-style", type=int, default=1)
    p.add_argument("--frames", type=int, default=240, help="Generated frames per clip (plus the 1-frame GT seed)")
    p.add_argument("--seed", type=int, default=None, help="None draws a fresh random clip each run; set an int to reproduce")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", default="rollout", choices=("rollout", "reconstruct", "vq"),
                   help="rollout = official autoregressive protocol; reconstruct = teacher-forced upper bound; "
                        "vq = GT encode->decode representation ceiling on the same clip")
    p.add_argument("--ode-steps", type=int, default=None,
                   help="Euler steps per generated frame; default inherits solver_steps from the checkpoint args")
    p.add_argument("--noise-std", type=float, default=1.0)
    p.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True,
                   help="Feedback latent snapping (formal protocol); --no-requantize uses continuous AE feedback")
    p.add_argument("--backend", default="genoview", choices=("genoview", "none"), help="genoview = live mesh window; none = headless")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "analysis" / "controller_compare" / "runs")
    return p.parse_args()


def pick_clips(segments, names, styles, window, clips_per_style, split, rng):
    """Pick `clips_per_style` random segments per requested style and a random `window`-frame slice in each."""
    pool = [s for s in segments if s["num_frames"] >= window]
    if split != "any":
        pool = [s for s in pool if s["split"] == split]
    if styles:
        wanted = [int(e) if str(e).isdigit() else names.index(e) for e in styles]
        pool = [s for s in pool if s["label"] in wanted]
    chosen, used = [], {}
    for seg in (pool[i] for i in rng.permutation(len(pool))):
        count = used.get(seg["label"], 0)
        if count < clips_per_style:
            start = int(seg["frame_start"]) + int(rng.integers(0, seg["num_frames"] - window + 1))
            chosen.append((seg, start))
            used[seg["label"]] = count + 1
    return chosen


def classify_style(classifier, meta, clip_raw, device):
    """Top-1 style over one generated window (GT seed excluded), mirroring the evaluation path."""
    window = int(meta["window_size"])
    if clip_raw.shape[0] < window:
        return None
    windows = np.asarray(clip_raw[:window], dtype=np.float32)[None]
    x_cls = motion375_clip_to_classifier_input(windows, joint_set=str(meta["joint_set"])).to(device)
    with torch.no_grad():
        _, logits = classifier(x_cls)
        probs = torch.softmax(logits.float(), dim=1)[0]
    pred = int(probs.argmax())
    return dict(pred_label=pred, pred_name=str(meta["class_names"][pred]), pred_conf=float(probs[pred]))


def root_ade_cm(ref_raw, gen_raw):
    """Mean per-frame root XZ distance (cm) over the full window, matching Utils.motion_eval."""
    ref_pos, _ = reconstruct_root_track_from_375(ref_raw, dt=1 / 60, root_wy_in_degrees=True)
    gen_pos, _ = reconstruct_root_track_from_375(gen_raw, dt=1 / 60, root_wy_in_degrees=True)
    return float(np.linalg.norm(gen_pos[:, [0, 2]] - ref_pos[:, [0, 2]], axis=-1).mean() * 100.0)


def motion_quality_cm(ref_raw, gen_raw):
    """Jitter / foot-sliding proxies plus global MPJPE, same definitions as the paper metrics."""
    from Utils.motion375 import reconstruct_selected_globals_from_375
    sums = compute_sequence_metric_sums(reference_375=ref_raw, prediction_375=gen_raw, cfg=MotionEvalConfig())
    jitter = sums.jitter_sum_cm / max(sums.jitter_count, 1)
    foot = sums.foot_sliding_sum_cm / max(sums.foot_sliding_count, 1)
    mpjpe = sums.mpjpe_sum_cm / max(sums.mpjpe_count, 1)
    ref_local, _ = reconstruct_selected_globals_from_375(ref_raw, dt=1 / 60, root_wy_in_degrees=True, use_global=False)
    gen_local, _ = reconstruct_selected_globals_from_375(gen_raw, dt=1 / 60, root_wy_in_degrees=True, use_global=False)
    local_mpjpe = float(np.linalg.norm(gen_local - ref_local, axis=-1).mean() * 100.0)
    return float(jitter), float(foot), float(mpjpe), local_mpjpe


def main():
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.flow_ckpt = args.flow_ckpt.expanduser().resolve()
    device = resolve_device(args.device)
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision="highest")
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    checkpoint = load_checkpoint(args.flow_ckpt, map_location="cpu")
    config = checkpoint.get("args", {})
    if config.get("conditioning_mode", "label") != "label":
        raise SystemExit("This comparison entry supports label-mode checkpoints")
    kind, base_path = _resolve_base_ckpt_path(args.vqvae_ckpt, None, flow_args=config, flow_ckpt_path=args.flow_ckpt)
    norm_path = _resolve_norm_stats_path(args.norm_stats, flow_args=config, flow_ckpt_path=args.flow_ckpt, base_ckpt_path=base_path)
    mean, std = load_norm_stats(norm_path)

    names = json.loads((args.db / "StyleNames.json").read_text())
    meta = json.loads((args.db / "meta.json").read_text())
    segments = json.loads((args.db / "segments.json").read_text())
    x_all = np.memmap(args.db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))

    window = int(args.frames) + 1  # 1 GT seed frame + generated frames
    chosen = pick_clips(segments, names, args.styles, window, args.clips_per_style, args.split, rng)
    if not chosen:
        raise SystemExit("No segments matched the request")

    vq = build_vqvae_from_checkpoint(base_path).to(device)
    model = _build_flow_controller(flow_args=config, flow_ckpt=checkpoint, base_model=vq, base_model_type=kind,
                                   num_styles=int(config.get("num_styles", len(names))), device=device)
    if args.ode_steps is None:
        args.ode_steps = int(config.get("solver_steps", 8))

    classifier, classifier_meta = None, None
    if not args.no_classifier:
        ckpt_path = Path(args.classifier_ckpt).expanduser().resolve()
        if ckpt_path.exists():
            classifier, classifier_meta = build_motion_feature_classifier_from_checkpoint(ckpt_path)
            classifier = classifier.to(device).eval()
            class_names = list(classifier_meta["class_names"])
            if len(class_names) < len(names) or class_names[: len(names)] != names:
                print("[Warn] classifier class names differ from DB StyleNames order; labels may be off", flush=True)
        else:
            print(f"[Warn] classifier not found at {ckpt_path}; skipping style classification", flush=True)

    target_label = None
    if args.target_style is not None:
        target_label = int(args.target_style) if str(args.target_style).isdigit() else names.index(args.target_style)

    print(f"[Info] seed={seed} split={args.split} frames={args.frames} mode={args.mode} "
          f"requantize={args.requantize} noise_std={args.noise_std} target_style="
          f"{names[target_label] if target_label is not None else 'matched(source)'}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg, start in chosen:
        raw = np.asarray(x_all[start : start + window], dtype=np.float32)
        raw_norm = (raw - mean) / std
        gen_label = int(seg["label"]) if target_label is None else target_label
        t0 = time.perf_counter()
        if args.mode == "vq":
            # Representation ceiling on the same clip: GT encode -> codebook -> decode.
            x_gt = torch.from_numpy(raw_norm.astype(np.float32)).to(device)[None]
            with torch.no_grad():
                z_q, _, _ = vq.encode(x_gt)
                motion_q = vq.decode(z_q)[0]
            seq_norm = torch.cat((x_gt[0, :, :84], torch.cat((x_gt[0, :1, 84:], motion_q[1:]), 0)), dim=-1)
            seq_norm = seq_norm.cpu().numpy()
        elif args.mode == "rollout":
            seq_norm, _ = rollout_sequence(
                model=model, base_model_type=kind,
                x_seed_norm=raw_norm[0].astype(np.float32), traj_plan_norm=raw_norm[:, :84].astype(np.float32),
                style_label=gen_label, manual_style_probs=None, manual_style_strength=1.0,
                ode_steps=args.ode_steps, noise_std=args.noise_std, solver="euler",
                device=device, requantize_z=bool(args.requantize), latent_clip=True)
        else:
            seq_norm, _ = reconstruct_sequence(
                model=model, base_model_type=kind, x_ref_norm=raw_norm.astype(np.float32),
                style_label=gen_label, manual_style_probs=None, manual_style_strength=1.0,
                ode_steps=args.ode_steps, noise_std=args.noise_std, solver="euler",
                device=device, requantize_z=bool(args.requantize), latent_clip=True)
        gen_ms = (time.perf_counter() - t0) * 1000.0 / max(seq_norm.shape[0] - 1, 1)
        gen_raw = np.asarray(seq_norm, dtype=np.float32) * std[None] + mean[None]

        row = dict(style=names[seg["label"]], suffix=seg["suffix"], segment_id=seg["source_entry_id"],
                   window_start=int(start), seed=int(seed), mode=args.mode, requantize=bool(args.requantize),
                   target_style=names[gen_label], matched=(target_label is None))
        row["root_ade_cm"] = root_ade_cm(raw, gen_raw)
        row["jitter_cm"], row["foot_sliding_cm"], row["mpjpe_cm"], row["local_mpjpe_cm"] = motion_quality_cm(raw, gen_raw)
        if classifier is not None:
            verdict = classify_style(classifier, classifier_meta, gen_raw[1:], device)  # GT seed excluded
            if verdict is not None:
                row.update(pred_style=verdict["pred_name"], pred_conf=verdict["pred_conf"],
                           pred_is_target=(verdict["pred_label"] == gen_label))
        row["gen_ms_per_frame"] = gen_ms
        rows.append(row)
        print(f"[Clip] {row['style']:<14}{seg['suffix']:<4} -> {row['target_style']:<14} "
              f"ADE={row['root_ade_cm']:.2f}cm jit={row['jitter_cm']:.2f} "
              + (f"pred={row['pred_style']} ({row['pred_conf']:.2f}) "
                 f"{'MATCH' if row.get('pred_is_target') else 'miss'} " if 'pred_style' in row else '')
              + f"{gen_ms:.1f}ms/frame", flush=True)

        if args.backend == "genoview":
            from Visualization.backends.dispatch_backend import visualize_pair_from_375
            tag = f"matched {row['style']}->{row['target_style']}" if row["matched"] else f"cross {row['style']}->{row['target_style']}"
            verdict_txt = f" | pred={row.get('pred_style')} {row.get('pred_conf', 0):.2f}" if "pred_style" in row else ""
            visualize_pair_from_375(
                ref_features_375=raw, pred_features_375=gen_raw, vis_backend="genoview",
                fps=60, dt=1 / 60, use_global=True,
                title=f"Label controller {args.mode} | {tag} | ADE {row['root_ade_cm']:.1f}cm{verdict_txt}"
                      f" | left=reference right=generated",
                db_path=args.db, show=True)

    out_path = args.out_dir / f"summary_s{seed}_{args.mode}.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=float))
    if rows:
        mean_ade = float(np.mean([r["root_ade_cm"] for r in rows]))
        mean_jitter = float(np.mean([r["jitter_cm"] for r in rows]))
        mean_foot = float(np.mean([r["foot_sliding_cm"] for r in rows]))
        mean_mpjpe = float(np.mean([r["mpjpe_cm"] for r in rows]))
        mean_lmpjpe = float(np.mean([r["local_mpjpe_cm"] for r in rows]))
        acc = [r["pred_is_target"] for r in rows if "pred_is_target" in r]
        print(f"[Summary] mode={args.mode} clips={len(rows)} mean RootADE={mean_ade:.2f}cm "
              f"MPJPE={mean_mpjpe:.2f}cm localMPJPE={mean_lmpjpe:.2f}cm "
              f"jitter={mean_jitter:.3f}cm F.S.={mean_foot:.3f}cm "
              + (f"style hits={sum(acc)}/{len(acc)} ({100.0 * sum(acc) / len(acc):.1f}%)" if acc else ""), flush=True)
    print(f"[Done] summary -> {out_path}")


if __name__ == "__main__":
    main()
