"""
Precompute style-conditioned VQ code priors from a cached frame-wise VQ target index file.

The output stores:
  - prior[style, slot, code]       = p(code | style, slot)
  - global_prior[slot, code]       = p(code | slot)
  - slot_saliency[slot]            = mean_style JS(prior_style_slot || global_slot)
  - selected_slots[M_style]        = most style-discriminative slots

The training script can then use this file to:
  1) decide which VQ slots act as style slots
  2) supervise the StyleCodeSelector with a soft style-code prior
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np

from Utils.runtime import require_unmirrored_db
from Utils.style_code_prior import save_style_code_prior
from Utils.vq_target_cache import VQTargetCache


MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"
MEMMAP_SEGMENTS_FILE = "segments.json"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Precompute style-conditioned VQ code priors")
    parser.add_argument(
        "--db",
        type=Path,
        default=script_dir / "database_100sty_375_memmap_nomirror",
        help="Input memmap database directory",
    )
    parser.add_argument(
        "--vq-target-cache",
        type=Path,
        default=script_dir / "cache" / "vq_targets_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048",
        help="VQ target cache directory produced by precompute_vq_targets.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "cache" / "style_code_prior_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to use when estimating style priors",
    )
    parser.add_argument(
        "--style-num-slots",
        type=int,
        default=8,
        help="How many discriminative slots to select for style control",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Laplace smoothing applied before normalizing counts into priors",
    )
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument('--vqvae-ckpt', type=Path, default=None, help='Content-identical relocated VQ source for provenance validation')
    parser.add_argument('--norm-stats', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _load_style_names(db_path: Path, *, num_styles: int) -> list[str]:
    style_names_path = db_path / MEMMAP_STYLES_FILE
    if style_names_path.exists():
        with style_names_path.open("r", encoding="utf-8") as f:
            style_names = [str(name) for name in json.load(f)]
        if len(style_names) < num_styles:
            style_names.extend(f"label_{idx}" for idx in range(len(style_names), num_styles))
        return style_names[:num_styles]
    return [f"label_{idx}" for idx in range(num_styles)]


def _load_split_frame_ids(db_path: Path, *, split: str, num_frames: int) -> np.ndarray:
    if split == "all":
        return np.arange(num_frames, dtype=np.int64)

    segments_path = db_path / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        raise FileNotFoundError(
            f"{segments_path} not found; split={split!r} requires a database with precomputed split metadata."
        )
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)
    frame_ids: list[np.ndarray] = []
    for segment in segments:
        if str(segment.get("split", "")) != split:
            continue
        frame_start = int(segment["frame_start"])
        segment_frames = int(segment["num_frames"])
        if segment_frames <= 0:
            continue
        frame_ids.append(np.arange(frame_start, frame_start + segment_frames, dtype=np.int64))
    if not frame_ids:
        raise RuntimeError(f"No frames found for split={split!r} in {segments_path}")
    result = np.concatenate(frame_ids, axis=0)
    if result.min() < 0 or result.max() >= num_frames:
        raise ValueError(f"Frame ids for split={split!r} fall outside [0, {num_frames})")
    return result


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.vq_target_cache = args.vq_target_cache.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f'{args.out} already exists; use a new path or explicit --overwrite')

    require_unmirrored_db(args.db)
    if args.style_num_slots < 1:
        raise ValueError(f"--style-num-slots must be >= 1, got {args.style_num_slots}")
    if args.smoothing < 0.0:
        raise ValueError(f"--smoothing must be >= 0, got {args.smoothing}")
    if args.chunk_rows < 1:
        raise ValueError(f"--chunk-rows must be >= 1, got {args.chunk_rows}")
    if args.log_interval < 1:
        raise ValueError(f"--log-interval must be >= 1, got {args.log_interval}")

    cache = VQTargetCache(args.vq_target_cache)
    if cache.meta.fingerprints:
        from Utils.reproducibility import artifact_fingerprints
        cache.validate(fingerprints=artifact_fingerprints(args.db,
            args.vqvae_ckpt or cache.meta.vqvae_ckpt, args.norm_stats or cache.meta.norm_stats))
    else:
        cache.validate(db=args.db)
    labels_path = args.db / MEMMAP_LABELS_FILE
    labels = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(cache.meta.num_frames,))
    num_styles = int(labels.max()) + 1 if labels.size else 0
    if num_styles < 1:
        raise RuntimeError("Database has no style labels")
    style_names = _load_style_names(args.db, num_styles=num_styles)
    frame_ids = _load_split_frame_ids(args.db, split=args.split, num_frames=cache.meta.num_frames)

    num_slots = int(cache.meta.num_codebooks)
    num_embeddings = int(cache.meta.num_embeddings)
    counts = np.zeros((num_styles, num_slots, num_embeddings), dtype=np.int64)

    print(f"[Info] db              : {args.db}")
    print(f"[Info] cache           : {args.vq_target_cache}")
    print(f"[Info] split           : {args.split}")
    print(f"[Info] frames          : {frame_ids.size}")
    print(f"[Info] num styles      : {num_styles}")
    print(f"[Info] num slots       : {num_slots}")
    print(f"[Info] num embeddings  : {num_embeddings}")
    print(f"[Info] smoothing       : {args.smoothing}")

    counts_flat = counts.reshape(-1)
    slot_offsets = np.arange(num_slots, dtype=np.int64)[None, :] * num_embeddings
    style_stride = num_slots * num_embeddings
    for chunk_idx, begin in enumerate(range(0, frame_ids.size, args.chunk_rows), start=1):
        ids = frame_ids[begin : begin + args.chunk_rows]
        labels_chunk = np.asarray(labels[ids], dtype=np.int64)
        indices_chunk = np.asarray(cache.indices[ids], dtype=np.int64)
        if np.any(indices_chunk < 0) or np.any(indices_chunk >= num_embeddings):
            raise ValueError("VQ target cache contains an out-of-range code index")
        flat_bins = labels_chunk[:, None] * style_stride + slot_offsets + indices_chunk
        np.add.at(counts_flat, flat_bins.reshape(-1), 1)
        end = begin + ids.size
        if chunk_idx % args.log_interval == 0 or end == frame_ids.size:
            print(f"[Count] rows {begin}:{end} / {frame_ids.size}")

    smoothing = float(args.smoothing)
    prior = counts.astype(np.float64) + smoothing
    prior /= np.clip(prior.sum(axis=-1, keepdims=True), 1.0, None)
    prior = prior.astype(np.float32)

    global_counts = counts.sum(axis=0)
    global_prior = global_counts.astype(np.float64) + smoothing
    global_prior /= np.clip(global_prior.sum(axis=-1, keepdims=True), 1.0, None)
    global_prior = global_prior.astype(np.float32)

    global_broadcast = global_prior[None, :, :]
    mixture = 0.5 * (prior + global_broadcast)
    prior_safe = np.clip(prior, 1e-8, None)
    global_safe = np.clip(global_broadcast, 1e-8, None)
    mixture_safe = np.clip(mixture, 1e-8, None)
    js = 0.5 * (
        np.sum(prior_safe * (np.log(prior_safe) - np.log(mixture_safe)), axis=-1)
        + np.sum(global_safe * (np.log(global_safe) - np.log(mixture_safe)), axis=-1)
    )
    slot_saliency = js.mean(axis=0).astype(np.float32)

    top_order = np.argsort(-slot_saliency, kind="stable")
    selected_slots = np.sort(top_order[: min(args.style_num_slots, num_slots)]).astype(np.int64)
    out_path = save_style_code_prior(
        args.out,
        prior=prior,
        global_prior=global_prior,
        slot_saliency=slot_saliency,
        selected_slots=selected_slots,
        style_names=style_names,
        counts=counts,
        split=args.split,
        smoothing=smoothing,
        # Provenance: the arrays cannot tell which codebook the prior came from, so record it.
        vqvae_ckpt=cache.meta.vqvae_ckpt,
        db=cache.meta.db,
        fingerprints=cache.meta.fingerprints,
        cache_sha256=cache.meta.indices_sha256,
    )

    selected_scores = ", ".join(
        f"{int(slot)}:{float(slot_saliency[slot]):.5f}" for slot in selected_slots.tolist()
    )
    print(f"[Done] saved           : {out_path}")
    print(f"[Done] selected slots  : {selected_slots.tolist()}")
    print(f"[Done] slot saliency   : {selected_scores}")


if __name__ == "__main__":
    main()
