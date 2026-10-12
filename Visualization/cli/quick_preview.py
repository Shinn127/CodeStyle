"""
Quick sanity-check visualization for 100STYLE 375-dim database.

This script randomly samples a contiguous 300-frame segment (default)
from a single style label, reconstructs motion from 375-dim traj+motion
features, and exports:
  - one GIF animation
  - one key-frame PNG
  - one JSON metadata file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent
REPO_ROOT = NEWCODE_DIR.parent

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Visualization.backends.matplotlib_backend import animation_plot_from_375  # noqa: E402
from Visualization.io.reference_bvh import SELECTED_24, load_selected_parents_24  # noqa: E402
from Utils.realtime import (  # noqa: E402
    choose_segment,
    load_db_arrays,
    parse_style_selector,
    resolve_db_coordinate_convert,
)


def ensure_output_paths(out_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{prefix}.gif"
    png_path = out_dir / f"{prefix}.png"
    json_path = out_dir / f"{prefix}.json"
    return gif_path, png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Random 300-frame quick visualization for 100STYLE 375-dim DB")
    parser.add_argument(
        "--db",
        type=Path,
        default=NEWCODE_DIR / "database_100sty_375_memmap_nomirror",
        help="Path to a 375-dim memmap DB directory",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Number of contiguous frames to sample")
    parser.add_argument("--seed", type=int, default=242, help="Random seed")
    parser.add_argument(
        "--style",
        type=str,
        default="",
        help="Optional style selector: style name (e.g., Zombie) or label id (e.g., 99)",
    )
    parser.add_argument("--fps", type=int, default=60, help="Output GIF fps")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Frame dt for root integration")
    parser.add_argument(
        "--root-wy-deg",
        action="store_true",
        default=True,
        help="Treat wy as degrees/sec when reconstructing root yaw (default: true)",
    )
    parser.add_argument(
        "--root-wy-rad",
        action="store_true",
        help="Treat wy as radians/sec when reconstructing root yaw",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Visualize local positions only (skip global root reconstruction)",
    )
    parser.add_argument(
        "--coordinate-convert",
        dest="coordinate_convert",
        action="store_true",
        default=None,
        help="Flip x into the Geno-compatible basis before quick plotting",
    )
    parser.add_argument(
        "--no-coordinate-convert",
        dest="coordinate_convert",
        action="store_false",
        help="Disable coordinate conversion even if stored in DB metadata",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "100sty",
        help="Root path to 100STYLE BVH files",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional reference BVH for deriving 24-joint parent topology",
    )
    parser.add_argument("--out-dir", type=Path, default=NEWCODE_DIR, help="Output directory for gif/png/json")
    parser.add_argument("--prefix", type=str, default="quick_vis_100sty_300", help="Output filename prefix")
    parser.add_argument("--show", action="store_true", help="Show animation directly in Matplotlib window")
    parser.add_argument("--no-gif", action="store_true", help="Do not export GIF")
    parser.add_argument("--no-png", action="store_true", help="Do not export key-frame PNG")
    parser.add_argument("--no-json", action="store_true", help="Do not export metadata JSON")
    args = parser.parse_args()

    if args.num_frames <= 1:
        raise ValueError("--num-frames must be > 1")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    db_path = args.db.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    need_gif = not args.no_gif
    need_png = not args.no_png
    need_json = not args.no_json
    need_show = args.show

    if not (need_gif or need_png or need_json or need_show):
        raise ValueError("Nothing to do. Enable at least one of: GIF/PNG/JSON output or --show")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    gif_path = png_path = json_path = None
    if need_gif or need_png or need_json:
        gif_path, png_path, json_path = ensure_output_paths(out_dir, args.prefix)

    x_source, labels_np, style_names = load_db_arrays(db_path)
    coordinate_convert = args.coordinate_convert
    if coordinate_convert is None:
        coordinate_convert = resolve_db_coordinate_convert(db_path)

    target_label = parse_style_selector(args.style, style_names)
    start, end, sampled_label, src_run = choose_segment(
        labels_np,
        num_frames=args.num_frames,
        seed=args.seed,
        target_label=target_label,
    )
    x_seg = np.asarray(x_source[start:end], dtype=np.float32)

    parents_24, bvh_ref = load_selected_parents_24(data_root=data_root, bvh_ref=args.bvh_ref)
    style_name = style_names[sampled_label] if 0 <= sampled_label < len(style_names) else f"label_{sampled_label}"
    interval_ms = int(round(1000.0 / args.fps))
    use_deg = not args.root_wy_rad
    title = f"100STYLE quick check | style={style_name} (label={sampled_label}) | seg=[{start},{end})"

    print(f"[Info] DB            : {db_path}")
    print("[Info] DB format     : memmap")
    print(f"[Info] X segment     : {x_seg.shape}")
    print(f"[Info] Selected style: {style_name} ({sampled_label})")
    print(f"[Info] Source run    : start={src_run[0]}, end={src_run[1]}, label={src_run[2]}")
    print(f"[Info] Segment       : start={start}, end={end}")
    print(f"[Info] Ref BVH       : {bvh_ref}")
    print(f"[Info] Parents(24)   : {len(parents_24)} joints from selected set {len(SELECTED_24)}")
    print(f"[Info] Coord convert : {coordinate_convert}")

    outputs = []
    if need_gif:
        outputs.append(str(gif_path))
    if need_png:
        outputs.append(str(png_path))
    if need_json:
        outputs.append(str(json_path))
    if need_show:
        outputs.append("matplotlib window")
    print(f"[Info] Outputs       : {', '.join(outputs)}")

    common_kwargs = {
        "features_375": x_seg,
        "parents": parents_24,
        "title": title,
        "dt": args.dt,
        "root_wy_in_degrees": use_deg,
        "use_global": not args.local_only,
        "coordinate_convert": bool(coordinate_convert),
    }

    if need_gif:
        animation_plot_from_375(
            interval=interval_ms,
            save_path=gif_path,
            single_frame=False,
            trail_len=min(120, args.num_frames),
            **common_kwargs,
        )

    if need_png:
        animation_plot_from_375(
            save_path=png_path,
            single_frame=True,
            frame_idx=args.num_frames // 2,
            **common_kwargs,
        )

    if need_show:
        animation_plot_from_375(
            interval=interval_ms,
            save_path=None,
            single_frame=False,
            trail_len=min(120, args.num_frames),
            **common_kwargs,
        )

    if need_json:
        meta = {
            "db": str(db_path),
            "db_format": "memmap",
            "style_label": int(sampled_label),
            "style_name": style_name,
            "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
            "segment": {"start": int(start), "end": int(end), "num_frames": int(args.num_frames)},
            "fps": int(args.fps),
            "dt": float(args.dt),
            "root_wy_in_degrees": bool(use_deg),
            "use_global": bool(not args.local_only),
            "coordinate_convert": bool(coordinate_convert),
            "bvh_ref": str(bvh_ref),
            "gif": str(gif_path) if gif_path is not None else None,
            "png": str(png_path) if png_path is not None else None,
        }
        with json_path.open("w", encoding="utf-8") as fw:
            json.dump(meta, fw, ensure_ascii=False, indent=2)

    print("[Done] Quick visualization export complete.")


if __name__ == "__main__":
    main()
