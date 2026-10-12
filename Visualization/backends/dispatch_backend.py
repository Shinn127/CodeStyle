from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Visualization.backends.genoview_backend import visualize_dual_motion375_genoview_from_375  # noqa: E402
from Visualization.backends.matplotlib_backend import visualize_side_by_side_from_375  # noqa: E402
from Visualization.io.reference_bvh import canonicalize_vis_backend, load_selected_parents_24  # noqa: E402


def visualize_pair_from_375(
    *,
    ref_features_375: np.ndarray,
    pred_features_375: np.ndarray,
    vis_backend: str,
    fps: int,
    dt: float,
    use_global: bool,
    db_path: Path | None = None,
    data_root: Path | None = None,
    bvh_ref: Path | None = None,
    root_wy_in_degrees: bool = True,
    coordinate_convert: bool = False,
    trail_len: int = 120,
    title: str | None = None,
    gif_path: Path | None = None,
    show: bool = True,
    geno_gap: float = 2.4,
) -> dict[str, Any]:
    backend = canonicalize_vis_backend(vis_backend)
    if backend == "genoview":
        if gif_path is not None:
            raise ValueError("--gif-path is only supported with --vis-backend quick/matplotlib")
        visualize_dual_motion375_genoview_from_375(
            ref_features_375=ref_features_375,
            pred_features_375=pred_features_375,
            db_path=db_path,
            fps=fps,
            dt=dt,
            use_global=use_global,
            root_wy_in_degrees=root_wy_in_degrees,
            coordinate_convert=coordinate_convert,
            title=title,
            model_gap=geno_gap,
        )
        return {
            "backend": backend,
            "reference_bvh": None,
        }

    if data_root is None:
        raise ValueError("data_root is required for quick/matplotlib visualization")

    parents_24, resolved_bvh = load_selected_parents_24(
        data_root=data_root.expanduser().resolve(),
        bvh_ref=bvh_ref,
    )
    visualize_side_by_side_from_375(
        ref_features_375=ref_features_375,
        pred_features_375=pred_features_375,
        parents=parents_24,
        fps=fps,
        dt=dt,
        use_global=use_global,
        root_wy_in_degrees=root_wy_in_degrees,
        coordinate_convert=coordinate_convert,
        trail_len=trail_len,
        title=title,
        save_path=gif_path,
        show=show,
    )
    return {
        "backend": backend,
        "reference_bvh": resolved_bvh,
    }
