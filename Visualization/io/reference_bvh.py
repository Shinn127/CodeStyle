from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils import bvh  # noqa: E402
from Utils.motion375 import SELECTED_24_INDICES  # noqa: E402


SELECTED_24 = list(SELECTED_24_INDICES)


def canonicalize_vis_backend(vis_backend: str) -> str:
    backend = str(vis_backend).strip().lower()
    if backend == "matplotlib":
        return "quick"
    if backend in {"genoview", "quick"}:
        return backend
    raise ValueError(f"Unsupported visualization backend: {vis_backend!r}")


def first_bvh_under(root: Path) -> Path:
    files = sorted(root.rglob("*.bvh"))
    if not files:
        raise FileNotFoundError(f"No .bvh files found under: {root}")
    return files[0]


def build_reduced_parents(full_parents: np.ndarray, selected: list[int]) -> list[int]:
    selected_set = set(selected)
    remap = {joint: idx for idx, joint in enumerate(selected)}

    reduced: list[int] = []
    for joint in selected:
        parent = int(full_parents[joint])
        while parent != -1 and parent not in selected_set:
            parent = int(full_parents[parent])
        reduced.append(-1 if parent == -1 else remap[parent])
    return reduced


def resolve_reference_bvh(data_root: Path, bvh_ref: Path | None = None) -> Path:
    if bvh_ref is not None:
        resolved = bvh_ref.expanduser().resolve()
    else:
        resolved = first_bvh_under(data_root.expanduser().resolve())
    if not resolved.exists():
        raise FileNotFoundError(f"Reference BVH not found: {resolved}")
    return resolved


def load_selected_parents_24(*, data_root: Path, bvh_ref: Path | None = None) -> tuple[list[int], Path]:
    resolved_bvh = resolve_reference_bvh(data_root=data_root, bvh_ref=bvh_ref)
    bvh_data = bvh.load_zeroeggs(str(resolved_bvh))
    parents_24 = build_reduced_parents(np.asarray(bvh_data["parents"], dtype=np.int32), SELECTED_24)
    return parents_24, resolved_bvh


__all__ = [
    "SELECTED_24",
    "build_reduced_parents",
    "canonicalize_vis_backend",
    "first_bvh_under",
    "load_selected_parents_24",
    "resolve_reference_bvh",
]
