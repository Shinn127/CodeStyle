"""Visualization helpers for motion375 data.

Submodules are loaded lazily so quick/Matplotlib tools do not depend on the
GenoView/Scipy stack unless that path is actually used.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SELECTED_24": ("Visualization.io.reference_bvh", "SELECTED_24"),
    "animation_plot": ("Visualization.backends.matplotlib_backend", "animation_plot"),
    "animation_plot_frame": ("Visualization.backends.matplotlib_backend", "animation_plot_frame"),
    "animation_plot_from_375": ("Visualization.backends.matplotlib_backend", "animation_plot_from_375"),
    "build_reduced_parents": ("Visualization.io.reference_bvh", "build_reduced_parents"),
    "canonicalize_vis_backend": ("Visualization.io.reference_bvh", "canonicalize_vis_backend"),
    "first_bvh_under": ("Visualization.io.reference_bvh", "first_bvh_under"),
    "load_selected_parents_24": ("Visualization.io.reference_bvh", "load_selected_parents_24"),
    "resolve_reference_bvh": ("Visualization.io.reference_bvh", "resolve_reference_bvh"),
    "visualize_dual_motion375_genoview_from_375": (
        "Visualization.backends.genoview_backend",
        "visualize_dual_motion375_genoview_from_375",
    ),
    "visualize_pair_from_375": ("Visualization.backends.dispatch_backend", "visualize_pair_from_375"),
    "visualize_side_by_side_from_375": ("Visualization.backends.matplotlib_backend", "visualize_side_by_side_from_375"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
