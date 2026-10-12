# Visualization Module Overview

`NewCode/Visualization` contains the visualization pipeline for `motion375`
data. It is organized into three layers:

- `cli/`
  - Command-line entry points.
  - These files are intended to be run directly with `python ...`.
- `backends/`
  - Rendering and visualization implementations.
  - This layer contains the Matplotlib quick-preview path, the GenoView
    renderer, and the shared backend dispatcher.
- `io/`
  - Visualization-specific data adaptation utilities.
  - This layer contains BVH topology helpers and motion375-to-Geno BVH export.

The package is memmap-only. HDF5 support has been removed from `NewCode`.

## Directory Layout

- `cli/quick_preview.py`
  - Quick visualization for random or style-filtered segments from the
    memmap database.
  - Supports Matplotlib display, GIF export, PNG export, and JSON metadata.
- `cli/genoview_preview.py`
  - Direct GenoView preview for motion375 clips stored in the memmap DB.
- `cli/export_genoview_bvh.py`
  - Export a motion375 clip to a Geno-compatible BVH file.
- `backends/matplotlib_backend.py`
  - Matplotlib plotting backend for single motion and side-by-side comparison.
- `backends/genoview_backend.py`
  - GenoView rendering backend.
  - Converts motion375 features into a full Geno skeleton and displays them in
    the GenoView runtime.
- `backends/dispatch_backend.py`
  - Shared backend dispatcher used by realtime simulation scripts.
  - Routes visualization calls to `quick` or `genoview`.
- `io/reference_bvh.py`
  - Reference BVH helpers.
  - Used to resolve a BVH file and derive the reduced 24-joint parent
    structure needed by the Matplotlib quick path.
- `io/genoview_bvh.py`
  - GenoView skeleton adaptation and BVH export helpers.
  - Contains the motion375-to-full-Geno expansion logic and BVH writing path.
- `__init__.py`
  - Lazy package exports.
  - Keeps quick/Matplotlib workflows from importing GenoView/Scipy-related
    code unless needed.

## Main Entry Points

### 1. Quick Preview

Use this to inspect a random or style-selected segment with Matplotlib.

Example:

```bash
conda run -n t2m python NewCode/Visualization/cli/quick_preview.py \
  --db NewCode/database_100sty_375_memmap_nomirror \
  --data-root data/100sty \
  --num-frames 300 \
  --fps 60 \
  --seed 242 \
  --show \
  --no-gif \
  --no-png \
  --no-json
```

Useful flags:

- `--style Zombie`
  - Sample only from a specific style.
- `--show`
  - Open a Matplotlib window.
- `--no-gif --no-png --no-json`
  - Disable file outputs for preview-only use.
- `--local-only`
  - Skip global root integration and visualize local positions only.

### 2. GenoView Preview

Use this when you want the full Geno skeleton and runtime rendering.

Example:

```bash
conda run -n t2m python NewCode/Visualization/cli/genoview_preview.py \
  --db NewCode/database_100sty_375_memmap_nomirror \
  --segment-index 0 \
  --num-frames 300 \
  --fps 60 \
  --title "motion375 GenoView" \
  --draw-trajectory
```

Useful flags:

- `--frame-start 0 --num-frames 300`
  - Use an explicit frame range instead of a segment index.
- `--draw-skeleton`
  - Overlay the skeleton.
- `--draw-trajectory`
  - Draw trajectory debug markers.
- `--dry-run`
  - Prepare data and print metadata without opening a window.

### 3. Export Geno-Compatible BVH

Use this to generate a BVH file from a motion375 clip.

Example:

```bash
conda run -n t2m python NewCode/Visualization/cli/export_genoview_bvh.py \
  --db NewCode/database_100sty_375_memmap_nomirror \
  --segment-index 0 \
  --num-frames 300 \
  --fps 60
```

Useful flags:

- `--frame-start 0 --num-frames 300`
  - Export a fixed frame range.
- `--out-bvh NewCode/exports/example.bvh`
  - Override the default output path.
- `--mirror-template`
  - Mirror the Geno template before filling missing joints.

## Backend Selection

The realtime scripts in `NewCode` route through the backend dispatcher:

- `quick`
  - Uses Matplotlib side-by-side visualization.
- `matplotlib`
  - Alias of `quick`.
- `genoview`
  - Uses the GenoView runtime.

The dispatcher lives in `backends/dispatch_backend.py`.

## Data Assumptions

- Input DB must be a memmap directory such as:
  - `NewCode/database_100sty_375_memmap_nomirror`
- The memmap DB is expected to contain:
  - `meta.json`
  - `X.f32`
  - `Labels.i32`
  - `StyleNames.json` (optional but recommended)
  - `segments.json` (needed for segment-based GenoView/BVH workflows)

## External Dependencies

### Quick / Matplotlib Path

Required:

- `numpy`
- `matplotlib`

### GenoView Path

Required:

- `numpy`
- `raylib` Python bindings used by the GenoView runtime
- GenoView project directory at:
  - `GenoViewPython-main/`

Expected GenoView resources include:

- `GenoViewPython-main/resources/Geno_stance.bvh`
- `GenoViewPython-main/resources/Geno.bin`

### BVH Export Path

Required:

- `numpy`
- `scipy`

`scipy` is imported lazily and is only required for BVH export.

## Public Imports

`Visualization/__init__.py` re-exports the main helpers lazily, including:

- `canonicalize_vis_backend`
- `load_selected_parents_24`
- `visualize_pair_from_375`
- `visualize_dual_motion375_genoview_from_375`
- `visualize_side_by_side_from_375`

For internal development, prefer importing from the concrete layer:

- `Visualization.cli...`
- `Visualization.backends...`
- `Visualization.io...`

## Maintenance Notes

- Keep CLI files thin. They should mostly parse arguments and call lower-level
  helpers.
- Put rendering logic in `backends/`.
- Put skeleton mapping, topology lookup, and export helpers in `io/`.
- If a new renderer is added later, it should usually follow this pattern:
  - backend implementation in `backends/`
  - optional CLI in `cli/`
  - any format conversion or asset prep in `io/`

## 在线label控制

`bash scripts/run_interactive_label_control.sh`打开真正逐帧生成的GenoView交互窗口，默认label150模型。WASD移动、QE转向、搜索列表或`[`/`]`切换风格；鼠标控制相机，Space暂停、R重置。详细操作、状态/坐标约定与headless smoke见[交互控制说明](../docs/INTERACTIVE_LABEL_CONTROL.md)。
