"""CLI entry point for the GenoView preview backend."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Visualization.backends.genoview_backend import main  # noqa: E402


if __name__ == "__main__":
    main()
