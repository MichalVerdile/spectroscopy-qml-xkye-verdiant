#!/usr/bin/env python3
"""Compatibility wrapper for corrected TTN 10.2 error analysis.

The old script used a sequential test split, threshold 0.5, and a wrong label
mapping. This wrapper routes to the verified analysis generator instead.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "generate_correct_error_analysis.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
