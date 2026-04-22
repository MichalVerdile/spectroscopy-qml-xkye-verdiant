#!/usr/bin/env python3
"""Compatibility wrapper for corrected error-analysis plots.

This legacy entry point used to contain hard-coded error tables and an
incorrect functional-group index mapping. It now delegates to the verified
generator, which computes CNN metrics from the saved CNN result pickle and TTN
10.2 metrics from the real checkpoint, split artifact, and selected thresholds.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "generate_correct_error_analysis.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
