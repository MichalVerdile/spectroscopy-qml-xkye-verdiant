#!/usr/bin/env python3
"""Cross-platform entrypoint; run --help for the frozen raw-native protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectroscopy_qml.benchmarking.raw_runner import main

if __name__ == "__main__":
    main()
