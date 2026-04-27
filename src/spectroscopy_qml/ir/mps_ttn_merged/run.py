import argparse
import sys
from pathlib import Path

src_dir = Path(__file__).parents[3]
sys.path.insert(0, str(src_dir))

from spectroscopy_qml.ir.mps_ttn_merged.train import main


if __name__ == "__main__":
    main()