"""Run source-tree scripts from any working directory without requiring activation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
