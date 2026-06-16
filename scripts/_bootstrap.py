"""Make `llmbrain` importable when running scripts directly (no install needed).

Colab can also `pip install -e .`; this keeps `python scripts/xx.py` working regardless.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
