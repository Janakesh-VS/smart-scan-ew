"""Makes `src/` importable for tests without requiring an editable install.

This keeps Phase 0 dependency-free (rule 13): no packaging tool run is
required just to execute `pytest`.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
