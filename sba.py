# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""sba.py — the root-level entry point: every command starts here.

The implementation lives in src/SBA.py; this shim only puts src/ on
sys.path and forwards argv. Lowercase on purpose — a root `SBA.py`
would collide with the src/SBA.py module on case-insensitive
filesystems (Windows) once `import SBA` resolves.

    python sba.py                    # PySide6 desktop app (default)
    python sba.py --tui              # Textual terminal app
    python sba.py --bench            # engine win-rate benchmark
    python sba.py --build-tables     # rebuild data/tt_normal.bin
    python sba.py --self-test        # headless test suite
    python sba.py --train-az ...     # AlphaZero training (forwards args)
    python sba.py --eval-az ...      # AlphaZero evaluation (forwards args)
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from SBA import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
