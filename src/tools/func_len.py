# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Function-length report — the second half of the `dev.py complexity`
warn-only task (C901 being the first half, ruff's side).

Measurement (deliberately naive and fixed): a function's length is
`last body statement's end_lineno - def lineno + 1` — the def line through
the block end, decorators NOT counted, blank/comment lines inside the body
included. Docstrings count as body. Nested defs are counted inside their
parent AND reported as their own entries, so a hotspot with inner closures
shows both views. A module-level `if __name__` block is not a function.

Anything above LIMIT (50) is printed, longest first. Report-only: exits 0
no matter what it finds — the GamePage split (H2) decides whether this
graduates into a gate. Run: `python src/tools/func_len.py [dir]` (default
scan root: src).
"""

import ast
import os
import sys

LIMIT = 50


def _span(node):
    """def line to block end, decorators excluded (see module docstring)."""
    end = max(getattr(stmt, "end_lineno", node.lineno) for stmt in node.body)
    return end - node.lineno + 1


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "src"
    rows = []
    files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            files += 1
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    span = _span(node)
                    if span > LIMIT:
                        rows.append((span, os.path.normpath(path), node.lineno, node.name))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    if not rows:
        print(f"function length: nothing over {LIMIT} lines ({files} files scanned)")
        return 0
    print(f"functions over {LIMIT} lines (def line -> block end, {files} files scanned):")
    for span, path, lineno, name in rows:
        print(f"  {span:>4}  {path}:{lineno}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
