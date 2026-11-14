# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""rich console shared by the CLI tools (train / bench / --build-tables).

rich ships as a dependency of textual (the TUI front end) and is now also
an explicit requirement — the CLI tools use it for tables and live
progress. Everything degrades: when rich is missing the call sites fall
back to plain prints, so a bare checkout still works, and on a
non-terminal (CI logs, pipes) rich strips the styling itself."""

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    _RICH = True
except ImportError:  # pragma: no cover - bare-venv fallback
    _RICH = False

_console = None


def have_rich():
    return _RICH


def console():
    """The shared rich Console (built lazily; None when rich is missing)."""
    global _console
    if _console is None and _RICH:
        _console = Console()
    return _console


def bench_table(title, header, rows):
    """Render the benchmark rows as a rich Table, or return None so the
    caller falls back to its plain prints. ``rows`` cells may be (text,
    style) tuples for per-cell styling."""
    if not _RICH:
        return None
    t = Table(title=title, box=box.SIMPLE_HEAVY, title_justify='left',
              pad_edge=False)
    for col in header:
        t.add_column(col, justify='left' if col in ('A', 'B') else 'right')
    for row in rows:
        t.add_row(*[cell if not isinstance(cell, tuple) else
                    f'[{cell[1]}]{cell[0]}[/{cell[1]}]' for cell in row])
    return t
