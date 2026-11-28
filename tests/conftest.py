"""Shared pytest fixtures: engine/config reset, a deterministic RNG, and
bit-level seeding for the Rust state layer.

The `src/` import path comes from pyproject.toml
(`tool.pytest.ini_options.pythonpath`) — no sys.path surgery here.
"""

import random

import pytest

from engines.ai import reset_engine_caches, set_engine_config


@pytest.fixture(autouse=True)
def clean_engine():
    set_engine_config({})  # ignore any user config.toml overrides
    random.seed(12345)
    reset_engine_caches()
    yield
    reset_engine_caches()


def rust_game_from_bits(px=None, po=None, mx=0, mo=0, current=0, active=-1):
    """A RustUltimateGame seeded from bit masks — the way to build exact
    positions (forced wins, terminal boards) now that the pure-Python
    UltimateGame is gone. px/po: per-micro 9-bit X/O masks (a 9-sequence or
    a {macro: mask} dict); mx/mo: the 9-bit macro badge masks; current
    0=X/1=O; active -1 = free routing."""
    from az.rustbridge import RustUltimateGame, SbaState

    def at(spec, m):
        if spec is None:
            return 0
        if isinstance(spec, dict):
            return spec.get(m, 0)
        return spec[m] if m < len(spec) else 0

    st = SbaState()
    for m in range(9):
        st.px[m] = at(px, m)
        st.po[m] = at(po, m)
    st.mx = mx
    st.mo = mo
    st.current = current
    st.active = active
    return RustUltimateGame.from_sba_state(st)
