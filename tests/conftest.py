"""Shared pytest fixture: engine/config reset and a deterministic RNG.

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
