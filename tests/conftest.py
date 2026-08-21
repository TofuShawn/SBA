"""Shared pytest fixtures: repo path, config reset, and deterministic RNG."""

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import reset_engine_caches, set_engine_config  # noqa: E402


@pytest.fixture(autouse=True)
def clean_engine():
    set_engine_config({})  # ignore any user sba.toml overrides
    random.seed(12345)
    reset_engine_caches()
    yield
    reset_engine_caches()
