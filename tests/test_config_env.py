# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""$SBA_CONFIG: applied at config IMPORT time (D53), so every entry point —
the root shim, src/SBA.py, anything that imports ai — reads AND saves the
same override file. Runs a subprocess because the variable must be set
before `import config` executes."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sba_config_overrides_read_and_write(tmp_path):
    """The user's repro: an override file with uct_c=9.9 / symmetry=false
    must be visible through cfg_engine at import, and save_engine_config
    must write the same file (no read-old/save-new split)."""
    cfg = tmp_path / 'override.toml'
    cfg.write_text('[engine]\nuct_c = 9.9\nsymmetry = false\n',
                   encoding='utf-8')
    env = dict(os.environ, SBA_CONFIG=str(cfg))
    code = (
        'import sys; sys.path.insert(0, r"%s"); import config; '
        'assert config.cfg_engine("uct_c") == 9.9, config.cfg_engine("uct_c"); '
        'assert config.cfg_engine("symmetry") is False; '
        'config.save_engine_config({"engine": {"uct_c": 8.8}}); '
        'assert config.cfg_engine("uct_c") == 8.8; print("OK")'
        % os.path.join(ROOT, 'src')
    )
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert 'OK' in r.stdout
    # the save landed in the OVERRIDE file, not the repo config.toml
    assert 'uct_c = 8.8' in cfg.read_text(encoding='utf-8')


def test_sba_config_rejects_paths_outside_the_write_guard():
    """The override rides the same write-path guard as every other path:
    outside repo/temp it must fail loudly at import, not silently fall
    back to the repo config.toml.

    Build the path from ROOT's PARENT so it is outside the allow-list on
    BOTH platforms. A hardcoded drive-letter path (Z:\\...) is only a
    filename under cwd on POSIX, where realpath would resolve it INSIDE
    the repo and the guard would let it through — the Linux CI runner
    used to fail this very assertion.
    """
    outside = os.path.join(os.path.dirname(ROOT), 'sba_guard_outside', 'c.toml')
    env = dict(os.environ, SBA_CONFIG=outside)
    code = ('import sys; sys.path.insert(0, r"%s"); '
            'import config; print("LOADED")'
            % os.path.join(ROOT, 'src'))
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert 'LOADED' not in r.stdout
    assert 'outside repo/temp' in r.stderr


def test_sba_config_rejects_parent_traversal_before_normalization():
    """The '..' check must run on the RAW path: realpath() resolves '..'
    away, so testing the normalized string could never fire. This path
    lands inside the repo after normalization and would otherwise pass.
    """
    traversal = os.path.join(ROOT, 'src', '..', 'override.toml')
    env = dict(os.environ, SBA_CONFIG=traversal)
    code = ('import sys; sys.path.insert(0, r"%s"); '
            'import config; print("LOADED")'
            % os.path.join(ROOT, 'src'))
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert 'LOADED' not in r.stdout
    assert 'may not contain ..' in r.stderr
