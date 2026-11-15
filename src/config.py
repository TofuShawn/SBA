# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""config — the foundation module: repo paths, the config.toml
loader, engine/session config access and persistence, and the
write-path guard. Every other module imports from here; nothing here
imports them (set_engine_config late-imports cbridge for the DLL
reset to keep the graph acyclic)."""

import logging
import os
import tempfile
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger('sba.config')

# the config file location; tests repoint this at tmp_path via
# monkeypatch.setattr(config, '_CONFIG_PATH', ...) -- no env involved
_CONFIG_PATH = os.path.join(ROOT, 'config.toml')

# the only trees a write helper may create files under: this repo and the
# system temp directory (tests use pytest tmp_path)
_ALLOWED_ROOTS = (os.path.realpath(ROOT), os.path.realpath(tempfile.gettempdir()))


def _writable_path(path):
    """Normalize a path and refuse anything the tool may not create files
    under: repo or system temp only, no '..' segments (path-traversal
    guard for the write helpers; reads use the same rule)."""
    rp = os.path.realpath(str(path))                      # 1. 规范化
    if '..' in rp.split(os.sep):                          # 2. 禁止 ..
        raise ValueError('path may not contain ..: %s' % path)
    if not any(rp == r or rp.startswith(r + os.sep) for r in _ALLOWED_ROOTS):
        raise ValueError('write path outside repo/temp: %s' % path)  # 3. 限目录
    return rp


# the $SBA_CONFIG override: applied HERE at import time, not in the entry
# shims. By the time a shim's `from SBA import main` returns, CONFIG below
# has already been loaded from _CONFIG_PATH — an entry-time mutation would
# leave reads on the old file while save_engine_config (which reads
# _CONFIG_PATH at call time) writes the new one. Import time is one place
# that every entry point hits identically.
if os.environ.get('SBA_CONFIG'):
    _CONFIG_PATH = _writable_path(os.environ['SBA_CONFIG'])


# ============================================================
# Engine configuration (config.toml)
# ============================================================

_CFG_DEFAULTS = {
    'engine': {
        'rollout_heuristic': True, 'tree_reuse': True, 'dynamic_uct': True,
        'progressive_widening': False, 'early_stop': False,
        'opening_book': True, 'micro_tablebase': True,
        'symmetry': True, 'object_pool': True, 'bitboard': False,
        'c_rollout_pattern': False,
        'c_mcts': True, 'c_minimax': True,
        'rust_game': True,        # D48: Rust state layer for AZ self-play
        'multithreaded': False, 'use_lmr': True, 'use_killers': True,
        'use_aspiration': True, 'use_eval_ordering': True,
        'uct_c': 1.4, 'rave_k': 150, 'tt_max': 300000,
        'workers': 8, 'reuse_cache': 32,
        'assistant_sims': 2000,   # AI assistant (Best Moves) search budget
    },
    'session': {'mcts_budget': 800, 'minimax_depth': 4},
}


def _load_config():
    """Load config.toml as a dict; missing/broken file -> {}."""
    path = _writable_path(_CONFIG_PATH)   # 规范化 + 禁止 .. + 限 repo/系统暂存目录
    try:
        with open(path, 'rb') as fh:
            return tomllib.load(fh)
    except Exception as exc:
        # a broken TOML must not silently masquerade as "all defaults" — the
        # next save_engine_config would overwrite the user's file with them
        log.warning('config.toml failed to load (%s); using code defaults', exc)
        return {}


CONFIG = _load_config()


def set_engine_config(overrides=None):
    """Reset the engine config to code defaults, then apply overrides."""
    global CONFIG
    base = {
        'engine': dict(_CFG_DEFAULTS['engine']),
        'session': dict(_CFG_DEFAULTS['session']),
    }
    if overrides:
        base['engine'].update(overrides.get('engine', {}))
        base['session'].update(overrides.get('session', {}))
    CONFIG = base
    # allow tests to flip c_rollout and re-resolve the DLL state
    # (late import: cbridge depends on this module)
    from cbridge import _reset_dll_state
    _reset_dll_state()


def cfg_engine(name, default=None):
    return CONFIG.get('engine', {}).get(name, _CFG_DEFAULTS['engine'].get(name, default))


def cfg_session(name, default=None):
    return CONFIG.get('session', {}).get(name, _CFG_DEFAULTS['session'].get(name, default))


def save_engine_config(overrides=None):
    """Apply overrides at runtime AND persist them to config.toml (D28).

    The file is the source of truth: only the keys explicitly handed in are
    rewritten, so hand-managed values (floats, the opening-book table,
    [session] tweaks) survive a Settings save. A broken TOML is never
    overwritten -- it is reported and left for the user to repair. Comment
    lines survive: each key line is rewritten in place via regex.

    Reads config._CONFIG_PATH at call time (tests repoint it at tmp_path).
    """
    import re
    path = _writable_path(_CONFIG_PATH)   # 规范化 + 禁止 .. + 限 repo/系统暂存目录
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        on_disk = tomllib.loads(text)
    except OSError:
        log.warning('save_engine_config: cannot read %s', path)
        return
    except Exception as exc:
        # never turn a broken file into a longer broken file
        log.warning('save_engine_config: %s is not valid TOML (%s); '
                    'refusing to overwrite', path, exc)
        return

    if overrides is None:
        # "Restore defaults": persist every scalar code default, but leave
        # hand-managed tables (the opening book) untouched.
        overrides = {
            'engine': {k: v for k, v in _CFG_DEFAULTS['engine'].items()
                       if not isinstance(v, dict)},
            'session': dict(_CFG_DEFAULTS['session']),
        }

    # defaults <- file <- overrides. Keeping the file in the chain is what
    # stops a save from silently discarding values the UI never touched.
    global CONFIG
    CONFIG = {
        'engine': {**_CFG_DEFAULTS['engine'],
                   **on_disk.get('engine', {}),
                   **overrides.get('engine', {})},
        'session': {**_CFG_DEFAULTS['session'],
                    **on_disk.get('session', {}),
                    **overrides.get('session', {})},
    }

    appended = []
    for section in ('engine', 'session'):
        for key, val in overrides.get(section, {}).items():
            if isinstance(val, bool):
                literal = 'true' if val else 'false'
                value_re = r'(?:true|false)'
            elif isinstance(val, int):
                literal, value_re = str(val), r'-?\d+'
            elif isinstance(val, float):
                literal = repr(val)
                value_re = r'-?[0-9.]+(?:[eE][+-]?\d+)?'
            else:
                continue  # tables / strings are hand-managed, never rewritten
            pattern = rf'(?m)^(\s*{re.escape(key)}\s*=\s*){value_re}(\s*(?:#.*)?)$'
            text, n = re.subn(
                pattern, lambda m: m.group(1) + literal + m.group(2), text)
            if not n:
                appended.append((section, key, literal))

    if appended:
        out, seen = [], set()
        for line in text.rstrip('\n').split('\n'):
            out.append(line)
            m = re.match(r'^\[([^\]]+)\]', line.strip())
            if m:
                sec = m.group(1)
                seen.add(sec)
                for a in [a for a in appended if a[0] == sec]:
                    out.append(f'{a[1]} = {a[2]}')
                appended = [a for a in appended if a[0] != sec]
        for sec in ('engine', 'session'):
            rest = [a for a in appended if a[0] == sec]
            if rest:
                if sec not in seen:
                    out.append(f'[{sec}]')
                out.extend(f'{a[1]} = {a[2]}' for a in rest)
        text = '\n'.join(out) + '\n'

    # bounded write: _writable_path has normalized the target and limited it
    # to the repo or the system temp dir; _load_config falls back to code
    # defaults if a crash mid-write ever leaves a partial TOML
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    log.info('save_engine_config: written to %s', path)
