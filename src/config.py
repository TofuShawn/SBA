# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""config — the foundation module: repo paths, the config.toml
loader, engine/session config access and persistence, and the
write-path guard. Every other module imports from here; nothing here
imports them (set_engine_config late-imports cbridge for the DLL
reset to keep the graph acyclic)."""

import contextlib
import logging
import os
import re
import tempfile
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger("sba.config")

# the config file location; $SBA_CONFIG (applied below) and tests
# (monkeypatch.setattr(config, '_CONFIG_PATH', ...)) both override it
_CONFIG_PATH = os.path.join(ROOT, "config.toml")

# the only trees a write helper may create files under: this repo and the
# system temp directory (tests use pytest tmp_path)
_ALLOWED_ROOTS = (os.path.realpath(ROOT), os.path.realpath(tempfile.gettempdir()))


def writable_path(path):
    """Normalize a path and refuse anything the tool may not create files
    under: repo or system temp only, no '..' segments (path-traversal
    guard for the write helpers; reads use the same rule)."""
    raw = str(path)
    # Check the RAW path: realpath() resolves '..' away, so testing the
    # normalized string could never fire (a dead guard until D53 review).
    # Split on both separators so Windows paths behave on POSIX too.
    if ".." in raw.replace("\\", "/").split("/"):
        raise ValueError("path may not contain ..: %s" % raw)
    rp = os.path.realpath(raw)  # 1. 规范化
    if not any(rp == r or rp.startswith(r + os.sep) for r in _ALLOWED_ROOTS):
        raise ValueError("write path outside repo/temp: %s" % raw)  # 2. 限目录
    return rp


# kept so older call sites / external scripts do not break; new code uses
# the public name above
_writable_path = writable_path


# the $SBA_CONFIG override: applied HERE at import time, not in the entry
# shims. By the time a shim's `from SBA import main` returns, CONFIG below
# has already been loaded from _CONFIG_PATH — an entry-time mutation would
# leave reads on the old file while save_engine_config (which reads
# _CONFIG_PATH at call time) writes the new one. Import time is one place
# that every entry point hits identically.
if os.environ.get("SBA_CONFIG"):
    try:
        _CONFIG_PATH = writable_path(os.environ["SBA_CONFIG"])
    except ValueError as exc:
        # Say what the allowed locations are: the override file is BOTH
        # read and written, so it is subject to the same guard as saves.
        raise ValueError(
            "$SBA_CONFIG must live in this repo or the system temp dir "
            "(the override is read and written through the same path guard); "
            "%s" % exc
        ) from exc


# ============================================================
# Engine configuration (config.toml)
# ============================================================

_CFG_DEFAULTS = {
    "engine": {
        "rollout_heuristic": True,
        "tree_reuse": True,
        "dynamic_uct": True,
        "progressive_widening": False,
        "early_stop": False,
        "opening_book": True,
        "micro_tablebase": True,
        "symmetry": True,
        "object_pool": True,
        "bitboard": False,
        "c_rollout_pattern": False,
        "c_mcts": True,
        "c_minimax": True,
        "rust_game": True,  # D48: Rust state layer for AZ self-play
        "multithreaded": False,
        "use_lmr": True,
        "use_killers": True,
        "use_aspiration": True,
        "use_eval_ordering": True,
        "pro_time_ms": 8000,  # Minimax Pro per-move wall-clock cap (D51)
        "uct_c": 1.4,
        "rave_k": 150,
        "tt_max": 300000,
        "workers": 8,
        "reuse_cache": 32,
        "assistant_sims": 2000,  # AI assistant (Best Moves) search budget
    },
    "session": {"mcts_budget": 800, "minimax_depth": 4},
}

# "significantly visited" threshold, shared by every engine and by the
# assistant panel: a child needs at least
# max(MIN_VISIT_FLOOR, budget // MIN_VISIT_DIVISOR) visits before it may be
# called the best move. One definition, so the recommendation shown in the
# UI and the move the engine plays cannot disagree.
MIN_VISIT_FLOOR = 5
MIN_VISIT_DIVISOR = 100

# early-stop heuristic (mcts.py): after EARLY_STOP_AT of the budget, stop
# when the leader holds EARLY_STOP_SHARE of the visits and wins
# EARLY_STOP_WR of them.
EARLY_STOP_AT = 0.7
EARLY_STOP_SHARE = 0.6
EARLY_STOP_WR = 0.55

# How long a multiprocess self-play game may stay silent before the parent
# assumes its worker died and resubmits it. Generous on purpose: a slow CPU
# game with a big net is minutes, not an hour.
# Shared by alphazero (mp.Pool games) and rustselfplay (exe rounds, watchdog
# + default timeout): it lives here — the leaf of the import graph — because
# Task 9 imports rustselfplay FROM alphazero, so rustselfplay importing the
# constant back from alphazero would be a cycle.
MP_GAME_TIMEOUT = 3600.0

# self-play temperature floor: the campaign schedule max(TEMP_MIN, 1 - i/games)
# never samples argmax-blurrier than this, on either self-play path.
TEMP_MIN = 0.05


def _load_config():
    """Load config.toml as a dict; missing/broken file -> {}."""
    path = writable_path(_CONFIG_PATH)  # 规范化 + 禁止 .. + 限 repo/系统暂存目录
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        # a broken TOML must not silently masquerade as "all defaults" — the
        # next save_engine_config would overwrite the user's file with them
        log.warning("%s failed to load (%s); using code defaults", path, exc)
        return {}


CONFIG = _load_config()


def set_engine_config(overrides=None):
    """Reset the engine config to code defaults, then apply overrides."""
    global CONFIG
    base = {
        "engine": dict(_CFG_DEFAULTS["engine"]),
        "session": dict(_CFG_DEFAULTS["session"]),
    }
    if overrides:
        base["engine"].update(overrides.get("engine", {}))
        base["session"].update(overrides.get("session", {}))
    CONFIG = base


def cfg_engine(name, default=None):
    return CONFIG.get("engine", {}).get(name, _CFG_DEFAULTS["engine"].get(name, default))


def cfg_session(name, default=None):
    return CONFIG.get("session", {}).get(name, _CFG_DEFAULTS["session"].get(name, default))


# ============================================================
# TOML persistence
#
# A regex is not a TOML parser, so the rewrite is deliberately narrow:
#   * only [engine] / [session] scalar keys are touched;
#   * the key line must sit INSIDE its own section (the text is split on
#     section headers first, so a same-named key in another section is
#     never rewritten);
#   * the value pattern accepts every legal spelling -- sign, digit
#     separators, hex/oct/bin -- so a hand-written `300_000` is matched
#     instead of being mistaken for a missing key (that mistake used to
#     append a duplicate key and destroy the file);
#   * the result is re-parsed and compared against the overrides before
#     anything is written; a mismatch aborts the save.
# ============================================================

_SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")

# one legal TOML scalar: bool, int (with separators / radix prefixes) or
# float (incl. exponent). Used only to consume the value we are replacing.
_SCALAR_RE = (
    r"(?:true|false"
    r"|[+-]?(?:0[xX][0-9A-Fa-f_]+|0[oO][0-7_]+|0[bB][01_]+"
    r"|\d[\d_]*(?:\.[\d_]*)?(?:[eE][+-]?\d+)?"
    r"|\.\d[\d_]*(?:[eE][+-]?\d+)?))"
)


def _toml_literal(value):
    """Render a scalar the way TOML spells it; None for hand-managed types."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return None


def _split_sections(text):
    """Split into segments: a preamble (header None) plus one per [section]."""
    segments = []
    cur = {"header": None, "lines": []}
    for line in text.splitlines():
        match = _SECTION_RE.match(line.strip())
        if match:
            segments.append(cur)
            cur = {"header": match.group(1), "lines": [line]}
        else:
            cur["lines"].append(line)
    segments.append(cur)
    return segments


def _rewrite_text(text, overrides):
    """Rewrite only the given keys, each inside its own [section].

    Returns the new text. Keys with no line to rewrite are appended at the
    end of their own section (creating the section if the file lacks it).
    """
    wanted = {}
    for section in ("engine", "session"):
        for key, value in overrides.get(section, {}).items():
            literal = _toml_literal(value)
            if literal is not None:
                wanted.setdefault(section, {})[key] = literal

    segments = _split_sections(text)
    seen_sections = {seg["header"] for seg in segments if seg["header"]}
    for seg in segments:
        section = seg["header"]
        if section not in wanted:
            continue
        pending = dict(wanted[section])
        body = seg["lines"][1:]  # lines[0] is the [section] header
        for idx, line in enumerate(body):
            for key, literal in list(pending.items()):
                pattern = re.compile(
                    r"^(\s*" + re.escape(key) + r"\s*=\s*)(" + _SCALAR_RE + r")(\s*(?:#.*)?)$"
                )
                match = pattern.match(line)
                if match:
                    body[idx] = match.group(1) + literal + match.group(3)
                    del pending[key]
                    break
        # anything not found belongs at the end of this section, not at the
        # end of the file (which may be a different section entirely)
        body.extend(f"{key} = {literal}" for key, literal in pending.items())
        seg["lines"] = seg["lines"][:1] + body

    for section in ("engine", "session"):
        if section in wanted and section not in seen_sections:
            segments.append(
                {
                    "header": section,
                    "lines": [f"[{section}]"] + [f"{k} = {v}" for k, v in wanted[section].items()],
                }
            )

    return "\n".join(line for seg in segments for line in seg["lines"]).rstrip("\n") + "\n"


def _verify_text(text, overrides):
    """Re-parse the rewritten text and check every override landed.

    Returns None when the text is valid and every scalar matches, else a
    human-readable reason. This is what makes the regex safe: a wrong
    rewrite can never reach the disk.
    """
    try:
        parsed = tomllib.loads(text)
    except Exception as exc:
        return "rewritten text is not valid TOML: %s" % exc
    for section in ("engine", "session"):
        for key, value in overrides.get(section, {}).items():
            if _toml_literal(value) is None:
                continue
            got = parsed.get(section, {}).get(key, "<missing>")
            if type(got) is not type(value) or got != value:
                return "%s.%s came out as %r, expected %r" % (section, key, got, value)
    return None


def _atomic_write(path, text):
    """Write via a sibling temp file + os.replace: a crash mid-write must
    leave the previous TOML intact."""
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def save_engine_config(overrides=None):
    """Apply overrides at runtime AND persist them to config.toml (D28).

    The file is the source of truth: only the keys explicitly handed in are
    rewritten, so hand-managed values (the opening-book table, [session]
    tweaks) survive a Settings save. Comment lines survive too.

    A broken TOML is never overwritten, and a rewrite that fails to verify
    is never written at all -- the previous file stays untouched and the
    reason is logged.

    Reads config._CONFIG_PATH at call time (tests repoint it at tmp_path).
    """
    path = writable_path(_CONFIG_PATH)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        on_disk = tomllib.loads(text)
    except OSError:
        log.warning("save_engine_config: cannot read %s", path)
        return
    except Exception as exc:
        # never turn a broken file into a longer broken file
        log.warning(
            "save_engine_config: %s is not valid TOML (%s); refusing to overwrite", path, exc
        )
        return

    if overrides is None:
        # "Restore defaults": persist every scalar code default, but leave
        # hand-managed tables (the opening book) untouched.
        overrides = {
            "engine": {k: v for k, v in _CFG_DEFAULTS["engine"].items() if not isinstance(v, dict)},
            "session": dict(_CFG_DEFAULTS["session"]),
        }

    # defaults <- file <- overrides. Keeping the file in the chain is what
    # stops a save from silently discarding values the UI never touched.
    global CONFIG
    CONFIG = {
        "engine": {
            **_CFG_DEFAULTS["engine"],
            **on_disk.get("engine", {}),
            **overrides.get("engine", {}),
        },
        "session": {
            **_CFG_DEFAULTS["session"],
            **on_disk.get("session", {}),
            **overrides.get("session", {}),
        },
    }

    new_text = _rewrite_text(text, overrides)
    problem = _verify_text(new_text, overrides)
    if problem:
        log.error("save_engine_config: refusing to write %s (%s)", path, problem)
        return
    _atomic_write(path, new_text)
    log.info("save_engine_config: written to %s", path)
