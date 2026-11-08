# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""ctypes interop with native/sba_engine.dll — the single boundary between
the Python AI layer and the C backends (D27).

Owns: DLL discovery/loading, the ABI signatures, state serialization
(UltimateGame / BitUltimateGame -> the flat C encoding) and thin wrappers:
rollout, full MCTS (with root stats), minimax (normal/ultimate), flat MCTS.
Every wrapper returns None when the DLL is unavailable so ai.py can fall
back to the pure-Python engines.
"""

import ctypes
import os
import random
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE_DLL = os.path.join(_ROOT, 'native', 'sba_engine.dll')
_LEGACY_DLL = os.path.join(_ROOT, 'native', 'sba_rollout.dll')

_lib = None
_has_engine = False
_load_lock = threading.Lock()

_XO = {1: 'X', 2: 'O', 3: 'D'}


def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _load():
    """Load sba_engine.dll (or the legacy rollout-only DLL); idempotent and
    thread-safe (the UI runs the assistant and the AI move search in two
    threads that can reach here simultaneously)."""
    global _lib, _has_engine
    if _lib is not None:
        return _lib
    with _load_lock:
        if _lib is not None:
            return _lib
    path = _ENGINE_DLL if os.path.isfile(_ENGINE_DLL) else (
        _LEGACY_DLL if os.path.isfile(_LEGACY_DLL) else None)
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(path)
        ptr = ctypes.POINTER(ctypes.c_int32)
        rollout_args = [ptr, ptr, ctypes.c_int32, ctypes.c_int32,
                        ctypes.c_uint64]
        _bind(lib, 'sba_rollout', ctypes.c_int32, rollout_args)
        _bind(lib, 'sba_rollout_pattern', ctypes.c_int32, rollout_args)
        state_args = [ptr, ptr, ctypes.c_int32, ctypes.c_int32]
        if os.path.basename(path) == 'sba_engine.dll':
            _bind(lib, 'sba_rollout_trace', ctypes.c_int32,
                  state_args + [ctypes.c_uint64,
                                ctypes.POINTER(ctypes.c_int32),
                                ctypes.POINTER(ctypes.c_int32),
                                ctypes.c_int32])
            _bind(lib, 'sba_mcts_search', ctypes.c_int32,
                  state_args + [ctypes.c_int32, ctypes.c_uint64,
                                ctypes.c_int32,
                                ctypes.POINTER(ctypes.c_int32),
                                ctypes.POINTER(ctypes.c_int32),
                                ptr, ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_int32),
                                ctypes.c_void_p, ctypes.c_int32,
                                ctypes.c_void_p, ctypes.c_int32])
            _bind(lib, 'sba_minimax_normal', ctypes.c_int32,
                  [ptr, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
                   ctypes.POINTER(ctypes.c_double)])
            _bind(lib, 'sba_minimax_ultimate', ctypes.c_int32,
                  state_args + [ctypes.c_int32,
                                ctypes.POINTER(ctypes.c_int32)])
            _bind(lib, 'sba_flat_mcts', ctypes.c_int32,
                  state_args + [ctypes.c_int32, ctypes.c_uint64,
                                ctypes.c_int32,
                                ctypes.POINTER(ctypes.c_int32)])
            _has_engine = True
        _lib = lib
    except OSError:
        _lib = None
    return _lib


def available(engine=False):
    """True when a DLL is loaded; engine=True additionally requires the full
    sba_engine.dll (not just the legacy rollout DLL)."""
    _load()
    return _lib is not None and (_has_engine or not engine)


def serialize_ultimate(state):
    """Flatten an UltimateGame / BitUltimateGame into the C ABI:
    (cells[81], macro[9], player 1/2, active -1..8)."""
    cells = [0] * 81
    macro = [0] * 9
    if type(state).__name__ == 'BitUltimateGame':
        for idx in range(81):
            b = 1 << idx
            if state.x & b:
                cells[idx] = 1
            elif state.o & b:
                cells[idx] = 2
        for m in range(9):
            if (state.mx >> m) & 1:
                macro[m] = 1
            elif (state.mo >> m) & 1:
                macro[m] = 2
            elif ((state.x | state.o) >> (m * 9)) & 0x1FF == 0x1FF:
                macro[m] = 3
    else:
        for m in range(9):
            row = state.micro[m]
            base = m * 9
            for i in range(9):
                c = row[i]
                if c == 'X':
                    cells[base + i] = 1
                elif c == 'O':
                    cells[base + i] = 2
            w = state.macro[m]
            if w == 'X':
                macro[m] = 1
            elif w == 'O':
                macro[m] = 2
            elif w == 'D':
                macro[m] = 3
    active = -1 if state.active_macro is None else state.active_macro
    player = 1 if state.current == 'X' else 2
    return cells, macro, player, active


def _arrays(cells, macro):
    return (ctypes.c_int32 * 81)(*cells), (ctypes.c_int32 * 9)(*macro)


def rollout_state(state, pattern=False, rng_seed=None):
    """One native rollout to terminal; returns 'X' / 'O' / 'D' (None if the
    DLL is missing)."""
    lib = _load()
    if lib is None:
        return None
    fn = lib.sba_rollout_pattern if pattern else lib.sba_rollout
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    return _XO[fn(ca, ma, player, active, seed)]


def rollout_trace(state, rng_seed=None, pattern=False):
    """Native rollout that also returns the move sequence (for the RAVE/GRAVE
    AMAF statistics). Policy follows the plain heuristic kernel, or the D26
    pattern kernel when `pattern` is set — the same switch the C MCTS tree
    takes, so the whole family shares one playout policy. Returns
    (result, [moves]) or (None, [])."""
    lib = _load()
    if lib is None or not _has_engine:
        return None, []
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    out_moves = (ctypes.c_int32 * 256)()
    out_len = ctypes.c_int32(0)
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    res = lib.sba_rollout_trace(ca, ma, player, active, seed,
                                out_moves, ctypes.byref(out_len),
                                1 if pattern else 0)
    seq = [(out_moves[k] // 9, out_moves[k] % 9) for k in range(out_len.value)]
    return _XO[res], seq


def mcts_search(state, iterations, use_pattern=False, rng_seed=None, prev=None):
    """Full native MCTS. Returns (best_move, stats, tree) where stats is a
    list of (move, visits, wins, mover) for the root children — wins counted
    from the mover's perspective, matching ai.py's MCTSNode; tree is the
    ctypes arena buffer holding the finished tree (pass it back as `prev` on
    the next search to reuse it, D30). Returns (None, [], None) when the full
    engine DLL is unavailable.

    Concurrency: the node arena is allocated PER CALL — the UI runs the
    assistant analysis (AnalysisWorker) and the AI move search (AIWorker)
    concurrently, and a shared buffer made them stomp each other into an
    infinite loop (the "stuck on move 3" bug). Allocation is ~1 MB and
    cheap. With `prev`, the C side walks the previous tree down to the
    current position and reuses the matching subtree (single-threaded
    searches only: concurrent writers must not share the prev buffer).
    """
    lib = _load()
    if lib is None or not _has_engine:
        return None, [], None
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    best = ctypes.c_int32(-1)
    count = ctypes.c_int32(0)
    moves = (ctypes.c_int32 * 81)()
    visits = (ctypes.c_double * 81)()
    wins = (ctypes.c_double * 81)()
    mover = (ctypes.c_int32 * 81)()
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    # per-call arena, 512 B per node upper bound. With tree reuse the kept
    # subtree (<= previous iterations) is copied in first, hence the 2x.
    factor = 2 if prev is not None else 1
    buf = (ctypes.c_uint8 * ((iterations * factor + 64) * 512))()
    prev_buf = None
    prev_bytes = 0
    if prev is not None:
        prev_buf, prev_bytes = prev
    r = lib.sba_mcts_search(ca, ma, player, active, iterations, seed,
                            1 if use_pattern else 0,
                            ctypes.byref(best), ctypes.byref(count),
                            moves, visits, wins, mover, buf, len(buf),
                            prev_buf, prev_bytes)
    if r == -3 and prev is not None:
        # reused subtree didn't fit: retry from a fresh root
        r = lib.sba_mcts_search(ca, ma, player, active, iterations, seed,
                                1 if use_pattern else 0,
                                ctypes.byref(best), ctypes.byref(count),
                                moves, visits, wins, mover, buf, len(buf),
                                None, 0)
    if r < 0:
        return None, [], None
    stats = [(moves[k], visits[k], wins[k], mover[k]) for k in range(count.value)]
    mv = best.value if best.value >= 0 else None
    tree = (buf, len(buf)) if mv is not None else None
    return mv, stats, tree


def minimax_normal(board9, player):
    """Full 3x3 solve; returns (move, score). None if unavailable."""
    lib = _load()
    if lib is None or not _has_engine:
        return None
    b = (ctypes.c_int32 * 9)(
        *[1 if c == 'X' else (2 if c == 'O' else 0) for c in board9])
    best = ctypes.c_int32(-1)
    score = ctypes.c_double(0.0)
    lib.sba_minimax_normal(b, 1 if player == 'X' else 2,
                           ctypes.byref(best), ctypes.byref(score))
    return (best.value if best.value >= 0 else None), score.value


def minimax_ultimate(state, depth):
    """Depth-limited alpha-beta; returns the best move or None."""
    lib = _load()
    if lib is None or not _has_engine:
        return None
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    best = ctypes.c_int32(-1)
    lib.sba_minimax_ultimate(ca, ma, player, active, depth, ctypes.byref(best))
    mv = best.value
    if mv < 0:
        return None
    return (mv // 9, mv % 9)


def flat_mcts(state, per_move, use_pattern=False, rng_seed=None):
    """Root-level playouts per legal move; returns the best move or None."""
    lib = _load()
    if lib is None or not _has_engine:
        return None
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    best = ctypes.c_int32(-1)
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    lib.sba_flat_mcts(ca, ma, player, active, per_move, seed,
                      1 if use_pattern else 0, ctypes.byref(best))
    mv = best.value
    if mv < 0:
        return None
    return (mv // 9, mv % 9)
