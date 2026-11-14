# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""ctypes interop with native/sba_engine.dll — the single boundary between
the Python AI layer and the C backends (D27/D37).

Owns: DLL discovery/loading, the ABI signatures, state serialization
(UltimateGame / BitUltimateGame -> the flat C encoding) and thin wrappers:
rollout, full MCTS (with root stats), minimax (normal/ultimate), flat MCTS.
Every wrapper returns None when the DLL is unavailable so ai.py can fall
back to the pure-Python engines.

D37: SbaSearchOpts carries the search switches into C, and the MCTS arena
is owned by the DLL. _CTree wraps the opaque handle and releases it on GC,
so tree reuse (D30) still works without Python ever touching C memory.
"""

import ctypes
import os
import random
import threading
import weakref

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE_DLL = os.path.join(_ROOT, 'native', 'sba_engine.dll')
_LEGACY_DLL = os.path.join(_ROOT, 'native', 'sba_rollout.dll')

_lib = None
_has_engine = False
_load_lock = threading.Lock()

_XO = {1: 'X', 2: 'O', 3: 'D'}


class SearchOpts(ctypes.Structure):
    """Mirror of SbaSearchOpts in sba_ultt.h. Keep the field order in sync."""
    _fields_ = [
        ('uct_c', ctypes.c_double),
        ('dynamic_uct', ctypes.c_int32),
        ('rollout_heuristic', ctypes.c_int32),
        ('micro_tablebase', ctypes.c_int32),
        ('symmetry', ctypes.c_int32),
        ('use_pattern', ctypes.c_int32),
    ]


DEFAULT_OPTS = {
    'uct_c': 1.4,
    'dynamic_uct': True,
    'rollout_heuristic': True,
    'micro_tablebase': True,
    'symmetry': True,
    'use_pattern': False,
}


def make_opts(values=None, **overrides):
    """Build a SearchOpts from a dict; unknown keys are ignored."""
    merged = dict(DEFAULT_OPTS)
    if values:
        merged.update({k: v for k, v in values.items() if k in merged})
    merged.update({k: v for k, v in overrides.items() if k in merged})
    return SearchOpts(
        float(merged['uct_c']),
        int(bool(merged['dynamic_uct'])),
        int(bool(merged['rollout_heuristic'])),
        int(bool(merged['micro_tablebase'])),
        int(bool(merged['symmetry'])),
        int(bool(merged['use_pattern'])),
    )


def _free_tree(handle):
    lib = _lib
    if lib is None or not handle:
        return
    try:
        lib.sba_tree_free(ctypes.c_void_p(handle))
    except (AttributeError, OSError):
        pass


class _CTree:
    """Owner of one C-allocated search tree. The handle is released when the
    wrapper is garbage-collected, so analysis searches that discard their
    tree do not leak the arena."""

    __slots__ = ('_handle', '_finalizer', '__weakref__')

    def __init__(self, handle):
        self._handle = ctypes.c_void_p(handle)
        self._finalizer = weakref.finalize(self, _free_tree, handle)

    def close(self):
        self._finalizer()


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
        # load AND bind under the lock: a second thread entering mid-bind
        # would otherwise call symbols whose argtypes are not declared yet
        path = _ENGINE_DLL if os.path.isfile(_ENGINE_DLL) else (
            _LEGACY_DLL if os.path.isfile(_LEGACY_DLL) else None)
        if path is None:
            return None
        try:
            lib = ctypes.CDLL(path)
            ptr = ctypes.POINTER(ctypes.c_int32)
            dbl = ctypes.POINTER(ctypes.c_double)
            void = ctypes.c_void_p
            opts_ptr = ctypes.POINTER(SearchOpts)
            rollout_args = [ptr, ptr, ctypes.c_int32, ctypes.c_int32,
                            ctypes.c_uint64]
            state_args = [ptr, ptr, ctypes.c_int32, ctypes.c_int32]
            _bind(lib, 'sba_rollout', ctypes.c_int32, rollout_args)
            _bind(lib, 'sba_rollout_pattern', ctypes.c_int32, rollout_args)
            if os.path.basename(path) == 'sba_engine.dll':
                _bind(lib, 'sba_rollout_random', ctypes.c_int32, rollout_args)
                _bind(lib, 'sba_rollout_trace', ctypes.c_int32,
                      state_args + [ctypes.c_uint64, ptr, ptr, opts_ptr])
                # cells, macro, player, active, iterations, seed, opts,
                # out_best, out_count, out_moves, out_visits, out_wins,
                # out_mover, out_draws, out_tree, prev_tree   (16 args)
                _bind(lib, 'sba_mcts_search', ctypes.c_int32,
                      state_args + [ctypes.c_int32, ctypes.c_uint64, opts_ptr,
                                    ptr, ptr, ptr, dbl, dbl, ptr, ptr,
                                    ctypes.POINTER(void), void])
                _bind(lib, 'sba_tree_free', ctypes.c_int32, [void])
                _bind(lib, 'sba_minimax_normal', ctypes.c_int32,
                      [ptr, ctypes.c_int32, ptr, dbl])
                _bind(lib, 'sba_minimax_ultimate', ctypes.c_int32,
                      state_args + [ctypes.c_int32, opts_ptr, ptr])
                _bind(lib, 'sba_flat_mcts', ctypes.c_int32,
                      state_args + [ctypes.c_int32, ctypes.c_uint64,
                                    opts_ptr, ptr])
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


def rollout_state(state, pattern=False, rng_seed=None, opts=None):
    """One native rollout to terminal; returns 'X' / 'O' / 'D' (None if the
    DLL is missing, or when a legacy DLL cannot serve an opts-driven policy:
    ai.py then falls back to the Python rollout)."""
    lib = _load()
    if lib is None:
        return None
    cfg = dict(DEFAULT_OPTS)
    if opts:
        cfg.update({k: v for k, v in opts.items() if k in cfg})
    if pattern:
        cfg['use_pattern'] = True
    if not cfg['rollout_heuristic']:
        if not _has_engine:
            return None
        fn = lib.sba_rollout_random
    else:
        fn = lib.sba_rollout_pattern if cfg['use_pattern'] else lib.sba_rollout
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    return _XO[fn(ca, ma, player, active, seed)]


def rollout_trace(state, rng_seed=None, pattern=False, opts=None):
    """Native rollout that also returns the move sequence (for the RAVE/GRAVE
    AMAF statistics). The playout policy comes from opts: uniform random when
    rollout_heuristic is off, else plain / D26 pattern. Returns
    (result, [moves]) or (None, [])."""
    lib = _load()
    if lib is None or not _has_engine:
        return None, []
    cfg = dict(DEFAULT_OPTS)
    if opts:
        cfg.update({k: v for k, v in opts.items() if k in cfg})
    if pattern:
        cfg['use_pattern'] = True
    c_opts = make_opts(cfg)
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    out_moves = (ctypes.c_int32 * 256)()
    out_len = ctypes.c_int32(0)
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    res = lib.sba_rollout_trace(ca, ma, player, active, seed,
                                out_moves, ctypes.byref(out_len),
                                ctypes.byref(c_opts))
    seq = [(out_moves[k] // 9, out_moves[k] % 9) for k in range(out_len.value)]
    return _XO[res], seq


def mcts_search(state, iterations, use_pattern=False, rng_seed=None,
                prev=None, opts=None):
    """Full native MCTS. Returns (best_move, stats, tree) where stats is a
    list of (move, visits, wins, mover, draws) for the root children, and
    tree is a _CTree owning the C-allocated arena (pass it back as `prev`
    on the next search to reuse it, D30). Returns (None, [], None) when the
    full engine DLL is unavailable.

    The arena lives inside the DLL and is allocated per call, so the
    assistant analysis and the AI move search can run concurrently without
    sharing writable memory. Handles are released by _CTree's finalizer.
    """
    lib = _load()
    if lib is None or not _has_engine:
        return None, [], None
    cfg = dict(DEFAULT_OPTS)
    if opts:
        cfg.update({k: v for k, v in opts.items() if k in cfg})
    if use_pattern:
        cfg['use_pattern'] = True
    c_opts = make_opts(cfg)
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    best = ctypes.c_int32(-1)
    count = ctypes.c_int32(0)
    moves = (ctypes.c_int32 * 81)()
    visits = (ctypes.c_double * 81)()
    wins = (ctypes.c_double * 81)()
    mover = (ctypes.c_int32 * 81)()
    draws = (ctypes.c_int32 * 81)()
    out_tree = ctypes.c_void_p()
    prev_handle = prev._handle if prev is not None else None
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    r = lib.sba_mcts_search(ca, ma, player, active, iterations, seed,
                            ctypes.byref(c_opts),
                            ctypes.byref(best), ctypes.byref(count),
                            moves, visits, wins, mover, draws,
                            ctypes.byref(out_tree), prev_handle)
    if r < 0:
        if out_tree.value:
            _free_tree(out_tree.value)
        return None, [], None
    stats = [(moves[k], visits[k], wins[k], mover[k], draws[k])
             for k in range(count.value)]
    mv = best.value if best.value >= 0 else None
    tree = _CTree(out_tree.value) if out_tree.value else None
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


def minimax_ultimate(state, depth, opts=None):
    """Depth-limited alpha-beta; returns the best move or None."""
    lib = _load()
    if lib is None or not _has_engine:
        return None
    c_opts = make_opts(opts)
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    best = ctypes.c_int32(-1)
    lib.sba_minimax_ultimate(ca, ma, player, active, depth,
                             ctypes.byref(c_opts), ctypes.byref(best))
    mv = best.value
    if mv < 0:
        return None
    return (mv // 9, mv % 9)


def flat_mcts(state, per_move, use_pattern=False, rng_seed=None, opts=None):
    """Root-level playouts per legal move; returns the best move or None."""
    lib = _load()
    if lib is None or not _has_engine:
        return None
    cfg = dict(DEFAULT_OPTS)
    if opts:
        cfg.update({k: v for k, v in opts.items() if k in cfg})
    if use_pattern:
        cfg['use_pattern'] = True
    c_opts = make_opts(cfg)
    cells, macro, player, active = serialize_ultimate(state)
    ca, ma = _arrays(cells, macro)
    best = ctypes.c_int32(-1)
    seed = rng_seed if rng_seed is not None else random.getrandbits(64)
    lib.sba_flat_mcts(ca, ma, player, active, per_move, seed,
                      ctypes.byref(c_opts), ctypes.byref(best))
    mv = best.value
    if mv < 0:
        return None
    return (mv // 9, mv % 9)
