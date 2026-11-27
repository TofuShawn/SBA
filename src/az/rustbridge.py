# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""rustbridge — the ctypes boundary to the sba_game.dll cdylib (Rust
Phase 1: the game LOGISTICS layer).

RustUltimateGame IS the ultimate game implementation (removal project:
the pure-Python UltimateGame retired) — it duck-types the interface
(legal_moves / make_move / result / clone / current / active_macro /
micro / macro) on top of the Rust bitboard state, so self-play, encode()
and the engines need no changes. The cdylib is a hard prerequisite:
`available()` gates nothing anymore, __init__ raises without it. The
parity fuzz in tests/test_rustgame.py pinned the two implementations
identical while both lived; the frozen encode goldens now carry that
contract.
"""

import atexit
import contextlib
import ctypes
import os
import threading

from core.game import EMPTY, LINES, O, X

ABI = 1

_ERR_OVER = -1
_ERR_RANGE = -2
_ERR_ROUTED = -3
_ERR_MACRO_CLOSED = -4
_ERR_OCCUPIED = -5
# not part of the make_move order: the handle was already freed (caller bug)
_ERR_CLOSED = -6

# sba_game_apply error code -> the same text UltimateGame would raise
_ERR_TEXT = {
    _ERR_OVER: "game is over",
    _ERR_RANGE: "cell out of range",
    _ERR_ROUTED: "move must be in the routed macro",
    _ERR_MACRO_CLOSED: "macro is not open",
    _ERR_OCCUPIED: "cell is occupied",
    _ERR_CLOSED: "handle is not live (already freed?)",
}

_RES_CODES = {0: None, 1: X, 2: O, 3: "D"}

# this file lives in src/az/, so three dirnames up to the repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Probe order: build.py's copy in src/az/ first, then the root target/ tree.
# Per-OS candidates have different lengths, so annotate the union shape for mypy.
_LIB_PATHS: tuple[str, ...]
if os.name == "nt":
    _LIB_PATHS = (
        os.path.join(_ROOT, "src", "az", "sba_game.dll"),
        os.path.join(_ROOT, "target", "x86_64-pc-windows-gnu", "release", "sba_game.dll"),
        os.path.join(_ROOT, "target", "release", "sba_game.dll"),
    )
else:  # Linux CI: the cdylib lives in target/release, no build.py copy
    _LIB_PATHS = (os.path.join(_ROOT, "src", "az", "sba_game.so"),)
_LIB_PATH = _LIB_PATHS[0]

_LIB = None
_TRIED = False

# deferred handle frees: the AZ search tree churns thousands of
# wrapper states per game; freeing each in __del__ costs one ctypes call
# per object, which profiled slower than the Python game's refcount
# deallocation. Handles accumulate here and release in one batch.
_DEAD_HANDLES: list[int] = []
_FREE_BATCH = 512
# guards the swap below. Two threads flushing at once used to compute the
# same `freed` count and then free whatever occupied those slots after the
# other thread had already truncated the list — a live handle could be
# freed twice (Rust Box::from_raw on the same pointer = heap corruption).
_FREE_LOCK = threading.Lock()


def _flush_handles():
    """Release queued handles. Registered with atexit so the last batch
    (fewer than _FREE_BATCH) is freed on a clean interpreter exit instead
    of waiting for the OS to reclaim it.

    The queue is swapped out under a lock and freed outside it: each thread
    gets its own list, so a concurrent append can never be mistaken for a
    handle that was already released.
    """
    global _DEAD_HANDLES
    lib = _LIB
    if lib is None:
        return
    with _FREE_LOCK:
        handles, _DEAD_HANDLES = _DEAD_HANDLES, []
    for handle in handles:
        # the DLL may already be gone at teardown
        with contextlib.suppress(Exception):
            lib.sba_game_free(handle)


atexit.register(_flush_handles)


class SbaState(ctypes.Structure):
    """Mirror of the Rust SbaState dump — field order IS the ABI
    (rust cdylib src/sba-game/src/lib.rs #[repr(C)] SbaState)."""

    _fields_ = [
        ("px", ctypes.c_uint16 * 9),
        ("po", ctypes.c_uint16 * 9),
        ("mx", ctypes.c_uint16),
        ("mo", ctypes.c_uint16),
        ("current", ctypes.c_uint8),
        ("active", ctypes.c_int32),
        ("result", ctypes.c_int32),
    ]


def _load():
    global _LIB, _TRIED, _LIB_PATH
    if _TRIED:
        return _LIB
    _TRIED = True
    for path in _LIB_PATHS:
        if os.path.isfile(path):
            _LIB_PATH = path
            break
    else:
        return None
    try:
        lib = ctypes.CDLL(_LIB_PATH)
        lib.sba_game_abi.argtypes = []
        lib.sba_game_abi.restype = ctypes.c_int32
        lib.sba_game_new.argtypes = []
        lib.sba_game_new.restype = ctypes.c_void_p
        lib.sba_game_load.argtypes = [ctypes.POINTER(SbaState)]
        lib.sba_game_load.restype = ctypes.c_void_p
        lib.sba_game_clone.argtypes = [ctypes.c_void_p]
        lib.sba_game_clone.restype = ctypes.c_void_p
        lib.sba_game_free.argtypes = [ctypes.c_void_p]
        lib.sba_game_free.restype = None
        lib.sba_game_apply.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        lib.sba_game_apply.restype = ctypes.c_int32
        lib.sba_game_result.argtypes = [ctypes.c_void_p]
        lib.sba_game_result.restype = ctypes.c_int32
        lib.sba_game_macro_open.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.sba_game_macro_open.restype = ctypes.c_int32
        lib.sba_game_legal.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.sba_game_legal.restype = ctypes.c_int32
        lib.sba_game_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(SbaState)]
        lib.sba_game_state.restype = None
        lib.sba_game_serialize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.sba_game_serialize.restype = ctypes.c_int32
        lib.sba_game_planes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.sba_game_planes.restype = ctypes.c_int32
        lib.sba_game_clone_apply.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.sba_game_clone_apply.restype = ctypes.c_void_p
        if lib.sba_game_abi() != ABI:
            return None
        _LIB = lib
    except OSError:
        _LIB = None
    return _LIB


def available():
    """True when the Rust state layer is built and ABI-compatible."""
    return _load() is not None


class RustUltimateGame:
    """UltimateGame duck-type over the Rust bitboard state.

    The handle is the authoritative state; `current` / `active_macro`
    are Python mirrors updated in make_move so the search loops read
    plain attributes exactly like UltimateGame (no ctypes call per read).
    """

    __slots__ = ("_h", "current", "active_macro")

    def __init__(self):
        lib = _load()
        if lib is None:
            raise RuntimeError(
                "sba_game native library not available (build it with `python dev.py build`)"
            )
        self._h = lib.sba_game_new()
        self.current = X
        self.active_macro = None

    def __del__(self):
        # batch the frees: a PUCT expansion creates ~thousands of wrapper
        # states per game and a per-object FFI free measured 30% slower than
        # the Python game in the stage-2 mix (the AZ tree churns states).
        # Handles park in _DEAD_HANDLES until the threshold flush.
        h = getattr(self, "_h", None)
        if not h:
            return
        queue = globals().get("_DEAD_HANDLES")
        if queue is None:  # interpreter teardown already dropped the module
            return
        queue.append(h)
        if len(queue) >= _FREE_BATCH:
            _flush_handles()

    @classmethod
    def from_sba_state(cls, st):
        """Adopt an SbaState bit position (test seeding / state transfer —
        px/po are per-micro 9-bit X/O masks, mx/mo the macro badges,
        current 0=X/1=O, active -1=free routing)."""
        lib = _load()
        if lib is None:
            raise RuntimeError("sba_game native library not available")
        g = cls.__new__(cls)
        g._h = lib.sba_game_load(ctypes.byref(st))
        if not g._h:
            raise RuntimeError("sba_game_load failed")
        g.current = X if st.current == 0 else O
        g.active_macro = None if st.active < 0 else st.active
        return g

    # ------------------------------------------------- hot-path interface

    def clone(self):
        g = type(self).__new__(type(self))
        g._h = _LIB.sba_game_clone(self._h)  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
        g.current = self.current
        g.active_macro = self.active_macro
        return g

    def make_move(self, m, i):
        rc = _LIB.sba_game_apply(self._h, m, i)  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
        if rc != 0:
            raise ValueError(_ERR_TEXT.get(rc, "invalid move"))
        # mirrors advance in lockstep with the handle (same rules)
        self.current = O if self.current == X else X
        self.active_macro = i if self.macro_open(i) else None

    def legal_moves(self):
        buf = (ctypes.c_int32 * 81)()
        n = _LIB.sba_game_legal(self._h, buf, 81)  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
        return [(v // 9, v % 9) for v in buf[:n]]

    # ------------------------------------------------- read-only surface

    def _state(self):
        st = SbaState()
        _LIB.sba_game_state(self._h, ctypes.byref(st))  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
        return st

    def result(self):
        return _RES_CODES.get(_LIB.sba_game_result(self._h))  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded

    def winner(self):
        r = _RES_CODES.get(_LIB.sba_game_result(self._h))  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
        return r if r in (X, O) else None

    def is_over(self):
        return self.result() is not None

    def is_full(self):
        return all(not self.macro_open(m) for m in range(9))

    def macro_open(self, m):
        return bool(_LIB.sba_game_macro_open(self._h, m))  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded

    def micro_winner(self, m):
        st = self._state()
        for a, b, c in LINES:
            line = (1 << a) | (1 << b) | (1 << c)
            if st.px[m] & line == line:
                return X
            if st.po[m] & line == line:
                return O
        return None

    @property
    def micro(self):
        st = self._state()
        out = []
        for m in range(9):
            row = []
            for i in range(9):
                if (st.px[m] >> i) & 1:
                    row.append(X)
                elif (st.po[m] >> i) & 1:
                    row.append(O)
                else:
                    row.append(EMPTY)
            out.append(row)
        return out

    @property
    def macro(self):
        st = self._state()
        out = []
        for m in range(9):
            if (st.mx >> m) & 1:
                out.append(X)
            elif (st.mo >> m) & 1:
                out.append(O)
            elif (st.px[m] | st.po[m]) == 0x1FF:  # parens: | binds looser than ==
                out.append("D")
            else:
                out.append(EMPTY)
        return out

    # ------------------------------------------------- fast bulk exports

    def sba_ffi(self):
        """(cells, macro, player, active) as FFI-ready ctypes arrays —
        cengine's serialize fast path: one Rust call replaces the per-cell
        Python property materialization on the stage-1 self-play path."""
        cells = (ctypes.c_int32 * 81)()
        macro = (ctypes.c_int32 * 9)()
        player = ctypes.c_int32()
        active = ctypes.c_int32()
        _LIB.sba_game_serialize(  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
            self._h, cells, macro, ctypes.byref(player), ctypes.byref(active)
        )
        return cells, macro, player.value, active.value

    def sba_planes_tensor(self):
        """The whole 4x9x9 encode tensor as float32 numpy — one Rust call
        fills all four planes (alphazero.encode adopts it wholesale)."""
        import numpy as np

        buf = np.zeros(324, dtype=np.float32)
        rc = _LIB.sba_game_planes(  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
            self._h, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if rc != 0:
            raise RuntimeError("sba_game_planes failed")
        return buf.reshape(4, 9, 9)

    def clone_apply(self, m, i):
        """clone + make_move in one FFI round trip — the PUCT expansion
        loop's hot path. The mirror fields advance in lockstep (the Rust
        side returns the child's routing macro)."""
        child = type(self).__new__(type(self))
        active = ctypes.c_int32()
        child._h = _LIB.sba_game_clone_apply(  # type: ignore[union-attr]  # _LIB is live: instances exist only after __init__/_load() succeeded
            self._h, m, i, ctypes.byref(active)
        )
        if not child._h:
            raise ValueError("clone_apply: illegal move")
        child.current = O if self.current == X else X
        child.active_macro = active.value if active.value >= 0 else None
        return child
