//! sba_game — the Ultimate Tic-Tac-Toe state layer in Rust (Phase 1). Owns the
//! game LOGISTICS only (apply / legal / result / clone / state dump); search stays in
//! Python and sba_engine.dll. Packed bitboards port BitUltimateGame, with per-macro
//! `u16` occupancy arrays (px/po) instead of Python's 81-bit int so shifts stay in a
//! machine word. Semantics mirror UltimateGame exactly, including make_move's error
//! order (game over -> routing -> occupied) and the ValueError texts.
//! ABI: `#[no_mangle] extern "C"` over opaque handles; the crate owns the state Box
//! and frees it in `sba_game_free`. No panics cross the boundary — every index is
//! validated or comes from a fixed 0..9 loop, and nothing unwraps an Option/Result.

// FFI boundary: exports take caller-owned buffers (buffer validity is the caller's
// contract) but in-crate HANDLE liveness is enforced by the LIVE registry below, so a
// freed or foreign handle returns ERR_CLOSED instead of reading freed memory.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub const ABI: i32 = 1;

// result() codes
pub const ONGOING: i32 = 0;
pub const X_WINS: i32 = 1;
pub const O_WINS: i32 = 2;
pub const DRAW: i32 = 3;

// sba_game_apply error codes (mirror game.py's ValueError order)
pub const OK: i32 = 0;
pub const ERR_OVER: i32 = -1;
pub const ERR_RANGE: i32 = -2;
pub const ERR_ROUTED: i32 = -3;
pub const ERR_MACRO_CLOSED: i32 = -4;
pub const ERR_OCCUPIED: i32 = -5;
// not part of the make_move order: the handle was NULL or is no longer in
// the live registry (freed, or never handed out) — a caller bug that must
// not masquerade as a game error like "cell out of range"
pub const ERR_CLOSED: i32 = -6;

use std::collections::HashSet;
use std::sync::Mutex;

/// Handles handed out and not yet freed. A dangling pointer cannot be told from a live
/// one by value, so every export checks membership: use-after-free becomes ERR_CLOSED
/// and a double free is a no-op. (Option because HashSet::new is not const.)
static LIVE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn with_live<R>(f: impl FnOnce(&mut HashSet<usize>) -> R) -> R {
    // a poisoned guard means a previous panic held the lock; the set itself
    // is intact, so recover instead of panicking across the FFI boundary
    let mut slot = LIVE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let set = slot.get_or_insert_with(HashSet::new);
    f(set)
}

fn handle_live(g: *const SbaGame) -> bool {
    with_live(|set| set.contains(&(g as usize)))
}

fn handle_track(g: *const SbaGame) {
    with_live(|set| set.insert(g as usize));
}

fn handle_untrack(g: *const SbaGame) -> bool {
    with_live(|set| set.remove(&(g as usize)))
}

const FULL9: u16 = 0x1FF; // a full micro board (nine cells)

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

/// The authoritative state (an opaque handle from the Python side).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SbaGame {
    px: [u16; 9], // X marks per micro board (bits 0..8)
    po: [u16; 9], // O marks per micro board
    mx: u16,      // macro boards won by X (bits 0..8)
    mo: u16,      // macro boards won by O
    current: u8,  // 0 = X to move, 1 = O to move
    active: i32,  // the macro the reply is routed into; -1 = free choice
}

/// A full state dump for encode-from-bits and the Python parity fuzz.
/// Field order IS the ABI — rustbridge.SbaState mirrors it exactly.
#[repr(C)]
pub struct SbaState {
    pub px: [u16; 9],
    pub po: [u16; 9],
    pub mx: u16,
    pub mo: u16,
    pub current: u8,
    pub active: i32,
    pub result: i32,
}

impl SbaGame {
    /// A fresh game: empty board, X to move, free choice.
    pub fn fresh() -> SbaGame {
        SbaGame {
            px: [0; 9],
            po: [0; 9],
            mx: 0,
            mo: 0,
            current: 0,
            active: -1,
        }
    }

    fn micro_winner(&self, m: usize) -> i32 {
        let px = self.px[m];
        let po = self.po[m];
        for &[a, b, c] in &LINES {
            let line = (1u16 << a) | (1u16 << b) | (1u16 << c);
            if px & line == line {
                return X_WINS;
            }
            if po & line == line {
                return O_WINS;
            }
        }
        ONGOING
    }

    fn macro_open(&self, m: usize) -> bool {
        if (self.mx | self.mo) >> m & 1 != 0 {
            return false;
        }
        self.px[m] | self.po[m] != FULL9
    }

    fn winner(&self) -> i32 {
        for &[a, b, c] in &LINES {
            let line = (1u16 << a) | (1u16 << b) | (1u16 << c);
            if self.mx & line == line {
                return X_WINS;
            }
            if self.mo & line == line {
                return O_WINS;
            }
        }
        ONGOING
    }

    fn is_full(&self) -> bool {
        for m in 0..9usize {
            if self.macro_open(m) {
                return false;
            }
        }
        true
    }

    /// Applies move (m, i); return-code semantics identical to sba_game_apply.
    pub fn apply(&mut self, m: i32, i: i32) -> i32 {
        // _assert_playable: result() is not None (a full-drawn board included)
        if raw_result(self) != ONGOING {
            return ERR_OVER;
        }
        if !(0..9).contains(&m) || !(0..9).contains(&i) {
            return ERR_RANGE;
        }
        // _assert_ultimate_routing: the forced macro, when open, is mandatory
        if self.active >= 0 && self.macro_open(self.active as usize) {
            if m != self.active {
                return ERR_ROUTED;
            }
        } else if !self.macro_open(m as usize) {
            return ERR_MACRO_CLOSED;
        }
        let m = m as usize;
        let i = i as usize;
        if (self.px[m] | self.po[m]) >> i & 1 != 0 {
            return ERR_OCCUPIED;
        }
        if self.current == 0 {
            self.px[m] |= 1u16 << i;
        } else {
            self.po[m] |= 1u16 << i;
        }
        match self.micro_winner(m) {
            X_WINS => self.mx |= 1u16 << m,
            O_WINS => self.mo |= 1u16 << m,
            _ => {}
        }
        self.active = if self.macro_open(i) { i as i32 } else { -1 };
        self.current = 1 - self.current;
        OK
    }

    /// Legal move codes k = m*9+i; same filter order as sba_game_legal:
    /// occupied cell → closed macro → active-macro routing.
    pub fn legal(&self) -> Vec<i32> {
        if self.winner() != ONGOING {
            return Vec::new();
        }
        let mut legal = Vec::new();
        for k in 0..81usize {
            let m = k / 9;
            let i = k % 9;
            if (self.px[m] | self.po[m]) >> i & 1 != 0 {
                continue;
            }
            // closed macros (won or drawn-full) are not playable, even when
            // empty cells remain — UltimateGame.legal_moves skips them FIRST
            if !self.macro_open(m) {
                continue;
            }
            if self.active >= 0
                && self.macro_open(self.active as usize)
                && m != self.active as usize
            {
                continue;
            }
            legal.push(k as i32);
        }
        legal
    }

    /// The raw result: X_WINS / O_WINS / DRAW / ONGOING.
    pub fn result(&self) -> i32 {
        raw_result(self)
    }

    pub fn is_over(&self) -> bool {
        self.result() != ONGOING
    }

    /// 0 = X to move, 1 = O to move.
    pub fn current(&self) -> u8 {
        self.current
    }

    /// The macro the reply is routed into; -1 = free choice.
    pub fn active(&self) -> i32 {
        self.active
    }

    /// The 4x9x9 encode tensor (324 floats, row-major, the layout
    /// alphazero.encode defines): plane 0 = X marks, plane 1 = O marks,
    /// plane 2 = 1.0 when X to move, plane 3 = the active macro block
    /// (0 everywhere on free choice).
    pub fn planes(&self) -> [f32; 324] {
        let mut planes = [0.0f32; 324];
        for m in 0..9usize {
            let (mr, mc) = (m / 3, m % 3);
            for i in 0..9usize {
                let (ir, ic) = (i / 3, i % 3);
                let idx = mr * 27 + ir * 9 + mc * 3 + ic;
                planes[idx] = if self.px[m] >> i & 1 != 0 { 1.0 } else { 0.0 };
                planes[81 + idx] = if self.po[m] >> i & 1 != 0 { 1.0 } else { 0.0 };
            }
        }
        let turn = if self.current == 0 { 1.0 } else { 0.0 };
        for p in planes[162..243].iter_mut() {
            *p = turn;
        }
        for p in planes[243..324].iter_mut() {
            *p = 0.0;
        }
        if self.active >= 0 {
            let (mr, mc) = ((self.active as usize) / 3, (self.active as usize) % 3);
            for dr in 0..3usize {
                for dc in 0..3usize {
                    planes[243 + mr * 27 + dr * 9 + mc * 3 + dc] = 1.0;
                }
            }
        }
        planes
    }
}

/// The canonical raw result: X_WINS / O_WINS / DRAW / ONGOING; the public
/// `result()` delegates here.
fn raw_result(g: &SbaGame) -> i32 {
    let w = g.winner();
    if w != ONGOING {
        return w;
    }
    if g.is_full() {
        return DRAW;
    }
    ONGOING
}

#[no_mangle]
pub extern "C" fn sba_game_abi() -> i32 {
    ABI
}

#[no_mangle]
pub extern "C" fn sba_game_new() -> *mut SbaGame {
    let g = Box::into_raw(Box::new(SbaGame::fresh()));
    handle_track(g);
    g
}

/// Adopt a caller-built state (rustbridge.RustUltimateGame.from_game).
#[no_mangle]
pub extern "C" fn sba_game_load(st: *const SbaState) -> *mut SbaGame {
    if st.is_null() {
        return std::ptr::null_mut();
    }
    let st = unsafe { &*st };
    let g = Box::into_raw(Box::new(SbaGame {
        px: st.px,
        po: st.po,
        mx: st.mx,
        mo: st.mo,
        current: st.current,
        active: st.active,
    }));
    handle_track(g);
    g
}

#[no_mangle]
pub extern "C" fn sba_game_clone(g: *const SbaGame) -> *mut SbaGame {
    if !handle_live(g) {
        return std::ptr::null_mut();
    }
    let clone = Box::into_raw(Box::new(unsafe { *g }));
    handle_track(clone);
    clone
}

/// Free a handle. A second free of the same address is a no-op (the
/// registry already dropped it), which turns the classic double-free into
/// nothing at all.
#[no_mangle]
pub extern "C" fn sba_game_free(g: *mut SbaGame) {
    if handle_untrack(g) {
        unsafe { drop(Box::from_raw(g)) };
    }
}

/// Mirror of UltimateGame.make_move: validate in the same order
/// (game over -> range -> routing -> occupied), then apply.
#[no_mangle]
pub extern "C" fn sba_game_apply(g: *mut SbaGame, m: i32, i: i32) -> i32 {
    if !handle_live(g) {
        return ERR_CLOSED;
    }
    unsafe { (*g).apply(m, i) }
}

#[no_mangle]
pub extern "C" fn sba_game_result(g: *const SbaGame) -> i32 {
    if !handle_live(g) {
        return ERR_CLOSED;
    }
    unsafe { (*g).result() }
}

#[no_mangle]
pub extern "C" fn sba_game_macro_open(g: *const SbaGame, m: i32) -> i32 {
    // a dead handle reports "not open" (0) — the boolean contract has no
    // error channel, and no caller acts on a dead handle's answer anyway
    if !handle_live(g) || !(0..9).contains(&m) {
        return 0;
    }
    unsafe { &*g }.macro_open(m as usize) as i32
}

/// Legal moves as packed indices (m * 9 + i) in UltimateGame's order.
/// Writes at most `cap` values and returns the TOTAL count, so a caller
/// can detect truncation (rustbridge always passes 81 — the maximum).
#[no_mangle]
pub extern "C" fn sba_game_legal(g: *const SbaGame, out: *mut i32, cap: i32) -> i32 {
    if !handle_live(g) || (out.is_null() && cap > 0) || cap < 0 {
        return -1;
    }
    let moves = unsafe { (*g).legal() };
    let cap = cap as usize;
    for (n, &k) in moves.iter().enumerate().take(cap) {
        unsafe { *out.add(n) = k };
    }
    moves.len() as i32
}

/// Dump the full state (encode-from-bits; result precomputed).
#[no_mangle]
pub extern "C" fn sba_game_state(g: *const SbaGame, out: *mut SbaState) {
    if !handle_live(g) || out.is_null() {
        return;
    }
    let g = unsafe { &*g };
    let st = unsafe { &mut *out };
    st.px = g.px;
    st.po = g.po;
    st.mx = g.mx;
    st.mo = g.mo;
    st.current = g.current;
    st.active = g.active;
    st.result = raw_result(g);
}

/// Pack the sba_engine.dll ABI in one call (cells[81] 0/X/O, macro[9] open/X/O/drawn,
/// player 1 X / 2 O, active -1..8), replacing per-cell Python property materialization.
#[no_mangle]
pub extern "C" fn sba_game_serialize(
    g: *const SbaGame,
    out_cells: *mut i32,
    out_macro: *mut i32,
    out_player: *mut i32,
    out_active: *mut i32,
) -> i32 {
    if !handle_live(g)
        || out_cells.is_null()
        || out_macro.is_null()
        || out_player.is_null()
        || out_active.is_null()
    {
        return -1;
    }
    let g = unsafe { &*g };
    for k in 0..81usize {
        let (m, i) = (k / 9, k % 9);
        let v = if g.px[m] >> i & 1 != 0 {
            1
        } else if g.po[m] >> i & 1 != 0 {
            2
        } else {
            0
        };
        unsafe { *out_cells.add(k) = v };
    }
    for m in 0..9usize {
        let v = if g.mx >> m & 1 != 0 {
            1
        } else if g.mo >> m & 1 != 0 {
            2
        } else if g.px[m] | g.po[m] == FULL9 {
            3
        } else {
            0
        };
        unsafe { *out_macro.add(m) = v };
    }
    unsafe {
        *out_player = if g.current == 0 { 1 } else { 2 };
        *out_active = g.active;
    }
    OK
}

/// FFI wrapper: fills the caller-owned 324-float buffer (the layout
/// alphazero.encode defines — see SbaGame::planes). One call replaces
/// the numpy interleave.
#[no_mangle]
pub extern "C" fn sba_game_planes(g: *const SbaGame, out: *mut f32) -> i32 {
    if !handle_live(g) || out.is_null() {
        return ERR_CLOSED;
    }
    let planes = unsafe { (*g).planes() };
    unsafe { std::slice::from_raw_parts_mut(out, 324).copy_from_slice(&planes) };
    OK
}

/// Clone + apply in one round trip for the PUCT expansion loop. Returns the child
/// handle (null if the move is illegal) and writes the routing macro (-1 = free).
#[no_mangle]
pub extern "C" fn sba_game_clone_apply(
    src: *const SbaGame,
    m: i32,
    i: i32,
    out_active: *mut i32,
) -> *mut SbaGame {
    if !handle_live(src) || out_active.is_null() {
        return std::ptr::null_mut();
    }
    let child = Box::into_raw(Box::new(unsafe { *src }));
    // the child must be tracked BEFORE apply — the registry check inside
    // sba_game_apply would otherwise reject our own fresh handle
    handle_track(child);
    let rc = sba_game_apply(child, m, i);
    if rc != OK {
        sba_game_free(child); // untrack + drop in one step
        return std::ptr::null_mut();
    }
    unsafe {
        *out_active = (*child).active;
    }
    child
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_and_legal_count() {
        let g = SbaGame::fresh();
        assert_eq!(g.current(), 0);
        assert_eq!(g.active(), -1);
        assert_eq!(g.legal().len(), 81); // free choice, empty board
        assert!(!g.is_over());
    }

    #[test]
    fn apply_alternates_and_routes() {
        let mut g = SbaGame::fresh();
        assert_eq!(g.apply(4, 4), 0);
        assert_eq!(g.current(), 1);
        assert_eq!(g.active(), 4); // reply routed into macro 4
        assert!(!g.legal().iter().any(|&k| k / 9 != 4));
    }

    #[test]
    fn planes_layout_matches_docstring() {
        let mut g = SbaGame::fresh();
        g.apply(0, 0);
        let p = g.planes();
        assert_eq!(p.len(), 324);
        assert_eq!(p[0], 1.0); // plane 0 cell (0,0): X mark
        assert_eq!(p[2], 0.0); // plane 2: O to move
        let active = &p[3 * 81..4 * 81];
        assert_eq!(active[0], 1.0); // plane 3 block: macro 0 top-left 3x3
        assert_eq!(active[2], 1.0);
        assert_eq!(active[18], 1.0);
        assert_eq!(active[4], 0.0); // outside the macro-0 block
    }

    #[test]
    fn same_seed_runs_identical() {
        // two identical move sequences produce identical planes
        let mut a = SbaGame::fresh();
        let mut b = SbaGame::fresh();
        for k in [40, 0, 41] {
            a.apply(k / 9, k % 9);
            b.apply(k / 9, k % 9);
        }
        assert_eq!(a.planes(), b.planes());
    }
}
