//! sba_game — the Ultimate Tic-Tac-Toe state layer in Rust (D48, Phase 1).
//!
//! Owns the game LOGISTICS only (apply / legal moves / result / clone /
//! state dump); the search algorithms stay in Python and in sba_engine.dll.
//! The state is packed bitboards porting src/game.py's BitUltimateGame,
//! with one difference forced by the languages: Python's 81-bit big-int
//! becomes per-macro `u16` occupancy arrays (`px` / `po`, nine bits each),
//! which keeps every shift inside a machine word and is otherwise the same
//! representation (macro wins as 9-bit `mx` / `mo`, side to move, and the
//! forced-routing macro; `-1` = free choice).
//!
//! Semantics mirror `UltimateGame` / `BitUltimateGame` exactly, including
//! the error order of `make_move` (game over -> routing -> occupied) and
//! the same ValueError texts on the Python side (`_ERR_TEXT` in
//! rustbridge.py). tests/test_rustgame.py fuzzes the parity.
//!
//! ABI: everything is `#[no_mangle] extern "C"` over opaque handles. The
//! crate owns the state `Box` and frees it in `sba_game_free` — no
//! caller-passed arenas (unlike sba_engine.dll, the state is a single
//! small struct, so there is no allocation on the hot path at all).
//!
//! No panics cross the boundary: every index is validated or comes from a
//! fixed 0..9 loop, and no function unwraps an Option/Result.

// This crate IS the FFI boundary: every export is called from ctypes with
// caller-owned buffers, so pointer validity is the caller's contract and
// the docs above. clippy 1.99 wants `unsafe fn` on every extern that
// dereferences one — we keep the exports safe-signatured and state the
// trust boundary here once, instead of 19 per-function attributes.
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
    px: [u16; 9],  // X marks per micro board (bits 0..8)
    po: [u16; 9],  // O marks per micro board
    mx: u16,       // macro boards won by X (bits 0..8)
    mo: u16,       // macro boards won by O
    current: u8,   // 0 = X to move, 1 = O to move
    active: i32,   // the macro the reply is routed into; -1 = free choice
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
    fn fresh() -> SbaGame {
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
}

/// The raw result: X_WINS / O_WINS / DRAW / ONGOING (mirror of result()).
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
    Box::into_raw(Box::new(SbaGame::fresh()))
}

/// Adopt a caller-built state (rustbridge.RustUltimateGame.from_game).
#[no_mangle]
pub extern "C" fn sba_game_load(st: *const SbaState) -> *mut SbaGame {
    if st.is_null() {
        return std::ptr::null_mut();
    }
    let st = unsafe { &*st };
    Box::into_raw(Box::new(SbaGame {
        px: st.px,
        po: st.po,
        mx: st.mx,
        mo: st.mo,
        current: st.current,
        active: st.active,
    }))
}

#[no_mangle]
pub extern "C" fn sba_game_clone(g: *const SbaGame) -> *mut SbaGame {
    if g.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(unsafe { *g }))
}

/// Free a handle. Only our own handles are ever passed here.
#[no_mangle]
pub extern "C" fn sba_game_free(g: *mut SbaGame) {
    if !g.is_null() {
        unsafe { drop(Box::from_raw(g)) };
    }
}

/// Mirror of UltimateGame.make_move: validate in the same order
/// (game over -> range -> routing -> occupied), then apply.
#[no_mangle]
pub extern "C" fn sba_game_apply(g: *mut SbaGame, m: i32, i: i32) -> i32 {
    if g.is_null() {
        return ERR_RANGE;
    }
    let g = unsafe { &mut *g };
    // _assert_playable: result() is not None (a full-drawn board included)
    if raw_result(g) != ONGOING {
        return ERR_OVER;
    }
    if !(0..9).contains(&m) || !(0..9).contains(&i) {
        return ERR_RANGE;
    }
    // _assert_ultimate_routing: the forced macro, when open, is mandatory
    if g.active >= 0 && g.macro_open(g.active as usize) {
        if m != g.active {
            return ERR_ROUTED;
        }
    } else if !g.macro_open(m as usize) {
        return ERR_MACRO_CLOSED;
    }
    let m = m as usize;
    let i = i as usize;
    if (g.px[m] | g.po[m]) >> i & 1 != 0 {
        return ERR_OCCUPIED;
    }
    if g.current == 0 {
        g.px[m] |= 1u16 << i;
    } else {
        g.po[m] |= 1u16 << i;
    }
    match g.micro_winner(m) {
        X_WINS => g.mx |= 1u16 << m,
        O_WINS => g.mo |= 1u16 << m,
        _ => {}
    }
    g.active = if g.macro_open(i) { i as i32 } else { -1 };
    g.current = 1 - g.current;
    OK
}

#[no_mangle]
pub extern "C" fn sba_game_result(g: *const SbaGame) -> i32 {
    if g.is_null() {
        return ONGOING;
    }
    raw_result(unsafe { &*g })
}

#[no_mangle]
pub extern "C" fn sba_game_macro_open(g: *const SbaGame, m: i32) -> i32 {
    if g.is_null() || !(0..9).contains(&m) {
        return 0;
    }
    unsafe { &*g }.macro_open(m as usize) as i32
}

/// Legal moves as packed indices (m * 9 + i) in UltimateGame's order.
/// Writes at most `cap` values and returns the TOTAL count, so a caller
/// can detect truncation (rustbridge always passes 81 — the maximum).
#[no_mangle]
pub extern "C" fn sba_game_legal(g: *const SbaGame, out: *mut i32, cap: i32) -> i32 {
    if g.is_null() || (out.is_null() && cap > 0) || cap < 0 {
        return -1;
    }
    let g = unsafe { &*g };
    if g.winner() != ONGOING {
        return 0;
    }
    let cap = cap as usize;
    let mut n: usize = 0;
    for k in 0..81usize {
        let m = k / 9;
        let i = k % 9;
        if (g.px[m] | g.po[m]) >> i & 1 != 0 {
            continue;
        }
        // closed macros (won or drawn-full) are not playable, even when
        // empty cells remain — UltimateGame.legal_moves skips them FIRST
        if !g.macro_open(m) {
            continue;
        }
        if g.active >= 0 && g.macro_open(g.active as usize) && m != g.active as usize {
            continue;
        }
        if n < cap {
            unsafe { *out.add(n) = k as i32 };
        }
        n += 1;
    }
    n as i32
}

/// Dump the full state (encode-from-bits; result precomputed).
#[no_mangle]
pub extern "C" fn sba_game_state(g: *const SbaGame, out: *mut SbaState) {
    if g.is_null() || out.is_null() {
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

/// Pack the sba_engine.dll ABI in one call (cells[81]: 0 empty / 1 X /
/// 2 O; macro[9]: 0 open / 1 X / 2 O / 3 drawn-full; player 1 X / 2 O;
/// active -1..8) — replaces the wrapper's per-cell Python property
/// materialization on the stage-1 self-play path.
#[no_mangle]
pub extern "C" fn sba_game_serialize(
    g: *const SbaGame,
    out_cells: *mut i32,
    out_macro: *mut i32,
    out_player: *mut i32,
    out_active: *mut i32,
) -> i32 {
    if g.is_null()
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

/// Fill the whole 4x9x9 encode tensor (324 floats, row-major, the layout
/// alphazero.encode defines): plane 0 = X marks, plane 1 = O marks,
/// plane 2 = 1.0 when X to move, plane 3 = the active macro block
/// (0 everywhere on free choice). One call replaces the numpy interleave.
#[no_mangle]
pub extern "C" fn sba_game_planes(g: *const SbaGame, out: *mut f32) -> i32 {
    if g.is_null() || out.is_null() {
        return -1;
    }
    let g = unsafe { &*g };
    let planes = unsafe { std::slice::from_raw_parts_mut(out, 324) };
    for m in 0..9usize {
        let (mr, mc) = (m / 3, m % 3);
        for i in 0..9usize {
            let (ir, ic) = (i / 3, i % 3);
            let idx = mr * 27 + ir * 9 + mc * 3 + ic;
            planes[idx] = if g.px[m] >> i & 1 != 0 { 1.0 } else { 0.0 };
            planes[81 + idx] = if g.po[m] >> i & 1 != 0 { 1.0 } else { 0.0 };
        }
    }
    let turn = if g.current == 0 { 1.0 } else { 0.0 };
    for p in planes[162..243].iter_mut() {
        *p = turn;
    }
    for p in planes[243..324].iter_mut() {
        *p = 0.0;
    }
    if g.active >= 0 {
        let (mr, mc) = ((g.active as usize) / 3, (g.active as usize) % 3);
        for dr in 0..3usize {
            for dc in 0..3usize {
                planes[243 + mr * 27 + dr * 9 + mc * 3 + dc] = 1.0;
            }
        }
    }
    OK
}

/// Clone + apply in one round trip (the PUCT expansion loop does
/// clone-then-apply across the boundary for every child node). Returns
/// the child handle, or null when the move is illegal. Writes the
/// child's routing macro (-1 = free choice) so the Python mirror stays
/// in lockstep without a second call.
#[no_mangle]
pub extern "C" fn sba_game_clone_apply(
    src: *const SbaGame,
    m: i32,
    i: i32,
    out_active: *mut i32,
) -> *mut SbaGame {
    if src.is_null() || out_active.is_null() {
        return std::ptr::null_mut();
    }
    let child = Box::into_raw(Box::new(unsafe { *src }));
    let rc = sba_game_apply(child, m, i);
    if rc != OK {
        unsafe { drop(Box::from_raw(child)) };
        return std::ptr::null_mut();
    }
    unsafe {
        *out_active = (*child).active;
    }
    child
}
