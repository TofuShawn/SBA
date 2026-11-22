//! Compatibility shim — the former 911-line search.rs was split (pure
//! move, zero behavior change) into three focused modules: `tree` (Node +
//! PUCT + backprop + expansion + softmax81 + terminal_value +
//! distribution, the Evaluator contract and the single-simulation
//! primitive), `wave` (the serial/wave search engine, SearchLimits and
//! the virtual-loss machinery) and `runner` (play_game + GameOutcome).
//! The explicit re-exports below keep every former `search::` path valid
//! for worker.rs, sba_debug.rs, sba_selfplay.rs and the bin tests.

pub use crate::runner::{GameOutcome, play_game};
pub use crate::tree::{Evaluator, distribution, terminal_value};
pub use crate::wave::{DEFAULT_BATCH_SIZE, SearchLimits, VIRTUAL_LOSS, search_position};
