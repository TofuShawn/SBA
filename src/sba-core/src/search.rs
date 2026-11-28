//! Compatibility shim: search.rs was split (pure move, zero behavior change) into
//! `tree` (Node + PUCT + backprop + expansion + Evaluator), `wave` (search engine,
//! SearchLimits, virtual loss) and `runner` (play_game). These re-exports keep every
//! former `search::` path valid.

pub use crate::runner::{play_game, GameOutcome};
pub use crate::tree::{distribution, terminal_value, Evaluator};
pub use crate::wave::{search_position, SearchLimits, DEFAULT_BATCH_SIZE, VIRTUAL_LOSS};
