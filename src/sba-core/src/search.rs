//! Compatibility shim: search.rs was split (pure move, zero behavior change) into
//! `tree` (Node + PUCT + backprop + expansion + Evaluator), `wave` (search engine,
//! SearchLimits, virtual loss) and `runner` (play_game). These re-exports keep every
//! former `search::` path valid.

pub use crate::runner::{GameOutcome, play_game};
pub use crate::tree::{Evaluator, distribution, terminal_value};
pub use crate::wave::{DEFAULT_BATCH_SIZE, SearchLimits, VIRTUAL_LOSS, search_position};
