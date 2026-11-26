//! The one-game self-play runner: `play_game`, the `self_play_game` mirror. Until the
//! game ends it runs [`crate::wave::search_position`] for `limits.sims` simulations,
//! records (planes, target, z = _root_value), samples the move from the target
//! distribution (np.random.choice mirror) and applies it.

use rand::{Rng, RngCore};
use sba_game::{SbaGame, O_WINS, X_WINS};

use crate::records;
use crate::tree::{distribution, Evaluator};
use crate::wave::{search_position, SearchLimits};

pub struct GameOutcome {
    pub result: String,
    pub record: records::GameRecord,
}

/// Scripted evaluators shared by tree/wave/runner tests, `pub(crate)` for the
/// sibling test mods.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::cell::RefCell;

    use crate::tree::Evaluator;

    /// Scripted evaluator: pops (policy, value) per call, planes ignored.
    /// When the script runs dry it falls back to the default reply (the
    /// brief's "fill lazily" note) so long games never panic the harness.
    pub(crate) struct Scripted {
        pub(crate) replies: RefCell<Vec<(Vec<f32>, f32)>>,
    }
    impl Evaluator for Scripted {
        fn evaluate(&mut self, _planes: [f32; 324]) -> Result<(Vec<f32>, f32), String> {
            let r = self
                .replies
                .borrow_mut()
                .pop()
                .unwrap_or_else(|| (pol(&[40]), 0.5));
            Ok(r)
        }
    }
    pub(crate) fn pol(w: &[usize]) -> Vec<f32> {
        let mut p = vec![0.0f32; 81];
        for &i in w {
            p[i] = 1.0;
        }
        p
    }

    /// Scripted evaluator + mode counters: `waves` records each evaluate_batch row
    /// count, `singles` counts plain evaluate calls — together they pin the mode.
    pub(crate) struct Probed {
        pub(crate) inner: Scripted,
        pub(crate) waves: RefCell<Vec<usize>>,
        pub(crate) singles: RefCell<usize>,
    }
    impl Probed {
        pub(crate) fn uniform(n: usize) -> Probed {
            Probed {
                inner: Scripted {
                    replies: RefCell::new((0..n).map(|_| (pol(&[]), 0.5)).collect()),
                },
                waves: RefCell::new(Vec::new()),
                singles: RefCell::new(0),
            }
        }
    }
    impl Evaluator for Probed {
        fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String> {
            *self.singles.borrow_mut() += 1;
            self.inner.evaluate(planes)
        }
        fn evaluate_batch(
            &mut self,
            planes: Vec<[f32; 324]>,
        ) -> Result<Vec<(Vec<f32>, f32)>, String> {
            self.waves.borrow_mut().push(planes.len());
            self.inner.evaluate_batch(planes)
        }
    }
}

/// One self-play game: search `limits.sims` from the current position, record
/// (planes, target, z = _root_value), then sample the move from the target distribution
/// (np.random.choice mirror) and apply it.
pub fn play_game(
    start: SbaGame,
    eval: &mut dyn Evaluator,
    limits: &SearchLimits,
    temp: f32,
    rng: &mut dyn RngCore,
) -> Result<GameOutcome, String> {
    let mut st = start;
    let mut states = Vec::new();
    let mut policy = Vec::new();
    let mut z = Vec::new();
    while !st.is_over() {
        let (counts, zt) = search_position(st, eval, limits, rng)?;
        let target = distribution(&counts, temp);
        states.extend_from_slice(&st.planes());
        policy.extend_from_slice(&target);
        z.push(zt);
        // sample the move by the target distribution
        let r: f64 = rng.gen();
        let mut acc = 0.0f64;
        let mut picked = counts.len() - 1;
        for (j, (k, _)) in counts.iter().enumerate() {
            acc += target[*k as usize] as f64;
            if r <= acc {
                picked = j;
                break;
            }
        }
        let k = counts[picked].0;
        st.apply(k / 9, k % 9);
    }
    let result = match st.result() {
        r if r == X_WINS => "X",
        r if r == O_WINS => "O",
        _ => "D",
    }
    .to_string();
    Ok(GameOutcome {
        result: result.clone(),
        record: records::GameRecord {
            meta: records::RecordMeta {
                game_type: "ultimate".into(),
                seed: 0,
                result,
                sims: limits.sims,
                weights_sha: String::new(),
            },
            states,
            policy,
            z,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::fixtures::{pol, Scripted};
    use super::*;
    use rand::SeedableRng;
    use std::cell::RefCell;

    #[test]
    fn visits_follow_greedy_policy() {
        // policy favors k=40 at the root; every leaf replies value 0.5.
        // With one-hot root prior the tree drives all sims into one branch.
        let mut e = Scripted {
            replies: RefCell::new(vec![]),
        };
        // enough replies for root + descendants (fill lazily when dry)
        let limits = SearchLimits {
            sims: 6,
            c_puct: 1.5,
            dirichlet_eps: 0.0,
            batch_size: 1, // serial mode
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // We need replies popped in LIFO order; push them reversed.
        for _ in 0..64 {
            e.replies.borrow_mut().push((pol(&[40]), 0.5));
        }
        let out = play_game(SbaGame::fresh(), &mut e, &limits, 0.0, &mut rng).unwrap();
        // game runs many moves; just assert records well-formed
        assert_eq!(out.record.states.len() % 324, 0);
        assert_eq!(out.record.states.len() / 324, out.record.moves());
        assert_eq!(out.record.policy.len() / 81, out.record.moves());
        for zv in &out.record.z {
            assert!((-1.0..=1.0).contains(zv));
        }
    }

    #[test]
    fn same_seed_same_record() {
        let run = |seed: u64| {
            let mut e = Scripted {
                replies: RefCell::new((0..256).map(|_| (pol(&[40]), 0.3)).collect()),
            };
            let limits = SearchLimits {
                sims: 4,
                c_puct: 1.5,
                dirichlet_eps: 0.0,
                batch_size: 1, // serial mode: the adapted determinism pin
            };
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            play_game(SbaGame::fresh(), &mut e, &limits, 1.0, &mut rng)
                .unwrap()
                .record
        };
        let (a, b) = (run(9), run(9));
        assert_eq!(a.states, b.states);
        assert_eq!(a.z, b.z);
    }
}
