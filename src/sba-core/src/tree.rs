//! The PUCT tree and the leaf-evaluation contract. Mirrors `src/alphazero.py`:
//! the same PUCT formula (Q stored from the node's side to move, parent takes the
//! negation), full expansion on first visit, per-ply backprop sign flips, root-only
//! Dirichlet noise (alpha = 10/len(moves)) and the counts + 1e-8 distribution.
//! `search_once` is one serial simulation; the network sits behind [`Evaluator`].

use rand::RngCore;
use rand_distr::{Dirichlet, Distribution};
use sba_game::{SbaGame, DRAW};

use crate::wave::SearchLimits;

/// One network consultation: policy logits (81 entries — the caller
/// applies softmax81) and the value from the side-to-move's perspective.
pub trait Evaluator {
    fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String>;

    /// Batched consultation, one row per leaf, RAW logits + value, reply order = input
    /// order. Default is the serial adapter; worker.rs overrides it to batch in one send.
    fn evaluate_batch(&mut self, planes: Vec<[f32; 324]>) -> Result<Vec<(Vec<f32>, f32)>, String> {
        planes.into_iter().map(|p| self.evaluate(p)).collect()
    }
}

/// Terminal position value from the player who would move next's
/// perspective (alphazero.terminal_value): at a game-over state that
/// player is always the loser, so a win is -1.0 and a draw is 0.0.
pub fn terminal_value(result: i32) -> f32 {
    if result == DRAW {
        0.0
    } else {
        -1.0
    }
}

/// Softmax over the 81 logits with max subtraction (numerically stable,
/// matching torch's F.softmax path in the reference).
pub(crate) fn softmax81(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Move-selection distribution: n = counts + 1e-8; temp <= 0.01 -> one-hot argmax
/// (ties to the first move, np.argmax); else n^(1/temp) normalized. 81 entries
/// indexed by move code, so it doubles as the record's policy plane.
pub fn distribution(counts: &[(i32, f32)], temp: f32) -> Vec<f32> {
    let n: Vec<f32> = counts.iter().map(|(_, v)| v + 1e-8).collect();
    let mut p = vec![0.0f32; 81];
    if temp <= 0.01 {
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in n.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        p[counts[bi].0 as usize] = 1.0;
    } else {
        for (i, &v) in n.iter().enumerate() {
            p[counts[i].0 as usize] = v.powf(1.0 / temp);
        }
        let s: f32 = p.iter().sum();
        for v in &mut p {
            *v /= s;
        }
    }
    p
}

pub(crate) struct Node {
    pub(crate) state: SbaGame,
    pub(crate) move_code: i32,
    pub(crate) prior: f32,
    pub(crate) visits: f32,
    pub(crate) wins: f32,
    /// Virtual stats for wave search: +1 / +VIRTUAL_LOSS per in-flight selection,
    /// stripped by backprop; always zero in serial mode.
    pub(crate) v_visits: f32,
    pub(crate) v_wins: f32,
    pub(crate) children: Vec<Node>,
}

impl Node {
    pub(crate) fn leaf(state: SbaGame, move_code: i32, prior: f32) -> Node {
        Node {
            state,
            move_code,
            prior,
            visits: 0.0,
            wins: 0.0,
            v_visits: 0.0,
            v_wins: 0.0,
            children: vec![],
        }
    }

    /// PUCT over real+virtual stats; unvisited children are INFINITY, ties go to the
    /// FIRST child to match the reference scan (`max_by` would pick the last).
    pub(crate) fn best_child_index(&self, c_puct: f32) -> usize {
        let sqrt_parent = (self.visits + self.v_visits).sqrt();
        let (mut best, mut best_val) = (0usize, f32::NEG_INFINITY);
        for (i, c) in self.children.iter().enumerate() {
            let visits = c.visits + c.v_visits;
            let uct = if visits == 0.0 {
                f32::INFINITY
            } else {
                -(c.wins + c.v_wins) / visits + c_puct * c.prior * sqrt_parent / (1.0 + visits)
            };
            if uct > best_val {
                best_val = uct;
                best = i;
            }
        }
        best
    }
}

/// Full expansion on first visit; root-only Dirichlet mixing (1-eps)*p + eps*eta with
/// alpha = 10/n — non-root keeps the raw prior, mirroring python's branch exactly.
pub(crate) fn expand_with_priors(
    node: &mut Node,
    probs: &[f32],
    limits: &SearchLimits,
    rng: &mut dyn RngCore,
    at_root: bool,
) -> Result<(), String> {
    let moves = node.state.legal();
    let n = moves.len();
    let eta: Vec<f64> = if at_root {
        if n == 1 {
            vec![1.0]
        } else {
            let alpha = 10.0 / n as f64;
            Dirichlet::new(&vec![alpha; n])
                .map_err(|e| e.to_string())?
                .sample(rng)
        }
    } else {
        Vec::new()
    };
    node.children = moves
        .iter()
        .enumerate()
        .map(|(j, &k)| {
            let p = probs[k as usize];
            let prior = if at_root {
                (1.0 - limits.dirichlet_eps) * p + limits.dirichlet_eps * eta[j] as f32
            } else {
                p
            };
            let (m, i) = (k / 9, k % 9);
            let mut child_state = node.state;
            child_state.apply(m, i);
            Node::leaf(child_state, k, prior)
        })
        .collect();
    Ok(())
}

/// Mirror of mcts_search's backprop: `visits += 1; wins += value` per node with the
/// sign flipped per ply to keep wins in each node's side-to-move frame. Python walks
/// leaf->root and flips after each add; this root->leaf walk pre-flips once, giving
/// the node at depth k the value * (-1)^(d-k) — same result, one pass.
pub(crate) fn backprop(root: &mut Node, path: &[usize], mut value: f32) {
    if path.len() % 2 == 1 {
        value = -value;
    }
    let mut cur: &mut Node = root;
    cur.visits += 1.0;
    cur.wins += value;
    value = -value;
    for &bi in path {
        cur = &mut cur.children[bi];
        cur.visits += 1.0;
        cur.wins += value;
        value = -value;
    }
}

/// One PUCT simulation: select down the best_child chain -> evaluate the
/// leaf -> expand in full -> backprop. Terminal leaves skip the network
/// and backprop terminal_value, like mcts_search. The Dirichlet flag is
/// the empty descent path: the evaluated node IS the root iff nothing was
/// descended through.
pub(crate) fn search_once(
    root: &mut Node,
    eval: &mut dyn Evaluator,
    limits: &SearchLimits,
    rng: &mut dyn RngCore,
) -> Result<(), String> {
    // descend read-only, recording child indices (borrow-safe per the task
    // ruling: walk the indices down, then re-walk mutably to mutate)
    let mut path: Vec<usize> = Vec::new();
    let mut probe: &Node = root;
    while !probe.children.is_empty() {
        let bi = probe.best_child_index(limits.c_puct);
        path.push(bi);
        probe = &probe.children[bi];
    }

    if probe.state.is_over() {
        backprop(root, &path, terminal_value(probe.state.result()));
        return Ok(());
    }

    // evaluate the leaf: policy logits -> softmax priors
    let planes = probe.state.planes();
    let (logits, value) = eval.evaluate(planes)?;
    let probs = softmax81(&logits);

    let mut leaf: &mut Node = root;
    for &bi in &path {
        leaf = &mut leaf.children[bi];
    }
    expand_with_priors(leaf, &probs, limits, rng, path.is_empty())?;

    backprop(root, &path, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::fixtures::{pol, Probed, Scripted};
    use crate::wave::search_position;
    use rand::SeedableRng;
    use sba_game::{O_WINS, X_WINS};
    use std::cell::RefCell;

    #[test]
    fn terminal_draw_is_zero_loss_is_minus_one() {
        // build a real terminal state: fill a full draw? simpler: trust
        // constants via a won micro->macro line
        let mut g = SbaGame::fresh();
        for (m, i) in [(0, 0), (1, 1), (2, 2)] {
            g.apply(m, i);
        }
        // not necessarily terminal — the assertion targets the mapping fn:
        assert_eq!(terminal_value(DRAW), 0.0);
        assert_eq!(terminal_value(X_WINS), -1.0);
        assert_eq!(terminal_value(O_WINS), -1.0);
    }

    #[test]
    fn low_temp_is_one_hot() {
        // the distribution helper: counts + 1e-8, temp<=0.01 -> argmax one-hot
        let d = distribution(&[(40, 5.0), (13, 3.0)], 0.0);
        let best = d
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        assert_eq!(best, 40);
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let d2 = distribution(&[(40, 5.0), (13, 3.0)], 1.0);
        assert!(d2[40] > d2[13]); // temp 1 keeps ordering, no explosion
    }

    #[test]
    fn non_root_priors_are_raw_probs() {
        // eps > 0: the ROOT expansion mixes (1-eps)*p + eps*eta, but a
        // non-root expansion must carry the RAW softmax prob (mcts_search's
        // non-root branch) — shrinking every non-root prior by (1-eps)
        // would break the exact mirror.
        let mut e = Scripted {
            replies: RefCell::new((0..8).map(|_| (pol(&[]), 0.5)).collect()),
        };
        let limits = SearchLimits {
            sims: 3,
            c_puct: 1.5,
            dirichlet_eps: 0.25,
            batch_size: 1, // search_once is the serial leaf evaluation
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let probs = softmax81(&pol(&[])); // uniform logits -> 1/81 each
        let mut root = Node::leaf(SbaGame::fresh(), -1, 0.0);
        for _ in 0..3 {
            search_once(&mut root, &mut e, &limits, &mut rng).unwrap();
        }
        // sim 1 expanded the root (with noise); sims 2-3 descend into the
        // first all-INFINITY child and expand it NON-root (k=0, then k=1).
        let first = &root.children[0];
        assert_eq!(first.children.len(), 8); // k=0 routes play to macro 0
        for c in &first.children {
            // raw prob, NOT (1-eps)*p — exact f32 equality holds post-fix
            assert_eq!(c.prior, probs[c.move_code as usize]);
        }
        // root side: the mix still normalizes to 1 and actually mixed noise
        let sum: f32 = root.children.iter().map(|c| c.prior).sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(root
            .children
            .iter()
            .any(|c| c.prior != probs[c.move_code as usize]));
    }

    /// batch_size 1 keeps the serial mode: no wave loop, one settled evaluate per sim.
    #[test]
    fn batch_size_one_keeps_serial_mode() {
        let mut probe = Probed::uniform(8);
        let limits = SearchLimits {
            sims: 3,
            c_puct: 1.5,
            dirichlet_eps: 0.0,
            batch_size: 1,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let (counts, _z) =
            search_position(SbaGame::fresh(), &mut probe, &limits, &mut rng).unwrap();
        assert!(probe.waves.borrow().is_empty());
        assert_eq!(*probe.singles.borrow(), 3);
        for (k, v) in &counts {
            let want = if *k < 2 { 1.0 } else { 0.0 };
            assert_eq!(*v, want, "child {k} visits");
        }
    }
}
