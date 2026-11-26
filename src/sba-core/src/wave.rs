//! Search engine: the two modes behind `SearchLimits::batch_size`, dispatched by
//! [`search_position`]. `batch_size` 0/1 runs the serial loop (`tree::search_once`);
//! any value above 1 runs the leaf-parallel WAVE loop (`search_wave`) that python's default
//! BATCH_SIZE=256 uses, with virtual-loss best_child selection and one batched
//! evaluate per round. Layer-3 diagnosis: serial is bit-faithful to python
//! `batch_size=1` (KL 0.0000), and the two modes do not converge with budget.

use rand::RngCore;
use sba_game::SbaGame;
use std::collections::HashSet;

use crate::tree::{
    Evaluator, Node, backprop, expand_with_priors, search_once, softmax81, terminal_value,
};

/// python's BATCH_SIZE: the default leaf-parallel wave width — the value
/// the campaign's self-play targets are generated with.
pub const DEFAULT_BATCH_SIZE: usize = 256;

/// Virtual loss added per in-flight selection in wave mode
/// (alphazero.VIRTUAL_LOSS): a visited branch looks like a win (v_wins
/// += 2.0 makes -q very negative) so the next pick in the wave diverges
/// onto untouched priors instead of re-queueing the same leaf.
pub const VIRTUAL_LOSS: f32 = 2.0;

/// Per-game search knobs. `batch_size` mirrors python's `mcts_search(batch_size=...)`:
/// 0/1 = serial (settled stats per selection), >1 = leaf-parallel wave mode with
/// [`DEFAULT_BATCH_SIZE`] (256) as python's default.
pub struct SearchLimits {
    pub sims: u32,
    pub c_puct: f32,
    pub dirichlet_eps: f32,
    pub batch_size: usize,
}

/// Node lookup by child-index path (root -> leaf). Paths are recorded
/// during a read-only descent and re-walked afterwards — mutable and
/// immutable variants for the borrow-safe walk the serial path uses too.
fn node_at<'a>(root: &'a Node, path: &[usize]) -> &'a Node {
    path.iter().fold(root, |n, &bi| &n.children[bi])
}

fn node_at_mut<'a>(root: &'a mut Node, path: &[usize]) -> &'a mut Node {
    let mut cur = root;
    for &bi in path {
        cur = &mut cur.children[bi];
    }
    cur
}

/// Virtual loss along a selection path, root INCLUDED (python's
/// `for n in path: n.v_visits += 1; n.v_wins += VIRTUAL_LOSS`): every node
/// on the path looks visited and won, so the next pick of the same wave
/// diverges off the in-flight branch.
fn add_virtual_loss(root: &mut Node, path: &[usize]) {
    let mut cur: &mut Node = root;
    cur.v_visits += 1.0;
    cur.v_wins += VIRTUAL_LOSS;
    for &bi in path {
        cur = &mut cur.children[bi];
        cur.v_visits += 1.0;
        cur.v_wins += VIRTUAL_LOSS;
    }
}

/// Wave-mode backprop: strip the virtual stats every path node picked up at
/// selection, THEN accumulate the real stats with the per-ply sign flip — the same
/// root->leaf walk as serial. The round trip is bit-exact, so nothing leaks onward.
fn backprop_wave(root: &mut Node, path: &[usize], value: f32) {
    let mut cur: &mut Node = root;
    cur.v_visits -= 1.0;
    cur.v_wins -= VIRTUAL_LOSS;
    for &bi in path {
        cur = &mut cur.children[bi];
        cur.v_visits -= 1.0;
        cur.v_wins -= VIRTUAL_LOSS;
    }
    backprop(root, path, value);
}

/// Leaf-parallel wave search — mcts_search's batched loop. Per round: select up to
/// `batch_size` leaves under virtual loss, stop at an already-queued unexpanded leaf,
/// backprop terminals immediately, then one batched evaluate followed by real backprop
/// + full expansion in selection order. Real stats settle only between waves.
fn search_wave(
    root: &mut Node,
    eval: &mut dyn Evaluator,
    limits: &SearchLimits,
    rng: &mut dyn RngCore,
) -> Result<(), String> {
    let mut iterations: u32 = 0;
    while iterations < limits.sims {
        // ---- selection: up to batch_size leaves under virtual loss -------
        // A child-index path identifies a node here: the tree shape is frozen until
        // the batch's expansion, so path equality is id equality.
        let mut batch: Vec<Vec<usize>> = Vec::new();
        let mut queued: HashSet<Vec<usize>> = HashSet::new();
        while iterations < limits.sims && batch.len() < limits.batch_size {
            // descend read-only, recording the path and terminality
            let (path, terminal) = {
                let mut path: Vec<usize> = Vec::new();
                let mut probe: &Node = root;
                while !probe.children.is_empty() {
                    let bi = probe.best_child_index(limits.c_puct);
                    path.push(bi);
                    probe = &probe.children[bi];
                }
                let terminal = probe.state.is_over();
                (path, terminal)
            };
            // python :373-374: `if node.untried and id(node) in queued: break`
            // — `node.untried` truthy is exactly "unexpanded non-terminal"
            // (a descended node has no children; terminal states carry an
            // empty untried from birth).
            if !terminal && queued.contains(&path) {
                break; // every reachable leaf is already queued: evaluate now
            }
            queued.insert(path.clone());
            add_virtual_loss(root, &path);
            iterations += 1;
            if terminal {
                let value = terminal_value(node_at(root, &path).state.result());
                backprop_wave(root, &path, value);
            } else {
                batch.push(path);
            }
        }
        // ---- ONE batched evaluate, then real backprop + expansion each ----
        if !batch.is_empty() {
            let planes: Vec<[f32; 324]> = batch
                .iter()
                .map(|p| node_at(root, p).state.planes())
                .collect();
            let replies = eval.evaluate_batch(planes)?;
            if replies.len() != batch.len() {
                return Err(format!(
                    "evaluator returned {} rows for {} leaves (strict zip, python :409)",
                    replies.len(),
                    batch.len()
                ));
            }
            for (path, (logits, value)) in batch.into_iter().zip(replies) {
                let probs = softmax81(&logits);
                // virtual stats OFF (:350-353) before the real stats
                // accumulate (:354-359) — then expand in full (:416-439)
                backprop_wave(root, &path, value);
                let at_root = path.is_empty();
                expand_with_priors(node_at_mut(root, &path), &probs, limits, rng, at_root)?;
            }
        }
    }
    Ok(())
}

/// Single-position root search: the search part of play_game without move sampling.
/// Runs `limits.sims` simulations (serial for batch_size 0/1, wave otherwise), then
/// returns the root counts in legal order and z = visit-weighted child Q (0.0 if none).
pub fn search_position(
    start: SbaGame,
    eval: &mut dyn Evaluator,
    limits: &SearchLimits,
    rng: &mut dyn RngCore,
) -> Result<(Vec<(i32, f32)>, f32), String> {
    let mut root = Node::leaf(start, -1, 0.0);
    if limits.batch_size <= 1 {
        for _ in 0..limits.sims {
            search_once(&mut root, eval, limits, rng)?;
        }
    } else {
        search_wave(&mut root, eval, limits, rng)?;
    }
    // visit counts in legal order
    let counts: Vec<(i32, f32)> = root
        .children
        .iter()
        .map(|c| (c.move_code, c.visits))
        .collect();
    // z = _root_value: visit-weighted mean of children Q (0.0 if none)
    let (tot_v, tot) = root.children.iter().fold((0.0f32, 0.0f32), |(v, n), c| {
        if c.visits > 0.0 {
            (v + c.visits * (c.wins / c.visits), n + c.visits)
        } else {
            (v, n)
        }
    });
    Ok((counts, if tot > 0.0 { tot_v / tot } else { 0.0 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::fixtures::{pol, Probed, Scripted};
    use rand::Rng;
    use rand::SeedableRng;
    use std::cell::RefCell;

    /// Wave semantics with the scripted evaluator: budget <= batch_size collects
    /// distinct leaves — wave 1 expands the root, wave 2 gives each child visits = 1.
    #[test]
    fn wave_budget_within_batch_visits_each_leaf_once() {
        let mut probe = Probed {
            // replies pop LIFO: root 0.25, then leaves 1.0 / -1.0 / 0.5
            inner: Scripted {
                replies: RefCell::new(vec![
                    (pol(&[]), 0.5),
                    (pol(&[]), -1.0),
                    (pol(&[]), 1.0),
                    (pol(&[]), 0.25),
                ]),
            },
            waves: RefCell::new(Vec::new()),
            singles: RefCell::new(0),
        };
        let limits = SearchLimits {
            sims: 4,
            c_puct: 1.5,
            dirichlet_eps: 0.0,
            batch_size: 256,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let (counts, z) = search_position(SbaGame::fresh(), &mut probe, &limits, &mut rng).unwrap();
        // two waves: [root] then [child0, child1, child2]
        assert_eq!(*probe.waves.borrow(), vec![1, 3]);
        assert_eq!(*probe.singles.borrow(), 0); // the serial path never ran
        assert_eq!(counts.len(), 81);
        for (k, v) in &counts {
            let want = if *k < 3 { 1.0 } else { 0.0 };
            assert_eq!(*v, want, "child {k} visits");
        }
        // z = visit-weighted mean of children Q: (1.0 + -1.0 + 0.5) / 3
        assert!((z - 0.5 / 3.0).abs() < 1e-6, "z {z}");
    }

    /// Wave shape on the fresh 81-legal board (budget 200, batch 256) mirrors python:
    /// [1, 81, 118] — root expansion, one leaf per root child, then the remainder.
    #[test]
    fn wave_shape_fresh_board_200() {
        let mut probe = Probed::uniform(256); // every sim is one eval row here
        let limits = SearchLimits {
            sims: 200,
            c_puct: 1.5,
            dirichlet_eps: 0.0,
            batch_size: 256,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let (counts, _z) =
            search_position(SbaGame::fresh(), &mut probe, &limits, &mut rng).unwrap();
        assert_eq!(*probe.waves.borrow(), vec![1, 81, 118]);
        assert_eq!(*probe.singles.borrow(), 0);
        // structural python-mirror properties: all 81 children visited, small max share
        assert_eq!(counts.len(), 81);
        let total: f32 = counts.iter().map(|(_, v)| v).sum();
        assert_eq!(total, 199.0); // 200 sims minus the root-expansion wave's own visit
        let max = counts.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
        assert!(max / total < 0.1, "max share {}", max / total);
        assert!(counts.iter().all(|(_, v)| *v >= 1.0));
    }

    /// Virtual stats are stripped before the real stats accumulate, so they end at
    /// exactly zero and nothing leaks into the next wave's selection.
    #[test]
    fn virtual_stats_stripped_real_stats_exact() {
        fn assert_no_vstats(n: &Node) {
            assert_eq!(n.v_visits, 0.0, "v_visits residue at {:?}", n.move_code);
            assert_eq!(n.v_wins, 0.0, "v_wins residue at {:?}", n.move_code);
            for c in &n.children {
                assert_no_vstats(c);
            }
        }
        let mut probe = Probed {
            // replies pop LIFO: root 0.25, then four wave-2 leaves
            // 1.0 / -1.0 / 0.5 / 0.25 for children 0..3
            inner: Scripted {
                replies: RefCell::new(vec![
                    (pol(&[]), 0.25),
                    (pol(&[]), 0.5),
                    (pol(&[]), -1.0),
                    (pol(&[]), 1.0),
                    (pol(&[]), 0.25),
                ]),
            },
            waves: RefCell::new(Vec::new()),
            singles: RefCell::new(0),
        };
        let limits = SearchLimits {
            sims: 5,
            c_puct: 1.5,
            dirichlet_eps: 0.0,
            batch_size: 256,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let mut root = Node::leaf(SbaGame::fresh(), -1, 0.0);
        search_wave(&mut root, &mut probe, &limits, &mut rng).unwrap();
        assert_eq!(*probe.waves.borrow(), vec![1, 4]);
        assert_no_vstats(&root);
        assert_eq!(root.visits, 5.0);
        // root perspective: +0.25, then -1.0 +1.0 -0.5 -0.25 per leaf backprop
        assert_eq!(root.wins, ((0.25 - 1.0 + 1.0) - 0.5) - 0.25);
        for (j, c) in root.children.iter().enumerate() {
            let want = match j {
                0 => (1.0, 1.0),
                1 => (1.0, -1.0),
                2 => (1.0, 0.5),
                3 => (1.0, 0.25),
                _ => (0.0, 0.0),
            };
            assert_eq!((c.visits, c.wins), want, "child {j}");
        }
    }

    /// Terminal leaves backprop terminal_value IMMEDIATELY inside the
    /// selection phase (python :377-381) — no network row is consumed: a
    /// terminal root burns the whole budget on immediate backprops with
    /// zero waves and no children.
    #[test]
    fn wave_terminal_leaves_skip_the_network() {
        // a real terminal state via seeded random playout
        let mut g = SbaGame::fresh();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        while !g.is_over() {
            let legal = g.legal();
            let k = legal[rng.gen_range(0..legal.len())];
            g.apply(k / 9, k % 9);
        }
        let tv = terminal_value(g.result());
        let mut probe = Probed::uniform(4);
        let limits = SearchLimits {
            sims: 50,
            c_puct: 1.5,
            dirichlet_eps: 0.0,
            batch_size: 256,
        };
        let mut root = Node::leaf(g, -1, 0.0);
        search_wave(&mut root, &mut probe, &limits, &mut rng).unwrap();
        assert!(probe.waves.borrow().is_empty()); // zero batched evaluates
        assert_eq!(*probe.singles.borrow(), 0);
        assert_eq!(root.visits, 50.0); // every sim was an immediate terminal backprop
        assert_eq!(root.wins, 50.0 * tv);
        assert!(root.children.is_empty());
        // the same through the search_position entry: no counts, z = 0.0
        let mut probe2 = Probed::uniform(4);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(7);
        let (counts, z) = search_position(g, &mut probe2, &limits, &mut rng2).unwrap();
        assert!(counts.is_empty());
        assert_eq!(z, 0.0);
        assert!(probe2.waves.borrow().is_empty());
    }
}
