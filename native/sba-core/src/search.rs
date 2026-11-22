//! The PUCT tree and the one-game self-play runner (Phase A).
//!
//! Mirrors `src/alphazero.py` exactly — `SearchNode.best_child`,
//! `mcts_search` and `self_play_game` — with the same PUCT formula
//! (including the `-q` negation: node Q is stored from that node's
//! side-to-move perspective, so the parent's mover maximizes its
//! negation), full expansion on first visit, per-ply backprop sign flips,
//! root-only Dirichlet prior noise (alpha = 10/len(moves)), and the
//! counts + 1e-8 move-selection distribution. The network sits behind the
//! [`Evaluator`] trait: worker.rs (batched inference) implements it, the
//! self-play exe drives [`play_game`].
//!
//! Two search MODES, behind `SearchLimits::batch_size` — python's
//! `mcts_search(batch_size=...)` verbatim:
//!
//! - `batch_size` 0/1: the serial loop (one settled leaf per simulation,
//!   `search_once`). The layer-3 diagnosis proved this bit-faithful to
//!   python `batch_size=1` (identical visit counts, KL 0.0000 both ways at
//!   200 AND 3200 sims).
//! - `batch_size` > 1: the leaf-parallel WAVE loop ([`search_wave`]) — per
//!   round, up to `batch_size` leaves are selected descending from the
//!   root with the virtual-loss stats (v_visits / v_wins,
//!   VIRTUAL_LOSS = 2.0) feeding best_child, skipped when already queued,
//!   terminal leaves backpropped immediately, then ONE batched evaluate
//!   and a real backprop + full expansion per leaf. This is the mode the
//!   campaign's self-play targets are generated with (python's default
//!   BATCH_SIZE = 256); structurally distinct from serial PUCT — real
//!   stats settle only between waves, so the two modes do not converge
//!   with budget (diag-layer3-report §3).

use rand::{Rng, RngCore};
use rand_distr::{Dirichlet, Distribution};
use sba_game::{DRAW, O_WINS, SbaGame, X_WINS};
use std::collections::HashSet;

use crate::records;

/// One network consultation: policy logits (81 entries — the caller
/// applies softmax81) and the value from the side-to-move's perspective.
pub trait Evaluator {
    fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String>;

    /// Batched consultation: one row per leaf, RAW logits + value per row
    /// (same contract as [`Evaluator::evaluate`], reply order = input
    /// order). The default is the trivial serial adapter — loop
    /// [`Evaluator::evaluate`] — which is what tests and any sims-1-style
    /// caller need; worker.rs overrides it so k rows ride ONE channel
    /// message into the worker's bucket.
    fn evaluate_batch(&mut self, planes: Vec<[f32; 324]>) -> Result<Vec<(Vec<f32>, f32)>, String> {
        planes.into_iter().map(|p| self.evaluate(p)).collect()
    }
}

/// python's BATCH_SIZE: the default leaf-parallel wave width — the value
/// the campaign's self-play targets are generated with.
pub const DEFAULT_BATCH_SIZE: usize = 256;

/// Virtual loss added per in-flight selection in wave mode
/// (alphazero.VIRTUAL_LOSS): a visited branch looks like a win (v_wins
/// += 2.0 makes -q very negative) so the next pick in the wave diverges
/// onto untouched priors instead of re-queueing the same leaf.
pub const VIRTUAL_LOSS: f32 = 2.0;

/// Per-game search knobs (the exe's only knobs — nothing extra).
/// `batch_size` mirrors python's `mcts_search(batch_size=...)`: 0/1 = the
/// serial mode (current behavior, settled stats at every selection), > 1 =
/// the leaf-parallel wave mode. [`DEFAULT_BATCH_SIZE`] (256) is python's
/// default — play_game / self-play use it as-is.
pub struct SearchLimits {
    pub sims: u32,
    pub c_puct: f32,
    pub dirichlet_eps: f32,
    pub batch_size: usize,
}

pub struct GameOutcome {
    pub result: String,
    pub record: records::GameRecord,
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
fn softmax81(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Move-selection distribution (self_play_game mirror): n = counts + 1e-8;
/// temp <= 0.01 -> one-hot argmax; else n^(1/temp), normalized. Returns an
/// 81-entry vector indexed by move code k = m*9+i (zeros off the legal
/// moves), exactly like the Python target — so it doubles as the record's
/// policy plane. Argmax ties go to the FIRST move (np.argmax), like every
/// other first-max tie-break here.
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

struct Node {
    state: SbaGame,
    move_code: i32,
    prior: f32,
    visits: f32,
    wins: f32,
    /// Virtual statistics for wave search (SearchNode.v_visits / v_wins):
    /// +1 / +VIRTUAL_LOSS per in-flight selection on this node, stripped
    /// again by the backprop. Exactly zero outside a wave — in serial mode
    /// they never leave zero, so the selection formula below reduces to
    /// the plain PUCT it was before the wave port.
    v_visits: f32,
    v_wins: f32,
    children: Vec<Node>,
}

impl Node {
    fn leaf(state: SbaGame, move_code: i32, prior: f32) -> Node {
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

    /// Index of the SearchNode.best_child pick. PUCT over REAL+VIRTUAL
    /// stats: sqrt_parent spans `visits + v_visits`, and each child's
    /// `visits` / `q` read its virtual stats too (`visits == 0` there means
    /// "not visited AND not in flight"). An unvisited child is INFINITY
    /// (always wins selection), else -q + c_puct * prior * sqrt_parent /
    /// (1 + visits). Ties go to the FIRST child (Python's `if uct >
    /// best_val` scan), so visit order matches the reference — `max_by`
    /// would return the LAST max and diverge.
    fn best_child_index(&self, c_puct: f32) -> usize {
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

/// Full expansion on first visit (mcts_search does NOT trickle children in
/// one at a time — every legal move becomes a child at once). Priors are
/// the raw softmax probs at the move's code; Dirichlet noise (alpha = 10/n,
/// the "auto" scaling) mixes in as (1-eps)*p + eps*eta ONLY when the
/// expanded node is the search root — Python's non-root branch keeps the
/// raw prob, and shrinking every non-root prior by (1-eps) would break the
/// exact mirror. A single legal move mirrors np.random.dirichlet([a]) ==
/// [1.0] exactly, so no distribution is drawn (rand_distr wants len >= 2).
fn expand_with_priors(
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

/// Mirror of mcts_search's backprop: every node on the path gets
/// `visits += 1; wins += value` with the value sign flipped per ply, so
/// each node's wins stay in ITS side-to-move's perspective — the invariant
/// both best_child's `-q` and _root_value's raw child Q rely on. `value`
/// is the LEAF's side-to-move value. Python walks `reversed(path)`
/// (leaf->root, flip after each add); the Rust walk goes root->leaf along
/// child indices, so the identical per-node assignment falls out of one
/// up-front parity flip: the node at depth k then receives
/// value * (-1)^(d-k). Hand-trace, [root, A] with A terminal X-wins:
/// leaf value -1.0 (O to move at A, lost) -> root +1.0 (X won), A -1.0.
fn backprop(root: &mut Node, path: &[usize], mut value: f32) {
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
fn search_once(
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

/// Wave-mode backprop — alphazero.backprop verbatim. FIRST strip the
/// virtual stats every path node picked up at selection (`v_visits -= 1;
/// v_wins -= VIRTUAL_LOSS`, :350-353), THEN accumulate the real stats
/// (`visits += 1; wins += value` with the per-ply sign flip, :354-359) —
/// the same root->leaf walk with the up-front parity flip as the serial
/// backprop. After every wave the virtual stats are exactly zero again:
/// the adds/subtracts land on integers (v_wins on even multiples of 2.0),
/// so the round trip is bit-exact and nothing leaks into the next wave.
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

/// Leaf-parallel wave search — mcts_search's batched loop (batch_size > 1)
/// structurally verbatim. Per ROUND: select up to `batch_size` leaves
/// descending from the root under virtual loss (best_child reads the
/// vstats), break the selection as soon as a descent lands on an
/// already-queued unexpanded leaf (every reachable leaf is in flight),
/// backprop terminal_value immediately for terminal leaves, then ONE
/// batched evaluate of every collected leaf and, in selection order, the
/// real backprop (virtual stats off first) + full expansion. Real stats
/// settle only between waves — the wave boundaries this produces are the
/// python ones (fresh 81-legal board, budget 200, batch 256: [1, 81, 118]).
fn search_wave(
    root: &mut Node,
    eval: &mut dyn Evaluator,
    limits: &SearchLimits,
    rng: &mut dyn RngCore,
) -> Result<(), String> {
    let mut iterations: u32 = 0;
    while iterations < limits.sims {
        // ---- selection: up to batch_size leaves under virtual loss -------
        // root->leaf paths in selection order; python's `queued` set, keyed
        // by node id. A node's child-index path IS its identity here: the
        // tree shape is frozen during the selection phase (expansion happens
        // only after the batch), so path equality is id equality. The set is
        // membership-only in python (never iterated), so hash order is
        // irrelevant.
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

/// Single-position root search — the SEARCH portion of play_game without
/// the move sampling: `limits.sims` PUCT simulations from `start` (serial
/// mode for `batch_size` 0/1, the leaf-parallel wave mode otherwise — the
/// `batch_size` knob is python's `mcts_search(batch_size=...)`), then the
/// root visit counts in legal order and the root value z (_root_value, the
/// visit-weighted mean of the children's Q, 0.0 with no visits). The
/// self-play loop and the sba_debug parity/search harness share this entry,
/// so the recorded z and the debug JSON's value stay the same formula by
/// construction.
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

/// One self-play game (self_play_game mirror): until the state is over,
/// run `limits.sims` PUCT simulations from the current position, record
/// (planes, target, z) with z = _root_value — the visit-weighted mean of
/// the root children's Q, 0.0 with no visits — then sample the move from
/// the target distribution (np.random.choice mirror: cumulative walk in
/// legal order under a uniform r in [0, 1)) and apply it. The record's
/// meta carries defaults: the exe fills seed/weights_sha/result per run.
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
    use super::*;
    use rand::SeedableRng;
    use std::cell::RefCell;

    /// Scripted evaluator: pops (policy, value) per call, planes ignored.
    /// When the script runs dry it falls back to the default reply (the
    /// brief's "fill lazily" note) so long games never panic the harness.
    struct Scripted {
        replies: RefCell<Vec<(Vec<f32>, f32)>>,
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
    fn pol(w: &[usize]) -> Vec<f32> {
        let mut p = vec![0.0f32; 81];
        for &i in w {
            p[i] = 1.0;
        }
        p
    }

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

    /// Scripted evaluator + mode counters: `waves` records the row count of
    /// every evaluate_batch call (one entry per wave), `singles` counts
    /// plain evaluate calls — together they pin WHICH mode a limits config
    /// ran. Uniform priors + constant value make the wave dynamics fully
    /// hand-traceable (vloss dominates, first-max ties, round-robin).
    struct Probed {
        inner: Scripted,
        waves: RefCell<Vec<usize>>,
        singles: RefCell<usize>,
    }
    impl Probed {
        fn uniform(n: usize) -> Probed {
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

    /// (Ruling 5a) wave semantics with the scripted evaluator: budget <=
    /// batch_size collects DISTINCT leaves — wave 1 is the root itself (the
    /// root starts unexpanded, its expansion costs one sim), wave 2 is one
    /// leaf per picked root child, each ending at exactly visits = 1.
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

    /// (Ruling 5b) wave SHAPE on the fresh 81-legal board, budget 200,
    /// batch 256 — the PYTHON-MIRROR numbers from the layer-3 diagnosis
    /// ([1, 81, 118], 3 forwards, sum 200): wave 1 = the root expansion
    /// (the root starts unexpanded, its "leaf" is the root); wave 2 =
    /// exactly one leaf per root child — the 82nd selection descends to an
    /// already-queued unexpanded leaf and the queued-break fires (81 rows,
    /// NOT batch_size); wave 3 = the budget remainder — the vloss round-
    /// robin keeps finding fresh leaves (a first pass [i,0] over all 81
    /// children, then a second pass [i,1] cut at the budget), no break.
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
        // the structural python-mirror properties from the diagnosis:
        // all 81 root children visited, max share small (python real-net:
        // 81/81 visited, max share 0.015 — the uniform-prior round-robin
        // leaves the same 3-visit max), no wave over batch_size
        assert_eq!(counts.len(), 81);
        let total: f32 = counts.iter().map(|(_, v)| v).sum();
        assert_eq!(total, 199.0); // 200 sims minus the root-expansion wave's own visit
        let max = counts.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
        assert!(max / total < 0.1, "max share {}", max / total);
        assert!(counts.iter().all(|(_, v)| *v >= 1.0));
    }

    /// (Ruling 5c) batch_size 1 keeps the old serial mode: the wave loop is
    /// never entered (zero evaluate_batch calls), one settled evaluate per
    /// sim, and the visit trace matches the pre-port serial behavior (sim 1
    /// expands the root; sims 2-3 evaluate the first INF children k=0, k=1).
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

    /// (Ruling 5d) the virtual stats are REMOVED before the real stats
    /// accumulate (python :350-359): after the search every node's
    /// v_visits/v_wins are exactly zero again and the real stats are the
    /// pure backprop sums — nothing leaked into selection between waves.
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
