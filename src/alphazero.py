# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""AlphaZero-style neural-guided MCTS for Tic Tac Toe.

Self-contained: works with any game object exposing the SBA.py interface
(NormalGame / UltimateGame): board/micro+macro, legal_moves(), is_over(),
result(), clone(), make_move(...), current.

The CLI trains/evaluates the Ultimate model (recommended — Normal Tic Tac
Toe is a solved game); the train()/load_model() functions also accept
'normal' and will use models/az_normal.pt:

    python alphazero.py train --games 400 --sims 80

Evaluation vs random:
    python alphazero.py eval --games 30 --sims 200

Models are saved to ./models/az_<game>.pt

Maintenance notes:
- The CLI targets Ultimate (decision D4); Normal support is kept in the
  library functions only, for experimentation on a solved game.
- Without a trained model in models/, the engine falls back to MCTS.
"""

import argparse
import math
import os
import random
import sys
import time

# Windows + TheRock ROCm wheels: MIOpen (PyTorch's conv backend) keeps its
# find-db and kernel cache in the system TEMP by default. When TEMP points at a
# network/slow drive, the first conv2d can hang or raise
# miopenStatusUnknownError, so pin the caches to the local project disk
# (user-set overrides are respected).
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.name == 'nt':
    os.environ.setdefault('MIOPEN_USER_DB_PATH', os.path.join(HERE, '.miopen'))
    os.environ.setdefault('MIOPEN_CUSTOM_CACHE_DIR', os.path.join(HERE, '.miopen_cache'))
    # MIOpen's runtime kernel benchmarking is unreliable on Windows (it can
    # report "Invalid elapsed time" and occasionally crash the process), so
    # use the persistent find-db without re-benchmarking. Combined with the
    # fixed search batch shape, each conv config is looked up once.
    os.environ.setdefault('MIOPEN_FIND_MODE', '1')
    os.environ.setdefault('MIOPEN_FIND_ENFORCE', 'NONE')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = os.path.join(HERE, 'models')
X = 'X'
O = 'O'
EMPTY = ''


# ---------------------------------------------------------------
# Game helpers (duck-typed against SBA.NormalGame / SBA.UltimateGame)
# ---------------------------------------------------------------

def is_normal(game):
    return hasattr(game, 'board')


def size_of(game):
    return 3 if is_normal(game) else 9


def move_index(game, move):
    if is_normal(game):
        return move
    m, i = move
    return m * 9 + i


def encode(game):
    """Return tensor [4, size, size]:
    plane 0 = X marks, plane 1 = O marks, plane 2 = turn (1.0 if X to move),
    plane 3 = active macro cell block (ultimate only, 0 everywhere if free)."""
    size = size_of(game)
    planes = np.zeros((4, size, size), dtype=np.float32)
    if is_normal(game):
        for i, mark in enumerate(game.board):
            if mark:
                r, c = divmod(i, 3)
                planes[0 if mark == X else 1, r, c] = 1.0
    else:
        for m in range(9):
            mr, mc = divmod(m, 3)
            for i, mark in enumerate(game.micro[m]):
                if mark:
                    r, c = divmod(i, 3)
                    planes[0 if mark == X else 1, mr * 3 + r, mc * 3 + c] = 1.0
        if game.active_macro is not None:
            mr, mc = divmod(game.active_macro, 3)
            planes[3, mr * 3:(mr + 1) * 3, mc * 3:(mc + 1) * 3] = 1.0
    if game.current == X:
        planes[2, :, :] = 1.0
    return torch.as_tensor(planes)  # zero-copy: planes is float32 & contiguous


def apply_move(game, move):
    if is_normal(game):
        game.make_move(move)
    else:
        game.make_move(*move)


def terminal_value(game):
    # Value from the perspective of the player who would move next: at a
    # game-over state that player is always the loser, so a win is -1.0 and a
    # draw is 0.0 (matches the network's current-player value convention).
    if game.result() == 'D':
        return 0.0
    return -1.0


# ---------------------------------------------------------------
# Network
# ---------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AZNet(nn.Module):
    def __init__(self, size, channels=32, blocks=3):
        super().__init__()
        self.size = size
        n_out = size * size
        self.conv_in = nn.Conv2d(4, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])
        self.pol_conv = nn.Conv2d(channels, 8, 1)
        self.pol_bn = nn.BatchNorm2d(8)
        self.pol_fc = nn.Linear(8 * n_out, n_out)
        self.val_conv = nn.Conv2d(channels, 8, 1)
        self.val_bn = nn.BatchNorm2d(8)
        self.val_fc1 = nn.Linear(8 * n_out, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for b in self.blocks:
            x = b(x)
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = self.pol_fc(p.flatten(1))
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = F.relu(self.val_fc1(v.flatten(1)))
        return p, torch.tanh(self.val_fc2(v))


# ---------------------------------------------------------------
# Neural-guided MCTS
# ---------------------------------------------------------------

ROLLOUT_WEIGHT = 0.5  # leaf value = (1-w)*network + w*random playout; 0 = pure network
VIRTUAL_LOSS = 2.0    # applied during batched selection so parallel workers diverge
BATCH_SIZE = 256      # default leaf-parallel batch size for network inference
DIRICHLET_EPS = 0.25  # root-prior noise weight in self-play (standard AZ value)


class SearchNode:
    """Node with plain-MCTS style incremental expansion and UCB exploration.

    Children are created one at a time (untried list) and unvisited children
    always win selection, which gives much stronger exploration than expanding
    every child at once -- important for Ultimate's wide branching factor.
    q is stored from the perspective of the player to move at this state.
    """

    __slots__ = ('state', 'move', 'children', 'visits', 'wins', 'q',
                 'untried', 'priors', 'v_visits', 'v_wins')

    def __init__(self, state, move):
        self.state = state
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.q = 0.0
        # Order never matters here: children are chosen via best_child(), and
        # expansion iterates untried in place, so no shuffle is needed.
        self.untried = state.legal_moves() if not state.is_over() else []
        self.priors = None  # {move: prior} filled once by the policy head
        self.v_visits = 0   # virtual statistics for batched (leaf-parallel) search
        self.v_wins = 0.0

    def best_child(self, c_puct):
        # AlphaZero PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)).
        # The numerator sqrt(sum of sibling visits) equals sqrt(visits to this node).
        sqrt_parent = math.sqrt(self.visits + self.v_visits)
        best, best_val = None, -math.inf
        for child in self.children:
            visits = child.visits + child.v_visits
            if visits == 0:
                uct = math.inf
            else:
                prior = self.priors.get(child.move, 1.0) if self.priors else 1.0
                q = (child.wins + child.v_wins) / visits
                uct = (-q + c_puct * prior * sqrt_parent / (1.0 + visits))
            if uct > best_val:
                best, best_val = child, uct
        return best


_C_ROLLOUT_TRIED = False


class _CRootView:
    """Minimal root-like view over C MCTS stats so _root_value() consumes
    the native search output unchanged (child.visits / child.wins)."""
    __slots__ = ('children',)

    def __init__(self, stats):
        from types import SimpleNamespace
        self.children = [SimpleNamespace(visits=v, wins=w)
                         for _m, v, w, _mo, _d in stats]


def _c_rollout_value(g):
    """Native heuristic playout via the C rollout kernel (D32); value from
    g's current player's perspective. Returns None when the DLL is missing
    or the position is somehow unresolvable — caller falls back to Python.
    The kernel runs the same win/block-biased policy as ai._rollout_move.
    """
    global _C_ROLLOUT_TRIED
    if not _C_ROLLOUT_TRIED:
        _C_ROLLOUT_TRIED = True
        try:
            from native import cengine as _ce  # noqa: F401
        except ImportError:
            return None
    try:
        from native import cengine as _ce
        r = _ce.rollout_state(g)
    except Exception:
        return None
    if r is None:
        return None
    if r == 'D':
        return 0.0
    return 1.0 if r == g.current else -1.0


def rollout_value(g):
    """Heuristic playout from g; value from the perspective of g's current player.

    Uses the native C rollout kernel when available (D32) and falls back to
    the ai.py win/block-biased policy (lazy import: ai.py loads alphazero on
    demand, never at module level). Tactical playouts are far stronger than
    uniform random on Ultimate Tic Tac Toe.
    """
    v = _c_rollout_value(g)
    if v is not None:
        return v
    from ai import _rollout_move
    player = g.current
    state = g.clone()
    while not state.is_over():
        legal = state.legal_moves()
        if not legal:
            break
        apply_move(state, _rollout_move(state, legal))
    r = state.result()
    if r == 'D':
        return 0.0
    return 1.0 if r == player else -1.0




@torch.no_grad()
def mcts_search(game, model, budget, c_puct=1.5, rollout_weight=None,
                batch_size=BATCH_SIZE, dirichlet_alpha=None,
                dirichlet_eps=DIRICHLET_EPS):
    """Neural-guided MCTS with leaf-parallel batched inference (standard AZ).

    Every node is expanded in full (all children created with priors) on its
    first visit, so selection always descends to an unexpanded leaf. Up to
    ``batch_size`` leaves are selected per round (virtual loss keeps parallel
    workers on different branches) and evaluated with one batched network call,
    cutting per-iteration kernel-launch overhead on ROCm/CUDA.

    Returns (root SearchNode, {move: visit_count}).

    ``dirichlet_alpha`` adds exploration noise to the root's priors on its
    first expansion: pass ``'auto'`` for 10/len(moves) (standard AZ scaling)
    or a float for a fixed concentration; None (default) means no noise.
    """
    if rollout_weight is None:
        rollout_weight = ROLLOUT_WEIGHT
    model.eval()  # inference: BatchNorm must use running stats, not batch stats
    root_state = game.clone()
    root = SearchNode(root_state, None)

    def backprop(path, value):
        for n in path:
            n.v_visits -= 1
            n.v_wins -= VIRTUAL_LOSS
        for n in reversed(path):
            n.visits += 1
            n.wins += value
            n.q = n.wins / n.visits
            value = -value

    iterations = 0
    while iterations < budget:
        leaves = []  # (path, node) unexpanded leaves -> value batch
        queued = set()  # node ids already collected this round
        while iterations < budget and len(leaves) < batch_size:
            node = root
            path = [root]
            while node.children:
                node = node.best_child(c_puct)
                path.append(node)
            if node.untried and id(node) in queued:
                break  # every reachable leaf is already queued: evaluate now
            queued.add(id(node))
            for n in path:
                n.v_visits += 1
                n.v_wins += VIRTUAL_LOSS
            iterations += 1
            if node.untried == []:
                backprop(path, terminal_value(node.state))
            else:
                leaves.append((path, node))

        # Network: one batched forward for every selected leaf. The batch is
        # padded to a fixed size so MIOpen sees one tensor shape per process
        # (one kernel lookup instead of a find pass per round size). Padded
        # rows are GPU-side tiles of the real batch — encode runs once per
        # real leaf (the old code re-encoded a duplicate state 256x per
        # round), and the padded outputs are discarded either way. Policies
        # and values ride home in ONE .cpu() sync instead of two.
        if leaves:
            n_real = len(leaves)
            pad = batch_size - n_real
            states = torch.stack(
                [encode(node.state) for _, node in leaves]).to(DEVICE)
            if pad:
                # tile whole copies up to EXACTLY batch_size rows: a clamp
                # (states[:pad] with pad > n_real, or a floor-div rep count)
                # would yield a per-round variable shape, and MIOpen picks a
                # pathological kernel for odd shapes (252 rows ran 12x slower
                # than 256 in isolation on this ROCm build)
                reps = -(-batch_size // n_real)   # ceil
                states = states.repeat((reps,) + (1,) * (states.dim() - 1))
                states = states[:batch_size]
            policies, values = model(states)
            joined = torch.cat(
                (F.softmax(policies, dim=1), values), dim=1).cpu().numpy()
            probs_batch = joined[:n_real, :-1]
            values_batch = joined[:n_real, -1]
            for (path, node), probs, value in zip(leaves, probs_batch, values_batch):
                value = float(value)
                if rollout_weight > 0:
                    value = ((1.0 - rollout_weight) * value
                             + rollout_weight * rollout_value(node.state))
                backprop(path, value)
                if node.untried:
                    # expand in full (idempotent: duplicate leaves skip)
                    moves = node.untried
                    if dirichlet_alpha is not None and node is root:
                        alpha = (10.0 / len(moves) if dirichlet_alpha == 'auto'
                                 else dirichlet_alpha)
                        eta = np.random.dirichlet([alpha] * len(moves))
                        node.priors = {
                            m: ((1.0 - dirichlet_eps)
                                * float(probs[move_index(node.state, m)])
                                + dirichlet_eps * float(e))
                            for m, e in zip(moves, eta)}
                    else:
                        node.priors = {m: float(probs[move_index(node.state, m)])
                                       for m in moves}
                    for m in moves:
                        child_state = node.state.clone()
                        apply_move(child_state, m)
                        node.children.append(SearchNode(child_state, m))
                    node.untried = []
    counts = {child.move: child.visits for child in root.children}
    return root, counts


def mcts_visit_counts(game, model, budget, c_puct=1.5, batch_size=BATCH_SIZE,
                      dirichlet_alpha=None, dirichlet_eps=DIRICHLET_EPS):
    _, counts = mcts_search(game, model, budget, c_puct, batch_size=batch_size,
                            dirichlet_alpha=dirichlet_alpha,
                            dirichlet_eps=dirichlet_eps)
    return counts


# ---------------------------------------------------------------
# Policy inspection (D40): the raw priors, no search
# ---------------------------------------------------------------

@torch.no_grad()
def policy_vector(game, model):
    """Raw policy head for `game`: an 81-dim vector masked to the legal
    moves and renormalised (mover's view — the net is state-only). Returns
    None when no usable model output lands on a legal move (e.g. an
    untrained net that prefers only illegal moves). One forward pass, no
    search — this is what the assistant's board heat shows, distinct from
    the visit-share ranking the MCTS analysis lists."""
    model.eval()  # BatchNorm running stats, same as the search path
    state = encode(game).unsqueeze(0).to(DEVICE)
    policies, _values = model(state)
    probs = F.softmax(policies, dim=1)[0].cpu().numpy()
    mask = np.zeros(81, dtype=np.float32)
    for m, i in game.legal_moves():
        mask[m * 9 + i] = 1.0
    probs = probs * mask
    total = probs.sum()
    if total <= 0:
        return None
    return probs / total


def select_move(game, model, budget=800, temp=0.0, c_puct=1.5,
                batch_size=BATCH_SIZE):
    model = model.to(DEVICE)  # callers may pass a CPU model
    root, counts = mcts_search(game, model, budget, c_puct, batch_size=batch_size)
    if not counts:
        return random.choice(game.legal_moves())
    moves = list(counts.keys())
    n = np.array([counts[m] for m in moves], dtype=np.float64)
    if temp <= 0.01:
        # Robust pick: among sufficiently visited children choose the move with
        # the best value for the side to move; fall back to the most visited.
        pool = [m for m in moves if counts[m] >= max(5, budget // 100)]
        if not pool:
            pool = moves
        if len(pool) > 1:
            child_by_move = {c.move: c for c in root.children}
            best_m, best_q = pool[0], -float('inf')
            for move in pool:
                q = -child_by_move[move].q
                if q > best_q:
                    best_q, best_m = q, move
            return best_m
        return pool[0]
    p = n ** (1.0 / temp)
    total = p.sum()
    if total <= 0:
        return random.choice(moves)
    p /= total
    return moves[int(np.random.choice(len(moves), p=p))]


def alphazero_move(game, model, budget=800, batch_size=BATCH_SIZE):
    return select_move(game, model, budget, temp=0.0, batch_size=batch_size)


# ---------------------------------------------------------------
# Training (self-play)
# ---------------------------------------------------------------

def model_path(game_type):
    return os.path.join(MODEL_DIR, 'az_%s.pt' % game_type)


def alphazero_model_dir():
    return MODEL_DIR


def make_game(game_type):
    if game_type == 'normal':
        from game import NormalGame
        return NormalGame()
    from game import UltimateGame
    return UltimateGame()


def _save_model(model, size, channels, blocks, path):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'size': size,
                'channels': channels, 'blocks': blocks}, path)


class _TrainingCsv:
    """Per-iteration training-curve logger (D41): one CSV row per completed
    game, flushed immediately so a killed run still leaves usable data.

    One file = one run lineage: a fresh run overwrites, a resumed run
    appends without a second header, so the report's single-run curve can
    span both stages of the recipe."""

    HEADER = ['stage', 'game', 'elapsed_s', 'lr', 'loss', 'positions',
              'temp', 'eval_w', 'eval_d', 'eval_l', 'ckpt']

    def __init__(self, path, resume=False):
        import csv as _csv
        self._append = bool(resume) and os.path.exists(path)
        self._f = open(path, 'a' if self._append else 'w', newline='',
                       encoding='utf-8')
        self._w = _csv.writer(self._f)
        if not self._append:
            self._w.writerow(self.HEADER)

    def write(self, stage, game, elapsed_s, lr, loss, positions, temp,
              ev=None, ckpt=''):
        self._w.writerow([stage, game, f'{elapsed_s:.1f}', f'{lr:.6f}',
                          '' if loss is None else f'{loss:.4f}',
                          positions, f'{temp:.3f}',
                          ev[0] if ev else '', ev[1] if ev else '',
                          ev[2] if ev else '', ckpt])
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


# ---------------------------------------------------------------
# Recipe helpers: D4 flip augmentation and root-mean value target
# ---------------------------------------------------------------

def _aug_state(state, t):
    """Apply D4 transform t (0..7) to the spatial planes of an encoded state."""
    if t == 0:
        return state
    s = state
    if t == 1:
        s = torch.rot90(s, -1, dims=(1, 2))
    elif t == 2:
        s = torch.rot90(s, 2, dims=(1, 2))
    elif t == 3:
        s = torch.rot90(s, 1, dims=(1, 2))
    elif t == 4:
        s = torch.flip(s, dims=(2,))
    elif t == 5:
        s = torch.flip(s, dims=(1,))
    elif t == 6:
        s = torch.transpose(s, 1, 2)
    else:  # t == 7: anti-diagonal reflection
        s = torch.flip(torch.transpose(s, 1, 2), dims=(1, 2))
    return s


def _aug_perms(size):
    """Policy-index permutations of the 8 D4 transforms of an NxN board.

    Derived by simulating ``_aug_state`` on one-hot grids, so the policy
    permutation is guaranteed to match the spatial tensor transform.
    """
    perms = []
    for t in range(8):
        perm = []
        for i in range(size * size):
            r, c = divmod(i, size)
            grid = torch.zeros(1, size, size)
            grid[0, r, c] = 1.0
            moved = _aug_state(grid, t)
            rr, cc = torch.nonzero(moved[0])[0].tolist()
            perm.append(rr * size + cc)
        perms.append(perm)
    return perms


_AUG_POLICY = {}


def _augment_sample(state, target, t, size):
    """Apply D4 transform t (0..7) to an encoded state tensor and policy.

    Both the spatial op and the policy permutation come from ``_aug_state``,
    so the active-macro plane and macro routing stay consistent. ``target``
    is a 1-D numpy array over all cells.
    """
    if t == 0:
        return state, target
    perms = _AUG_POLICY.get(size)
    if perms is None:
        perms = _aug_perms(size)
        _AUG_POLICY[size] = perms
    # Scatter: the probability at old cell i moves to new cell perms[t][i]
    # (the forward map), matching the spatial transform direction.
    moved = np.empty_like(target)
    moved[perms[t]] = target
    return _aug_state(state, t), moved


def _root_value(root):
    """Mean value for the side to move, weighted by root children visits.

    root.mover is None (its own wins only ever count draws), so the value is
    derived from the children; each child's wins are recorded from the
    perspective of the player who moved into it (= the side to move).
    """
    tot_v = tot = 0.0
    for child in root.children:
        if child.visits:
            tot_v += child.visits * (child.wins / child.visits)
            tot += child.visits
    return tot_v / tot if tot else 0.0


def self_play_game(make_game_fn, model, sims, temp,
                   dirichlet_eps=DIRICHLET_EPS, value_target='root_mean',
                   use_network=True):
    game = make_game_fn()
    states, targets, players, values = [], [], [], []
    while not game.is_over():
        legal = game.legal_moves()
        if use_network:
            root, counts = mcts_search(game, model, sims,
                                       dirichlet_alpha='auto',
                                       dirichlet_eps=dirichlet_eps)
        else:
            # Stage 1: pure MCTS (heuristic rollouts, no network) so policy
            # and value targets come from classical search quality. The
            # native C tree search (D30) runs it ~37x faster; root stats
            # carry (move, visits, wins, mover) exactly like MCTSNode.
            from native import cengine as _ce
            from ai import _c_search_opts
            root_mv, stats, _tree = _ce.mcts_search(
                game, sims, opts=_c_search_opts())
            if stats:
                counts = {(m // 9, m % 9): int(v)
                          for m, v, w, _mo, _d in stats}
                root = _CRootView(stats)
            else:
                from ai import mcts_search as pure_mcts
                root = pure_mcts(game.clone(), sims)
                counts = {c.move: c.visits for c in root.children}
        value = _root_value(root)
        n = np.array([counts.get(m, 0) for m in legal], dtype=np.float64)
        n = n + 1e-8
        if temp > 0.01:
            p = n ** (1.0 / temp)
        else:
            p = np.zeros_like(n)
            p[int(np.argmax(n))] = 1.0
        p = p / p.sum()
        states.append(encode(game))
        target = np.zeros(size_of(game) ** 2, dtype=np.float32)
        for mv, pr in zip(legal, p):
            target[move_index(game, mv)] = pr
        targets.append(target)
        players.append(game.current)
        values.append(value)
        idx = int(np.random.choice(len(legal), p=p))
        apply_move(game, legal[idx])
    result = game.result()
    if value_target == 'root_mean':
        zs = list(values)
    else:
        zs = []
        for p in players:
            zs.append(0.0 if result == 'D' else (1.0 if p == result else -1.0))
    return states, targets, zs


# ---------------------------------------------------------------
# Multi-process self-play (Windows spawn-safe)
# ---------------------------------------------------------------
# Worker processes each own a GIL and a model copy, so the CPU-bound
# rollouts/tree code parallelizes for real (unlike threads). Every task
# carries the latest weights, so workers stay stateless; results come back
# through a shared queue and the main process alone runs the optimizer.

_MP_RESULTS_Q = None


def _mp_worker_init(results_q):
    global _MP_RESULTS_Q
    _MP_RESULTS_Q = results_q
    # spawn children inherit the parent's console; 8 of them spamming
    # MIOpen/torch warnings tears apart a fullscreen TUI - divert to a file
    try:
        import atexit
        log_path = os.path.join(MODEL_DIR, '..', 'logs',
                                'mp-worker-%d.log' % os.getpid())
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        sink = open(os.path.abspath(log_path), 'w', buffering=1)
        sys.stdout = sink
        sys.stderr = sink
        atexit.register(sink.close)
    except Exception:
        pass


def _mp_play_game(idx, weights, size, channels, blocks, sims,
                  dirichlet_eps, games, game_type, rollout_weight,
                  value_target, use_network):
    global ROLLOUT_WEIGHT
    ROLLOUT_WEIGHT = rollout_weight
    model = AZNet(size, channels, blocks).to(DEVICE)
    model.load_state_dict(weights)
    model.eval()
    temp = max(0.05, 1.0 - idx / games)
    try:
        states, targets, zs = self_play_game(
            lambda: make_game(game_type), model, sims, temp, dirichlet_eps,
            value_target=value_target, use_network=use_network)
        _MP_RESULTS_Q.put((idx, (states, targets, zs)))
    except Exception as e:  # noqa: BLE001 - keep the run alive
        _MP_RESULTS_Q.put((idx, e))


def train(game_type, games=300, sims=80, eval_every=25, eval_games=20,
          lr=1e-3, seed=None, save=True, quiet=False,
          steps_per_game=5, replay_cap=500, channels=128, blocks=5,
          ckpt_every=None, dirichlet_eps=DIRICHLET_EPS, workers=1,
          weight_decay=1e-4, value_target='root_mean', policy_loss='kl',
          augment=True, stage=2, resume=None, save_path=None,
          eval_ai='MCTS', eval_sims=800, csv_path=None,
          on_event=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    size = 3 if game_type == 'normal' else 9
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        size = ckpt['size']
        channels = ckpt.get('channels', 128)
        blocks = ckpt.get('blocks', 5)
        model = AZNet(size, channels, blocks).to(DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        if not quiet:
            print('[%s] resumed from %s (channels=%d blocks=%d)'
                  % (game_type, resume, channels, blocks), flush=True)
    else:
        model = AZNet(size, channels, blocks).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    make_game_fn = lambda: make_game(game_type)
    start = time.time()
    save_path = save_path or model_path(game_type)
    use_network = stage == 2
    # D41: the training curve defaults to next to the model; explicit
    # --csv wins, save=False without --csv means no file at all
    if csv_path is None and save:
        csv_path = os.path.splitext(save_path)[0] + '.csv'
    csv_log = _TrainingCsv(csv_path, resume) if csv_path else None
    positions_total = 0
    # rich progress on a real terminal; plain milestone prints elsewhere
    # (CI logs, pipes, quiet runs). The CSV is the data record either way.
    progress = None
    task = None
    if not quiet:
        try:
            import cli_output
            con = cli_output.console()
        except ImportError:
            con = None
        if con is not None and sys.stdout.isatty():
            from rich.progress import (BarColumn, Progress, SpinnerColumn,
                                       TaskProgressColumn, TextColumn,
                                       TimeElapsedColumn, TimeRemainingColumn)
            progress = Progress(
                SpinnerColumn(),
                TextColumn('[bold]{task.description}'),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TextColumn('{task.completed}/{task.total} games'),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=con)
            progress.start()
            task = progress.add_task(f'train {game_type} @ {sims} sims/move',
                                     total=games)

    def evaluate():
        # in-training gauge: AZ@100 vs a classical MCTS opponent (C tree) —
        # Random loses its meaning once the network plays at all
        from ai import get_ai_move
        wins = draws = losses = 0
        for _ in range(eval_games):
            for az_first in (True, False):
                game = make_game_fn()
                while not game.is_over():
                    az_turn = game.current == X if az_first else game.current == O
                    if az_turn:
                        apply_move(game, alphazero_move(game, model, 100))
                    else:
                        apply_move(game, get_ai_move(game, eval_ai, eval_sims))
                r = game.result()
                az_won = (r == X and az_first) or (r == O and not az_first)
                if az_won:
                    wins += 1
                elif r == 'D':
                    draws += 1
                else:
                    losses += 1
        return wins, draws, losses

    replay = []
    try:
        if workers > 1:
            return _train_multiprocess(
                make_game_fn, model, optimizer, evaluate, replay, games, sims,
                eval_every, lr, save, quiet, steps_per_game, replay_cap, size,
                channels, blocks, ckpt_every, dirichlet_eps, workers, game_type,
                start, ROLLOUT_WEIGHT, value_target, use_network,
                policy_loss, augment, save_path, csv_log, progress, task,
                on_event)
        for it in range(1, games + 1):
            temp = max(0.05, 1.0 - it / games)
            states, targets, zs = self_play_game(
                make_game_fn, model, sims, temp, dirichlet_eps,
                value_target=value_target, use_network=use_network)
            positions_total += len(states)
            replay.append((states, targets, zs))
            if len(replay) > replay_cap:
                replay.pop(0)
            # Linear LR annealing: full rate early, a 5x drop by the final game.
            optimizer.param_groups[0]['lr'] = lr * max(0.2, 1.0 - it / games)
            model.train()
            total_loss = 0.0
            for _ in range(steps_per_game):
                g_states, g_targets, g_zs = random.choice(replay)
                n_pos = len(g_states)
                # Fixed batch size (with replacement when a game is short) keeps
                # one tensor shape per process: MIOpen only looks up kernels once.
                idx = np.random.choice(n_pos, 128, replace=True)
                states_batch = [g_states[i] for i in idx]
                targets_batch = [g_targets[i] for i in idx]
                if augment:
                    for j in range(len(idx)):
                        t = random.randrange(8)
                        if t:
                            states_batch[j], targets_batch[j] = _augment_sample(
                                states_batch[j], targets_batch[j], t, size)
                batch = torch.stack(states_batch).to(DEVICE)
                target = torch.stack(
                    [torch.as_tensor(tg) for tg in targets_batch]).to(DEVICE)
                z = torch.tensor([g_zs[i] for i in idx], dtype=torch.float32).to(DEVICE)
                optimizer.zero_grad()
                pred_p, pred_v = model(batch)
                logp = F.log_softmax(pred_p, dim=1)
                if policy_loss == 'kl':
                    loss_p = (target * (target.clamp_min(1e-8).log() - logp)
                              ).sum(dim=1).mean()
                else:
                    loss_p = -(logp * target).sum(dim=1).mean()
                loss_v = F.mse_loss(pred_v[:, 0], z)
                (loss_p + loss_v).backward()
                optimizer.step()
                total_loss += float(loss_p + loss_v)
            model.eval()
            ckpt_at = ''
            if save and ckpt_every and it % ckpt_every == 0:
                _save_model(model, size, channels, blocks, save_path)
                ckpt_at = save_path
                if on_event:
                    on_event({'type': 'ckpt', 'game': it, 'path': save_path})
                if not quiet:
                    line = 'checkpoint %d/%d -> %s' % (it, games, save_path)
                    if progress:
                        progress.console.print('[yellow]' + line + '[/yellow]')
                    else:
                        print('[%s] checkpoint %d/%d -> %s'
                              % (game_type, it, games, save_path), flush=True)
            if it % max(1, eval_every) == 0 or it == games:
                if on_event:
                    on_event({'type': 'eval_start', 'game': it})
            ev = None
            # the gauge runs regardless of quiet — quiet mutes the PRINT, not
            # the data: the CSV curve needs the win-rate column
            if it % max(1, eval_every) == 0 or it == games:
                el = time.time() - start
                ev = evaluate()
                w, d, l = ev
                if not quiet:
                    line = ('eval game %d/%d - win=%d draw=%d loss=%d - loss=%.3f'
                            ' - %.0fs - AZ@100 vs %s@%d'
                            % (it, games, w, d, l, total_loss / steps_per_game,
                               el, eval_ai, eval_sims))
                    if progress:
                        progress.console.print(line)
                    else:
                        print('[%s] game %d/%d  win=%d draw=%d loss=%d  loss=%.3f  (%.0fs)'
                              '  [eval: AZ@100 vs %s@%d]'
                              % (game_type, it, games, w, d, l,
                                 total_loss / steps_per_game, el, eval_ai, eval_sims),
                              flush=True)
            if csv_log:
                csv_log.write(stage, it, time.time() - start,
                              optimizer.param_groups[0]['lr'],
                              total_loss / steps_per_game, positions_total,
                              temp, ev, ckpt_at)

            if progress:
                progress.advance(task)
            if on_event:
                on_event({'type': 'game', 'game': it, 'games': games,
                          'stage': stage,
                          'lr': optimizer.param_groups[0]['lr'],
                          'loss': total_loss / steps_per_game,
                          'positions': positions_total, 'temp': temp,
                          'eval': ev, 'elapsed': time.time() - start})

        if save:
            _save_model(model, size, channels, blocks, save_path)
            if not quiet:
                print('saved', save_path)
            if on_event:
                on_event({'type': 'saved', 'path': save_path})
        return model
    finally:
        if csv_log:
            csv_log.close()
        if progress:
            progress.stop()


def _train_multiprocess(make_game_fn, model, optimizer, evaluate, replay, games,
                        sims, eval_every, lr, save, quiet, steps_per_game,
                        replay_cap, size, channels, blocks, ckpt_every,
                        dirichlet_eps, workers, game_type, start,
                        rollout_weight, value_target, use_network,
                        policy_loss, augment, save_path, csv_log=None,
                        progress=None, task=None, on_event=None):
    """Multi-process self-play.

    A worker pool plays the games (each task carries the latest weights and
    uses its own GPU model copy); the main process collects results, runs the
    optimizer, checkpoints and evaluates. Spawn context is required on
    Windows; the pool transparently replaces crashed workers.
    """
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    results_q = ctx.Queue()
    pool = ctx.Pool(workers, initializer=_mp_worker_init, initargs=(results_q,))

    def submit(idx):
        weights = {k: v.cpu() for k, v in model.state_dict().items()}
        return pool.apply_async(
            _mp_play_game,
            (idx, weights, size, channels, blocks, sims,
             dirichlet_eps, games, game_type, rollout_weight,
             value_target, use_network))

    next_idx = 1
    first_wave = min(workers, games)
    for k in range(min(workers * 2, games)):
        if 0 < k < first_wave:
            # every first task makes its worker build a HIP/CUDA context;
            # 8 simultaneous inits crashed the whole GPU stack (explorer
            # died with python) - stagger them so contexts come up one by
            # one; later tasks reuse warm workers
            time.sleep(1.5)
        submit(next_idx)
        next_idx += 1

    done = 0
    pos_total = 0
    stage = 2 if use_network else 1
    while done < games:
        idx, data = results_q.get()
        done += 1
        if progress:
            progress.advance(task)
        if isinstance(data, Exception):
            if on_event:
                on_event({'type': 'worker_failed', 'game': idx,
                          'error': str(data)})
            else:
                print('[%s] self-play worker failed game %d: %s'
                      % (game_type, idx, data), flush=True)
        else:
            states, targets, zs = data
            pos_total += len(states)
            replay.append((states, targets, zs))
            if len(replay) > replay_cap:
                replay.pop(0)
            optimizer.param_groups[0]['lr'] = lr * max(0.2, 1.0 - done / games)
            model.train()
            total_loss = 0.0
            for _ in range(steps_per_game):
                g_states, g_targets, g_zs = random.choice(replay)
                n_pos = len(g_states)
                # Fixed batch size (with replacement when a game is short)
                # keeps one tensor shape per process for MIOpen.
                idx2 = np.random.choice(n_pos, 128, replace=True)
                states_batch = [g_states[i] for i in idx2]
                targets_batch = [g_targets[i] for i in idx2]
                if augment:
                    for j in range(len(idx2)):
                        t = random.randrange(8)
                        if t:
                            states_batch[j], targets_batch[j] = _augment_sample(
                                states_batch[j], targets_batch[j], t, size)
                batch = torch.stack(states_batch).to(DEVICE)
                target = torch.stack(
                    [torch.as_tensor(tg) for tg in targets_batch]).to(DEVICE)
                z = torch.tensor([g_zs[i] for i in idx2],
                                 dtype=torch.float32).to(DEVICE)
                optimizer.zero_grad()
                pred_p, pred_v = model(batch)
                logp = F.log_softmax(pred_p, dim=1)
                if policy_loss == 'kl':
                    loss_p = (target * (target.clamp_min(1e-8).log() - logp)
                              ).sum(dim=1).mean()
                else:
                    loss_p = -(logp * target).sum(dim=1).mean()
                loss_v = F.mse_loss(pred_v[:, 0], z)
                (loss_p + loss_v).backward()
                optimizer.step()
                total_loss += float(loss_p + loss_v)
            model.eval()
            ckpt_at = ''
            if save and ckpt_every and done % ckpt_every == 0:
                _save_model(model, size, channels, blocks, save_path)
                ckpt_at = save_path
                if on_event:
                    on_event({'type': 'ckpt', 'game': done,
                              'path': save_path})
                if not quiet:
                    line = 'checkpoint %d/%d -> %s' % (done, games, save_path)
                    if progress:
                        progress.console.print('[yellow]' + line + '[/yellow]')
                    else:
                        print('[%s] checkpoint %d/%d -> %s'
                              % (game_type, done, games, save_path), flush=True)
            ev = None
            # gauge runs regardless of quiet — quiet mutes the print only
            if done % max(1, eval_every) == 0 or done == games:
                if on_event:
                    on_event({'type': 'eval_start', 'game': done})
                el = time.time() - start
                ev = evaluate()
                w, d, l = ev
                if not quiet:
                    line = ('eval game %d/%d - win=%d draw=%d loss=%d - '
                            'loss=%.3f - %.0fs'
                            % (done, games, w, d, l,
                               total_loss / steps_per_game, el))
                    if progress:
                        progress.console.print(line)
                    else:
                        print('[%s] game %d/%d  win=%d draw=%d loss=%d  loss=%.3f  (%.0fs)'
                              % (game_type, done, games, w, d, l,
                                 total_loss / steps_per_game, el), flush=True)
            if csv_log:
                # workers report the per-task index; its temperature schedule
                # mirrors _mp_play_game's
                csv_log.write(stage, done, time.time() - start,
                              optimizer.param_groups[0]['lr'],
                              total_loss / steps_per_game, pos_total,
                              max(0.05, 1.0 - idx / games), ev, ckpt_at)
            if on_event:
                on_event({'type': 'game', 'game': done, 'games': games,
                          'stage': stage,
                          'lr': optimizer.param_groups[0]['lr'],
                          'loss': total_loss / steps_per_game,
                          'positions': pos_total,
                          'temp': max(0.05, 1.0 - idx / games),
                          'eval': ev, 'elapsed': time.time() - start})
        if next_idx <= games:
            submit(next_idx)
            next_idx += 1

    pool.close()
    pool.join()
    if save:
        _save_model(model, size, channels, blocks, save_path)
        if on_event:
            on_event({'type': 'saved', 'path': save_path})
        if not quiet:
            print('saved', save_path)
    return model


def load_model(game_type):
    path = model_path(game_type)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location='cpu')
    model = AZNet(ckpt['size'],
                  ckpt.get('channels', 32), ckpt.get('blocks', 3)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main(argv=None):
    global ROLLOUT_WEIGHT
    ap = argparse.ArgumentParser(description='AlphaZero for (Ultimate) Tic Tac Toe')
    sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train',
                        help='self-play + optimization (the two-stage D22 recipe)')
    tr.add_argument('--games', type=int, default=300,
                    help='self-play games to train on (default: 300; the '
                         'D22 scale curve used 600/2000/6000)')
    tr.add_argument('--sims', type=int, default=80,
                    help='MCTS simulations per move in self-play (default: 80; '
                         'stage 1 typically 800, see --stage)')
    tr.add_argument('--eval-every', type=int, default=25,
                    help='run the eval gauge every N games (default: 25; long '
                         'runs should raise it - the gauge costs real search '
                         'time, roughly the self-play time at the default)')
    tr.add_argument('--eval-games', type=int, default=20,
                    help='games per eval event, doubled for both first players '
                         '(default: 20)')
    tr.add_argument('--eval-ai', type=str, default='MCTS',
                    help='in-training eval opponent (default: MCTS via the C tree)')
    tr.add_argument('--eval-sims', type=int, default=800,
                    help='sim budget of the in-training eval opponent (default: 800)')
    tr.add_argument('--csv', type=str, default=None,
                    help='training-curve CSV (one row per game; default: '
                         'next to the model, <model>.csv; fresh run '
                         'overwrites, resume appends)')
    tr.add_argument('--lr', type=float, default=1e-3,
                    help='Adam learning rate, linearly annealed to 1/5 by the '
                         'last game (default: 1e-3)')
    tr.add_argument('--seed', type=int, default=None,
                    help='random seed for reproducible runs (default: None)')
    tr.add_argument('--rollout-weight', type=float, default=ROLLOUT_WEIGHT,
                    help='random-playout mix in the leaf value (0 = pure network)')
    tr.add_argument('--channels', type=int, default=128,
                    help='conv channels of the residual tower (default: 128)')
    tr.add_argument('--blocks', type=int, default=5,
                    help='number of residual blocks (default: 5)')
    tr.add_argument('--ckpt-every', type=int, default=25,
                    help='save a checkpoint every N games (default: 25)')
    tr.add_argument('--dirichlet-eps', type=float, default=DIRICHLET_EPS,
                    help='root-prior exploration noise weight (0 disables)')
    tr.add_argument('--workers', type=int, default=1,
                    help='parallel self-play processes, each with its own GPU model copy (default: 1)')
    tr.add_argument('--replay-cap', type=int, default=500,
                    help='replay buffer cap (default: 500)')
    tr.add_argument('--weight-decay', type=float, default=1e-4,
                    help='Adam L2 weight decay (default: 1e-4)')
    tr.add_argument('--value-target', choices=('outcome', 'root_mean'),
                    default='root_mean',
                    help='value target: final outcome or MCTS root mean value')
    tr.add_argument('--policy-loss', choices=('ce', 'kl'), default='kl',
                    help='policy head loss: cross entropy or masked KL')
    tr.add_argument('--no-augment', action='store_true',
                    help='disable D4 flip augmentation')
    tr.add_argument('--stage', type=int, choices=(1, 2), default=2,
                    help='1 = pure-MCTS self-play data, 2 = neural self-play')
    tr.add_argument('--resume', type=str, default=None,
                    help='path to a checkpoint to continue training from')
    tr.add_argument('--model-path', type=str, default=None,
                    help='where to save the model (default: models/az_<game>.pt)')
    ev = sub.add_parser('eval',
                        help='play a trained model against Random (quick '
                             'sanity gauge; the in-training gauge plays MCTS)')
    ev.add_argument('--games', type=int, default=20,
                    help='games to play, doubled for both first players '
                         '(default: 20)')
    ev.add_argument('--sims', type=int, default=200,
                    help='AZ MCTS budget per move (default: 200)')
    ev.add_argument('--rollout-weight', type=float, default=ROLLOUT_WEIGHT,
                    help='random-playout mix in the leaf value (0 = pure network)')
    args = ap.parse_args(argv)
    ROLLOUT_WEIGHT = args.rollout_weight
    if args.cmd == 'train':
        train('ultimate', args.games, args.sims, args.eval_every,
              args.eval_games, args.lr, args.seed,
              channels=args.channels, blocks=args.blocks,
              ckpt_every=args.ckpt_every,
              dirichlet_eps=args.dirichlet_eps,
              workers=args.workers,
              replay_cap=args.replay_cap, weight_decay=args.weight_decay,
              value_target=args.value_target, policy_loss=args.policy_loss,
              augment=not args.no_augment, stage=args.stage,
              resume=args.resume, save_path=args.model_path,
              eval_ai=args.eval_ai, eval_sims=args.eval_sims,
              csv_path=args.csv)
    else:
        model = load_model('ultimate')
        if model is None:
            print('no trained model; run: python alphazero.py train')
            return 1
        wins = draws = losses = 0
        for _ in range(args.games):
            for az_first in (True, False):
                game = make_game('ultimate')
                while not game.is_over():
                    az_turn = game.current == X if az_first else game.current == O
                    if az_turn:
                        apply_move(game, alphazero_move(game, model, args.sims))
                    else:
                        apply_move(game, random.choice(game.legal_moves()))
                r = game.result()
                az_won = (r == X and az_first) or (r == O and not az_first)
                if az_won:
                    wins += 1
                elif r == 'D':
                    draws += 1
                else:
                    losses += 1
        print('AlphaZero(ultimate) vs Random over %d games: win=%d draw=%d loss=%d'
              % (args.games * 2, wins, draws, losses))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
