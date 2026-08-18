# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: MIT

"""AlphaZero-style neural-guided MCTS for (Ultimate) Tic Tac Toe.

Self-contained: works with any game object exposing the SBA.py interface
(NormalGame / UltimateGame): board/micro+macro, legal_moves(), is_over(),
result(), clone(), make_move(...), current.

Training:
    python alphazero.py train --game normal --games 400 --sims 80
    python alphazero.py train --game ultimate --games 300 --sims 80

Evaluation vs random:
    python alphazero.py eval --game normal --games 30 --sims 200

Models are saved to ./models/az_<game>.pt
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HERE = os.path.dirname(os.path.abspath(__file__))
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
    return torch.tensor(planes, dtype=torch.float32)


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

ROLLOUT_WEIGHT = 0.5  # leaf value = (1-w)*network + w*random playout


class SearchNode:
    """Node with plain-MCTS style incremental expansion and UCB exploration.

    Children are created one at a time (untried list) and unvisited children
    always win selection, which gives much stronger exploration than expanding
    every child at once -- important for Ultimate's wide branching factor.
    q is stored from the perspective of the player to move at this state.
    """

    __slots__ = ('state', 'move', 'children', 'visits', 'wins', 'q',
                 'untried', 'priors')

    def __init__(self, state, move):
        self.state = state
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.q = 0.0
        self.untried = state.legal_moves() if not state.is_over() else []
        random.shuffle(self.untried)
        self.priors = None  # {move: prior} filled once by the policy head

    def best_child(self, c_puct):
        log_n = math.log(max(1, self.visits))
        best, best_val = None, -math.inf
        for child in self.children:
            if child.visits == 0:
                uct = math.inf
            else:
                prior = self.priors.get(child.move, 1.0) if self.priors else 1.0
                uct = (-child.q + c_puct * prior * math.sqrt(log_n / child.visits))
            if uct > best_val:
                best, best_val = child, uct
        return best


def rollout_value(g):
    """Random playout from g; value from the perspective of g's current player."""
    player = g.current
    state = g.clone()
    while not state.is_over():
        legal = state.legal_moves()
        if not legal:
            break
        apply_move(state, random.choice(legal))
    r = state.result()
    if r == 'D':
        return 0.0
    return 1.0 if r == player else -1.0


@torch.no_grad()
def mcts_search(game, model, budget, c_puct=1.5, rollout_weight=None):
    """Run neural-guided MCTS; return (root SearchNode, {move: visit_count})."""
    if rollout_weight is None:
        rollout_weight = ROLLOUT_WEIGHT
    root_state = game.clone()
    root = SearchNode(root_state, None)
    for _ in range(budget):
        node = root
        state = root_state.clone()
        path = [root]
        while node.untried == [] and node.children:
            node = node.best_child(c_puct)
            apply_move(state, node.move)
            path.append(node)
        if node.untried:
            if node.priors is None:
                policy, _ = model(encode(node.state).unsqueeze(0).to(DEVICE))
                probs = F.softmax(policy[0], dim=0).cpu().numpy()
                node.priors = {m: float(probs[move_index(node.state, m)])
                               for m in node.untried}
            m = node.untried.pop()
            apply_move(state, m)
            child = SearchNode(state.clone(), m)
            node.children.append(child)
            node = child
            path.append(node)
        if state.is_over() or not state.legal_moves():
            value = terminal_value(state)
        else:
            policy, value = model(encode(state).unsqueeze(0).to(DEVICE))
            value = float(value[0, 0].item())
            if rollout_weight > 0:
                value = ((1.0 - rollout_weight) * value
                         + rollout_weight * rollout_value(state))
        for n in reversed(path):
            n.visits += 1
            n.wins += value
            n.q = n.wins / n.visits
            value = -value
    counts = {child.move: child.visits for child in root.children}
    return root, counts


def mcts_visit_counts(game, model, budget, c_puct=1.5):
    _, counts = mcts_search(game, model, budget, c_puct)
    return counts


def select_move(game, model, budget=800, temp=0.0, c_puct=1.5):
    root, counts = mcts_search(game, model, budget, c_puct)
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
    p /= p.sum()
    return moves[int(np.random.choice(len(moves), p=p))]


def alphazero_move(game, model, budget=800):
    return select_move(game, model, budget, temp=0.0)


# ---------------------------------------------------------------
# Training (self-play)
# ---------------------------------------------------------------

def model_path(game_type):
    return os.path.join(MODEL_DIR, 'az_%s.pt' % game_type)


def make_game(game_type):
    if game_type == 'normal':
        from game import NormalGame
        return NormalGame()
    from game import UltimateGame
    return UltimateGame()


def self_play_game(make_game_fn, model, sims, temp):
    game = make_game_fn()
    states, targets, players = [], [], []
    while not game.is_over():
        legal = game.legal_moves()
        counts = mcts_visit_counts(game, model, sims)
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
        idx = int(np.random.choice(len(legal), p=p))
        apply_move(game, legal[idx])
    result = game.result()
    zs = []
    for p in players:
        zs.append(0.0 if result == 'D' else (1.0 if p == result else -1.0))
    return states, targets, zs


def train(game_type, games=300, sims=80, eval_every=25, eval_games=20,
          lr=1e-3, seed=None, save=True, quiet=False,
          steps_per_game=5, replay_cap=100):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    size = 3 if game_type == 'normal' else 9
    model = AZNet(size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    make_game_fn = lambda: make_game(game_type)
    start = time.time()

    def evaluate():
        wins = draws = losses = 0
        for _ in range(eval_games):
            for az_first in (True, False):
                game = make_game_fn()
                while not game.is_over():
                    az_turn = game.current == X if az_first else game.current == O
                    if az_turn:
                        apply_move(game, alphazero_move(game, model, 100))
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
        return wins, draws, losses

    replay = []
    for it in range(1, games + 1):
        temp = max(0.3, 1.0 - it / games)
        states, targets, zs = self_play_game(make_game_fn, model, sims, temp)
        replay.append((states, targets, zs))
        if len(replay) > replay_cap:
            replay.pop(0)
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_game):
            g_states, g_targets, g_zs = random.choice(replay)
            n_pos = len(g_states)
            idx = np.random.choice(n_pos, min(32, n_pos), replace=False)
            batch = torch.stack([g_states[i] for i in idx]).to(DEVICE)
            target = torch.stack([torch.tensor(g_targets[i]) for i in idx])
            z = torch.tensor([g_zs[i] for i in idx], dtype=torch.float32)
            optimizer.zero_grad()
            pred_p, pred_v = model(batch)
            loss_p = -(F.log_softmax(pred_p, dim=1) * target).sum(dim=1).mean()
            loss_v = F.mse_loss(pred_v[:, 0], z)
            (loss_p + loss_v).backward()
            optimizer.step()
            total_loss += float(loss_p + loss_v)
        model.eval()
        if not quiet and (it % max(1, games // 10) == 0 or it == games):
            el = time.time() - start
            w, d, l = evaluate()
            print('[%s] game %d/%d  win=%d draw=%d loss=%d  loss=%.3f  (%.0fs)'
                  % (game_type, it, games, w, d, l, total_loss / steps_per_game, el), flush=True)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = model_path(game_type)
        torch.save({'state_dict': model.state_dict(), 'size': size}, path)
        if not quiet:
            print('saved', path)
    return model


def load_model(game_type):
    path = model_path(game_type)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location='cpu')
    model = AZNet(ckpt['size']).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description='AlphaZero for (Ultimate) Tic Tac Toe')
    sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--game', choices=['normal', 'ultimate'], default='normal')
    tr.add_argument('--games', type=int, default=300)
    tr.add_argument('--sims', type=int, default=80)
    tr.add_argument('--eval-every', type=int, default=25)
    tr.add_argument('--eval-games', type=int, default=20)
    tr.add_argument('--lr', type=float, default=1e-3)
    tr.add_argument('--seed', type=int, default=None)
    ev = sub.add_parser('eval')
    ev.add_argument('--game', choices=['normal', 'ultimate'], default='normal')
    ev.add_argument('--games', type=int, default=20)
    ev.add_argument('--sims', type=int, default=200)
    args = ap.parse_args(argv)
    if args.cmd == 'train':
        train(args.game, args.games, args.sims, args.eval_every,
              args.eval_games, args.lr, args.seed)
    else:
        model = load_model(args.game)
        if model is None:
            print('no trained model; run: python alphazero.py train --game %s' % args.game)
            return 1
        wins = draws = losses = 0
        for _ in range(args.games):
            for az_first in (True, False):
                game = make_game(args.game)
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
        print('AlphaZero(%s) vs Random over %d games: win=%d draw=%d loss=%d'
              % (args.game, args.games * 2, wins, draws, losses))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
