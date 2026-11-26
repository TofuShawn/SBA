# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""parity_search — spec §6.1 layers 3-4, the local-GPU acceptance harness.

Layer 3 (search-stats parity): --positions seeded positions (random
openings + self-play midgames) searched by BOTH engines at --sims — the
Rust wave search (`sba_debug search --batch-size 256`, leaf-parallel
virtual-loss waves, dirichlet off) vs the Python batched PUCT
(`alphazero.mcts_search(batch_size=256)`, the same wave mode, dirichlet
off and rollout_weight=0.0 — the only configuration where both
implementations are asked the same question: same-shaped wave search over
pure network values). The wave port made this apples-to-apples: before it,
rust ran serial PUCT while python ran waves, and the mode gap did not
close with budget (diag-layer3-report). Bars: top-1 (argmax-of-counts,
first-max tie-break on legal order) agreement >= 95%, policy KL
(normalized visit shares, counts + 1e-8 smoothing — the self-play target
convention) <= 0.1 nats.

Layer 4 (strength parity): the Rust engine (`sba_debug search` per move,
argmax pick — process spawn per move is sanctioned for this one-time
acceptance) vs the Python AZ engine (`alphazero.alphazero_move`, its
shipped defaults incl. the 0.5 rollout blend), alternating colors over
--games matches. Both sides run the wave mode (batch_size 256, python's
default both sides). Both engines are deterministic at temp 0, so each
game opens from a seeded 0..4-ply random position — otherwise the match
would replay one identical game per color. Bar: Rust score (draws half)
within 40-60%.

Nothing here runs in CI — both layers need the local GPU + the sba_debug
exe (`python dev.py build`).

Usage:
  python src/tools/parity_search.py --weights models/az_ultimate.pt --layer 3
  python src/tools/parity_search.py --weights models/az_ultimate.pt --layer 4 --games 40
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))  # src/tools/
ROOT = os.path.dirname(os.path.dirname(HERE))  # the repo root
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):  # SRC for the sba packages under src/
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from az.alphazero import (  # noqa: E402
    DEVICE,
    AZNet,
    _root_value,
    alphazero_move,
    apply_move,
    export_safetensors,
    make_game,
    mcts_search,
)
from az.rustselfplay import find_exe  # noqa: E402

SPAWN_TIMEOUT = 600  # per-search budget; a spawn costs ~3.5s incl. GPU init


def _first_argmax(vec):
    """Index of the FIRST maximum (np.argmax convention, and the tie-break
    both engines use on legal order)."""
    best = 0
    for i in range(1, len(vec)):
        if vec[i] > vec[best]:
            best = i
    return best


def _load_model(weights):
    """The train() checkpoint layout ({state_dict, size, channels, blocks})
    -> (model, label)."""
    if not os.path.isfile(weights):
        raise SystemExit(f"weights not found: {weights}")
    import torch

    ckpt = torch.load(weights, map_location=DEVICE, weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"{weights}: not a _save_model checkpoint (state_dict/size/...)")
    size, channels, blocks = ckpt["size"], ckpt.get("channels", 128), ckpt.get("blocks", 5)
    model = AZNet(size, channels, blocks).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, f"{weights} (size={size} ch={channels} blk={blocks})"


def _rust_search(exe, weights, history, sims, batch_size=256):
    """One sba_debug search spawn -> ({move_code_str: visits}, root_value).
    batch_size mirrors python's mcts_search batch_size (default 256 = the
    wave mode, 1 = serial) so both engines run the same-shaped search.
    Any nonzero exit / bad JSON is a loud failure naming the position."""
    moves = ";".join(f"{m},{i}" for m, i in history)
    cmd = [
        exe,
        "search",
        "--weights",
        weights,
        "--moves",
        moves,
        "--sims",
        str(sims),
        "--batch-size",
        str(batch_size),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SPAWN_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"sba_debug search timed out (moves={moves!r})") from exc
    if proc.returncode != 0:
        raise SystemExit(
            f"sba_debug search failed rc={proc.returncode} (moves={moves!r}): "
            f"{proc.stderr[-300:]!r}"
        )
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise SystemExit(f"sba_debug search printed nothing (moves={moves!r})")
    try:
        ev = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"sba_debug search: bad JSON {lines[-1]!r}") from exc
    if not isinstance(ev, dict):
        raise SystemExit(f"sba_debug search: unexpected JSON {lines[-1]!r}")
    counts = {str(k): float(v) for k, v in ev.get("counts", {}).items()}
    return counts, float(ev.get("value", 0.0))


def _gen_positions(seed, n, model, gen_sims):
    """--positions move histories: half seeded random openings (depth
    cycling 0..8), half midgames from seeded self-play at the campaign
    recipe (dirichlet auto/0.25, temp 1.0) — every ply from ply 4 on is a
    candidate until the quota fills. Returns a list of [(m, i), ...]."""
    positions = []
    n_open = n // 2
    idx = 0
    while len(positions) < n_open:
        depth = idx % 9
        rng = random.Random(seed * 1009 + idx)
        g = make_game("ultimate")
        history = []
        for _ in range(depth):
            if g.is_over():
                break
            legal = g.legal_moves()
            mv = legal[rng.randrange(len(legal))]
            history.append(mv)
            apply_move(g, mv)
        if not g.is_over():
            positions.append(history)
        idx += 1
    np.random.seed(seed)
    random.seed(seed)
    g = make_game("ultimate")
    history = []
    guard = 0
    while len(positions) < n and guard < 5000:
        guard += 1
        if g.is_over():
            g = make_game("ultimate")
            history = []
            continue
        if len(history) >= 4:
            positions.append(history)
        if len(positions) >= n:
            break
        # the campaign's own self-play step: dirichlet-noised root, temp 1
        _root, counts = mcts_search(g, model, gen_sims, dirichlet_alpha="auto", dirichlet_eps=0.25)
        legal = g.legal_moves()
        nv = np.array([counts.get(m, 0.0) for m in legal], dtype=np.float64) + 1e-8
        p = nv / nv.sum()
        mv = legal[int(np.random.choice(len(legal), p=p))]
        history = history + [mv]
        apply_move(g, mv)
    if len(positions) < n:
        raise SystemExit(f"position generation stalled at {len(positions)}/{n}")
    return positions


def _layer3(exe, weights, model, positions, sims, seed):
    """Rust PUCT vs Python PUCT on the same positions: top-1 agreement +
    policy KL (both directions) + value delta."""
    agree = 0
    n_cmp = 0
    kl_rp, kl_pr, vdelta = [], [], []
    t_py = t_ru = 0.0
    for idx, history in enumerate(positions):
        g = make_game("ultimate")
        for mv in history:
            apply_move(g, mv)
        legal = g.legal_moves()
        if g.is_over() or not legal:
            continue
        t0 = time.perf_counter()
        root, counts = mcts_search(g, model, sims, rollout_weight=0.0, batch_size=256)
        t_py += time.perf_counter() - t0
        t0 = time.perf_counter()
        ru_counts, ru_value = _rust_search(exe, weights, history, sims, batch_size=256)
        t_ru += time.perf_counter() - t0
        py_vec = np.array([counts.get(m, 0.0) for m in legal], dtype=np.float64)
        ru_vec = np.array([ru_counts.get(str(m * 9 + i), 0.0) for m, i in legal], dtype=np.float64)
        agree += _first_argmax(py_vec) == _first_argmax(ru_vec)
        n_cmp += 1
        p = ru_vec + 1e-8  # self-play target smoothing, both sides alike
        q = py_vec + 1e-8
        p /= p.sum()
        q /= q.sum()
        kl_rp.append(float((p * np.log(p / q)).sum()))
        kl_pr.append(float((q * np.log(q / p)).sum()))
        vdelta.append(abs(ru_value - float(_root_value(root))))
        if (idx + 1) % 25 == 0:
            print(f"  [{idx + 1}/{len(positions)} searched]", flush=True)
    agree_pct = 100.0 * agree / n_cmp
    mean_rp = float(np.mean(kl_rp))
    mean_pr = float(np.mean(kl_pr))
    mean_vd = float(np.mean(vdelta))
    worst_kl = max(mean_rp, mean_pr)
    print()
    print(f"layer 3: search-stats parity  ({n_cmp} positions, seed={seed})")
    print(
        f"  settings: sims={sims} c_puct=1.5 dirichlet OFF both sides; "
        "wave mode batch_size=256 BOTH sides; py rollout_weight=0.0 "
        "(pure wave search + pure net values)"
    )
    print(
        f"  top-1 agreement: {agree}/{n_cmp} = {agree_pct:.2f}%  "
        f"(bar >= 95%) -> {'MEET' if agree_pct >= 95.0 else 'MISS'}"
    )
    print(
        f"  policy KL: rust||py mean {mean_rp:.4f} nats, py||rust mean {mean_pr:.4f} nats "
        f"(bar <= 0.1, larger direction) -> "
        f"{'MEET' if worst_kl <= 0.1 else 'MISS'}"
    )
    print(f"  mean |root value delta|: {mean_vd:.4f}")
    print(f"  elapsed: rust spawns {t_ru:.1f}s  python searches {t_py:.1f}s")
    return {"agree": agree, "n": n_cmp, "agree_pct": agree_pct, "kl": worst_kl}


def _layer4(exe, weights, model, games, sims, seed):
    """Rust engine vs the Python AZ engine, alternating colors, seeded
    random openings for diversity. Returns the rust score fraction."""
    wins = draws = losses = 0
    t_py = t_ru = 0.0
    for i in range(games):
        rust_is_x = i % 2 == 0
        rng = random.Random(seed * 7919 + i)
        k = i % 5  # opening plies 0..4: deterministic engines otherwise
        # replay one identical game per color
        g = make_game("ultimate")
        history = []
        for _ in range(k):
            if g.is_over():
                break
            legal = g.legal_moves()
            mv = legal[rng.randrange(len(legal))]
            # opening plies MUST be recorded: sba_debug replays --moves on a
            # fresh state, so an unrecorded opening would leave the exe
            # searching a different (earlier) position than the python game
            history.append(mv)
            apply_move(g, mv)
        while not g.is_over():
            legal = g.legal_moves()
            if not legal:
                break
            if (g.current == "X") == rust_is_x:
                t0 = time.perf_counter()
                ru_counts, _value = _rust_search(exe, weights, history, sims, batch_size=256)
                t_ru += time.perf_counter() - t0
                vec = [ru_counts.get(str(m * 9 + j), 0.0) for m, j in legal]
                mv = legal[_first_argmax(vec)]
            else:
                t0 = time.perf_counter()
                mv = alphazero_move(g, model, budget=sims)
                t_py += time.perf_counter() - t0
            history = history + [mv]
            apply_move(g, mv)
        result = g.result()
        if result == "D":
            draws += 1
            verdict = "D"
        elif result == ("X" if rust_is_x else "O"):
            wins += 1
            verdict = "rust W"
        else:
            losses += 1
            verdict = "rust L"
        print(
            f"  g{i:03d}/{games}  rust={'X' if rust_is_x else 'O'}  open={k} "
            f"plies={len(history)} result={result} -> {verdict}",
            flush=True,
        )
    score = (wins + 0.5 * draws) / games
    print()
    print(f"layer 4: strength head-to-head  ({games} games, seed={seed})")
    print(
        f"  settings: sims={sims} both sides; rust = sba_debug per move (argmax), "
        "python = alphazero_move (shipped defaults)"
    )
    print(f"  rust W/D/L: {wins} / {draws} / {losses}")
    verdict = "MEET" if 0.40 <= score <= 0.60 else "MISS"
    print(f"  rust score: {100.0 * score:.1f}%  (bar 40-60%) -> {verdict}")
    print(f"  elapsed: rust spawns {t_ru:.1f}s  python moves {t_py:.1f}s")
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/az_ultimate.pt")
    ap.add_argument("--layer", choices=["3", "4", "both"], default="both")
    ap.add_argument("--positions", type=int, default=200, help="layer 3 position count")
    ap.add_argument("--games", type=int, default=20, help="layer 4 game count")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--gen-sims", type=int, default=64, help="midgame generator self-play sims")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--exe", default=None, help="explicit sba_debug path (else probe)")
    a = ap.parse_args()

    model, label = _load_model(a.weights)
    exe = find_exe(a.exe, name="sba_debug")
    if not exe:
        raise SystemExit("sba_debug exe not found — build it first: python dev.py build")
    weights = os.path.join(tempfile.mkdtemp(prefix="parity-w-"), "w.safetensors")
    sha = export_safetensors(model, weights)

    print(f"parity_search — weights {label}, device {DEVICE}")
    print(f"exe: {exe}   weights sha256: {sha[:16]}…")

    results = {}
    if a.layer in ("3", "both"):
        print(f"\ngenerating {a.positions} positions (seed={a.seed})…", flush=True)
        positions = _gen_positions(a.seed, a.positions, model, a.gen_sims)
        depths = [len(h) for h in positions]
        print(
            f"  positions: {len(positions)}  ply range {min(depths)}..{max(depths)} "
            f"mean {sum(depths) / len(depths):.1f}"
        )
        results["layer3"] = _layer3(exe, weights, model, positions, a.sims, a.seed)
    if a.layer in ("4", "both"):
        results["layer4"] = _layer4(exe, weights, model, a.games, a.sims, a.seed)
    if len(results) > 1:
        print("\nsummary:")
    for name, res in results.items():
        if name == "layer3":
            print(
                f"  layer 3: agreement {res['agree_pct']:.2f}% / KL {res['kl']:.4f} nats "
                f"({res['agree']}/{res['n']})"
            )
        else:
            print(f"  layer 4: rust score {100.0 * res:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
