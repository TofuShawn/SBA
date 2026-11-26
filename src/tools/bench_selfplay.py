# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""bench_selfplay — the sba-core Phase A acceptance A/B (spec §6.2).

Measures END-TO-END self-play throughput for both engines, same machine /
weights / settings (games, sims, threads, temp):

  python — the old path: an mp.Pool (spawn) of alphazero.self_play_game
           workers, one GPU model copy per worker. The trainer's
           `--selfplay-engine python` shape: every game dispatched
           concurrently (apply_async, all games in flight — a blocking
           per-game apply would serialize onto one worker and flatter the
           rust side), the trainer's own worker init (console redirect).
  rust   — the sba_selfplay exe: root-parallel threads + batched burn
           inference (Phase A), driven through rustselfplay.run_round.

The rust/python pos/s ratio is the Phase A acceptance bar (>= 4x at the
campaign settings — sims 800, threads 8). Both sides run at temp 1.0 fixed
(the campaign's start temperature; run_round gets temp_start == temp_end so
no side plays an easier schedule).

After the timed rust round, one untimed probe round (its first 8 games —
batch_stats fires every 8) re-runs the same sims/threads and reports the
exe's own batch_stats events (avg_batch_fill — the batching-health metric
the spec asks recorded; run_round consumes those events internally, so the
probe parses stdout directly).

Usage:  python src/tools/bench_selfplay.py [--weights models/az_ultimate.pt]
            [--games 16] [--sims 800] [--threads 8] [--engine both|rust|python]
"""

import argparse
import json
import multiprocessing as mp
import os
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

import torch  # noqa: E402

from az.alphazero import DEVICE, AZNet, _mp_worker_init, export_safetensors  # noqa: E402
from config import TEMP_MIN  # noqa: E402
from az.rustselfplay import RoundArgs, build_round_cmd, find_exe, load_records, run_round  # noqa: E402

TEMP = 1.0  # campaign start temperature; both sides play all games at it


def _one_game(state_dict, size, channels, blocks, sims, temp):
    """One python self-play game, the old path's worker shape.

    Spawn pickles the state_dict, not the module — the model is rebuilt on
    the worker side (same as alphazero._mp_play_game), the game comes from
    alphazero.make_game so the worker rides the same rust state layer the
    trainer uses. Imports stay inside the function: the child re-imports
    this module on spawn and torch loads there anyway.
    """
    from az.alphazero import AZNet, DEVICE, make_game, self_play_game

    model = AZNet(size, channels, blocks).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    states, targets, zs = self_play_game(lambda: make_game("ultimate"), model, sims, temp)
    return states, targets, zs


def bench_python(state_dict, model_meta, games, sims, threads):
    """The old mp.Pool path, end to end: pool spin-up + all games.

    Returns (positions, elapsed_seconds). Any worker exception is
    re-raised with its game index — a partial round is a failed bench, not
    a quiet one.
    """
    size, channels, blocks = model_meta
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    positions = 0
    # the trainer's own worker init: per-worker console redirect to logs/
    # (8 spawn children otherwise tear the console apart with MIOpen noise)
    with ctx.Pool(threads, initializer=_mp_worker_init, initargs=(None,)) as pool:
        asyncs = [
            pool.apply_async(_one_game, (state_dict, size, channels, blocks, sims, TEMP))
            for _ in range(games)
        ]
        for idx, res in enumerate(asyncs):
            try:
                states, _targets, _zs = res.get()
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise SystemExit(f"python worker game {idx} failed: {exc!r}") from exc
            positions += len(states)
    return positions, time.perf_counter() - t0


def bench_rust(model, games, sims, threads, procs=1):
    """The sba_selfplay exe path, end to end (weights export excluded —
    export_safetensors runs before the timer starts). `procs` splits the
    round across K concurrent exe processes (stream-overlap attempt —
    run_round partitions seeds/records/threads and merges the watchdog).
    Returns (positions, elapsed, sha, exe_path, weights_path).
    """
    exe = find_exe()
    if not exe:
        raise SystemExit("sba_selfplay exe not found — build it first: python dev.py build")
    weights = os.path.join(tempfile.mkdtemp(prefix="bench-w-"), "w.safetensors")
    out_dir = os.path.join(tempfile.mkdtemp(prefix="bench-sp-"), "games")
    sha = export_safetensors(model, weights)
    t0 = time.perf_counter()
    res = run_round(
        RoundArgs(
            weights_path=weights,
            out_dir=out_dir,
            games=games,
            sims=sims,
            threads=threads,
            seed_base=0,
            temp_start=TEMP,
            temp_end=TEMP,
        ),
        exe=exe,
        procs=procs,
    )
    elapsed = time.perf_counter() - t0
    if res["crashed"]:
        raise SystemExit(
            f"rust round crashed: games_done={res['games_done']} stderr_tail={res['stderr_tail']!r}"
        )
    records = load_records(out_dir)
    if not records:
        raise SystemExit("rust round produced no records — nothing to count")
    positions = sum(len(s) for s, _t, _z in records)
    return positions, elapsed, sha, exe, weights


def probe_batch_stats(exe, weights, games, sims, threads):
    """One untimed re-run of the same settings, stdout parsed directly for
    the exe's batch_stats events (every 8 games: batches, avg_batch_fill,
    pos_per_sec). Returns the event list; a probe failure warns instead of
    failing the bench — the timed round already proved the exe works.
    """
    out_dir = os.path.join(tempfile.mkdtemp(prefix="bench-fill-"), "games")
    # the argv comes from the bridge's own builder (one copy of the flag
    # contract, no mirror to drift) — run_round consumes the non-game_done
    # events internally instead of forwarding them, so the probe spawns and
    # parses stdout itself
    rargs = RoundArgs(
        weights_path=weights,
        out_dir=out_dir,
        games=games,
        sims=sims,
        threads=threads,
        seed_base=0,
        temp_start=TEMP,
        temp_end=TEMP,
        temp_min=TEMP_MIN,
    )
    cmd = [exe, *build_round_cmd(rargs)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=games * 120 + 600)
    except subprocess.TimeoutExpired:
        print("WARNING: batch_stats probe timed out — fill not recorded")
        return []
    if proc.returncode != 0:
        print(f"WARNING: batch_stats probe rc={proc.returncode}: {proc.stderr[-300:]!r}")
        return []
    events = []
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "batch_stats":
            events.append(ev)
    return events


def _load_model(weights):
    """Load the checkpoint the way train() does ({state_dict, size,
    channels, blocks}) — returns (model, meta, label)."""
    if not os.path.isfile(weights):
        raise SystemExit(f"weights not found: {weights}")
    ckpt = torch.load(weights, map_location=DEVICE, weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"{weights}: not a _save_model checkpoint (state_dict/size/...)")
    size, channels, blocks = ckpt["size"], ckpt.get("channels", 128), ckpt.get("blocks", 5)
    model = AZNet(size, channels, blocks).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    label = f"{weights} (size={size} ch={channels} blk={blocks})"
    return model, (size, channels, blocks), label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/az_ultimate.pt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--engine", choices=["both", "rust", "python"], default="both")
    ap.add_argument(
        "--procs",
        type=int,
        default=1,
        help="exe processes for the rust round (disjoint seed ranges, "
        "merged round; round game threads split across them)",
    )
    a = ap.parse_args()
    model, model_meta, label = _load_model(a.weights)

    print(f"self-play A/B — weights {label}, device {DEVICE}")
    print(
        f"settings: games={a.games} sims={a.sims} threads={a.threads} procs={a.procs} temp={TEMP}"
    )

    rows = []  # (name, positions, elapsed)
    sha = exe = weights = None
    if a.engine in ("both", "python"):
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        positions, elapsed = bench_python(state_dict, model_meta, a.games, a.sims, a.threads)
        rows.append(("python mp.Pool", positions, elapsed))
    if a.engine in ("both", "rust"):
        positions, elapsed, sha, exe, weights = bench_rust(
            model, a.games, a.sims, a.threads, a.procs
        )
        rows.append((f"rust sba_selfplay x{a.procs}", positions, elapsed))

    print()
    rates = [(name, pos / el, el, pos) for name, pos, el in rows]
    base = rates[0][1] if rates else 1.0
    for name, rate, el, pos in rates:
        line = f"{name:18s} {pos:5d} pos  {el:8.2f}s  {rate:8.1f} pos/s"
        if len(rows) > 1:
            line += f"  ({rate / base:5.2f}x pos/s)"
        print(line)
    if len(rows) > 1:
        speedup = rates[1][1] / rates[0][1]
        probe_games = min(a.games, 8)  # batch_stats fires every 8 games: one
        # event is enough for the fill metric — don't double the gate cost
        verdict = "MEET" if speedup >= 4.0 else "MISS"
        print(f"throughput speedup: {speedup:.2f}x  (Phase A bar >= 4x) -> {verdict}")
        print(f"weights sha256: {sha}")
        # the probe replays ONE process's shape: at K procs each owns
        # threads//K game threads (run_round's split), so probe at that
        # thread count for a representative batch fill
        events = probe_batch_stats(exe, weights, probe_games, a.sims, max(1, a.threads // a.procs))
        if events:
            detail = ", ".join(
                f"batches={ev.get('batches')} fill={ev.get('avg_batch_fill'):.3f} "
                f"pps={ev.get('pos_per_sec', 0.0):.1f}"
                for ev in events
            )
            fills = [ev.get("avg_batch_fill", 0.0) for ev in events]
            print(f"batch_stats (probe, {probe_games} of {a.games} games, untimed): {detail}")
            print(f"mean avg_batch_fill: {sum(fills) / len(fills):.3f} over {len(events)} events")
        else:
            print("batch_stats: no events captured (see warnings above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
