"""Task 9: train() --selfplay-engine wiring (rust rounds, retry-once, auto).

The fakes never spawn the real exe: run_round / find_exe / resolve_engine are
patched on the rustselfplay module (train() must consume them THROUGH that
module for the patches to bite), and the fake run_round writes real
safetensors records so load_records returns them.
"""

import csv as _csv
import os

import numpy as np
import pytest

import alphazero
import rustselfplay


def _write_records(out_dir, games, seed_base, sims):
    """Ruling 1 fake shape: ONE record file PER GAME (games files, 1 row
    each) — a round of N games yields N records, so train() absorbs exactly
    N games and fires exactly N `game` events."""
    from safetensors.numpy import save_file

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed_base + 1)
    for g in range(games):
        save_file(
            {
                "states": rng.standard_normal((1, 4, 9, 9), dtype=np.float32),
                "policy": rng.random((1, 81), dtype=np.float32),
                "z": rng.uniform(-1, 1, (1,)).astype(np.float32),
            },
            os.path.join(out_dir, "g%05d.safetensors" % g),
            metadata={
                "format_version": "1",
                "game_type": "ultimate",
                "seed": str(seed_base + g),
                "result": "D",
                "sims": str(sims),
                "weights_sha": "x",
            },
        )


@pytest.fixture
def fake_rust(monkeypatch, tmp_path):
    """Patch the bridge so train() sees a rust engine without any exe."""
    calls = []

    def fake_run_round(args, **kw):
        calls.append(
            {
                "exe": kw["exe"],
                "games": args.games,
                "seed_base": args.seed_base,
                "out_dir": args.out_dir,
            }
        )
        _write_records(args.out_dir, args.games, args.seed_base, args.sims)
        res = {"games_done": args.games, "crashed": False, "stderr_tail": ""}
        # the real bridge announces the round once at the end (Task 8 contract)
        if kw.get("on_event"):
            kw["on_event"](
                {"type": "selfplay_round", "games_done": res["games_done"], "crashed": False}
            )
        return res

    monkeypatch.setattr(rustselfplay, "run_round", fake_run_round)
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: "/fake/exe")
    monkeypatch.setattr(rustselfplay, "resolve_engine", lambda arg, exe: ("rust", "/fake/exe"))
    # round artifacts under the test's tmp_path (the repo logs/ tree stays clean)
    monkeypatch.setattr(alphazero, "MODEL_DIR", str(tmp_path / "models"))
    return calls


def test_train_rust_engine_runs_and_emits(fake_rust, tmp_path):
    events = []
    model = alphazero.train(
        alphazero.TrainCfg(
            games=4,
            sims=2,
            eval_every=100,
            eval_games=1,
            eval_sims=32,
            lr=1e-4,
            steps_per_game=1,
            channels=8,
            blocks=1,
            save=False,
            quiet=True,
            replay_cap=10,
            csv_path=str(tmp_path / "curve.csv"),
            selfplay_engine="rust",
            round_games=2,
        ),
        hooks=alphazero.TrainHooks(on_event=events.append),
    )
    assert model is not None
    # two rounds of 2 games; the seed stream continues across rounds (spec
    # §3.4: seed = --seed-base + absolute game index — round 2 must NOT
    # replay round 1's stream; with seed=None, round 1 starts at 0 and round
    # 2 at round_games)
    assert [c["games"] for c in fake_rust] == [2, 2]
    assert [c["seed_base"] for c in fake_rust] == [0, 2]
    types = [e["type"] for e in events]
    assert types.count("game") == 4  # one optimize step + one event per game
    assert [e["game"] for e in events if e["type"] == "game"] == [1, 2, 3, 4]
    assert types.count("selfplay_round") == 2
    # engine arbitration is announced once, with the TUI's workers value
    assert [e for e in events if e["type"] == "selfplay_engine"] == [
        {"type": "selfplay_engine", "engine": "rust", "workers": 1}
    ]
    # CSV: one row per game (D41 contract holds through the rust path)
    with open(tmp_path / "curve.csv") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 4


def test_train_rust_retry_after_crash(fake_rust, monkeypatch):
    state = {"n": 0}
    calls = fake_rust

    def flaky_run_round(args, **kw):
        state["n"] += 1
        calls.append(
            {
                "exe": kw["exe"],
                "games": args.games,
                "seed_base": args.seed_base,
                "out_dir": args.out_dir,
            }
        )
        if state["n"] == 1:
            # game 1 completed (its record landed) before the exe died
            _write_records(args.out_dir, 1, args.seed_base, args.sims)
            res = {"games_done": 1, "crashed": True, "stderr_tail": "boom"}
        else:
            _write_records(args.out_dir, args.games, args.seed_base, args.sims)
            res = {"games_done": args.games, "crashed": False, "stderr_tail": ""}
        # the real bridge announces the round once at the end (Task 8 contract)
        if kw.get("on_event"):
            kw["on_event"](
                {
                    "type": "selfplay_round",
                    "games_done": res["games_done"],
                    "crashed": res["crashed"],
                }
            )
        return res

    monkeypatch.setattr(rustselfplay, "run_round", flaky_run_round)
    events = []
    alphazero.train(
        alphazero.TrainCfg(
            games=2,
            sims=2,
            eval_every=100,
            eval_games=1,
            eval_sims=32,
            lr=1e-4,
            steps_per_game=1,
            channels=8,
            blocks=1,
            save=False,
            quiet=True,
            replay_cap=10,
            selfplay_engine="rust",
            round_games=2,
        ),
        hooks=alphazero.TrainHooks(on_event=events.append),
    )
    assert state["n"] == 2  # one retry, exactly
    # retry spawn (ruling 4): the round's MISSING games only, seed advanced
    # by the completed count, records landing in a fresh -retry dir
    assert calls[0]["games"] == 2 and calls[0]["seed_base"] == 0
    assert calls[1]["games"] == 1
    assert calls[1]["seed_base"] == 1
    assert calls[1]["out_dir"] == calls[0]["out_dir"] + "-retry"
    types = [e["type"] for e in events]
    assert types.count("worker_lost") == 1
    assert [e["game"] for e in events if e["type"] == "worker_lost"] == [1]
    # the worker_lost sits between the crashed round and the retry
    rounds = [i for i, t in enumerate(types) if t == "selfplay_round"]
    lost = types.index("worker_lost")
    assert rounds[0] < lost < rounds[1]
    # records came from BOTH dirs and were absorbed in order
    assert [e["game"] for e in events if e["type"] == "game"] == [1, 2]


def test_train_rust_second_crash_raises(fake_rust, monkeypatch):
    def always_crash(args, **kw):
        return {"games_done": 0, "crashed": True, "stderr_tail": "panic: nope"}

    monkeypatch.setattr(rustselfplay, "run_round", always_crash)
    # the exe's stderr tail rides along in the message (spec §5.4: "then raise")
    with pytest.raises(RuntimeError, match=r"died twice.*panic: nope"):
        alphazero.train(
            alphazero.TrainCfg(
                games=2,
                sims=2,
                eval_every=100,
                eval_games=1,
                eval_sims=32,
                lr=1e-4,
                steps_per_game=1,
                channels=8,
                blocks=1,
                save=False,
                quiet=True,
                selfplay_engine="rust",
                round_games=2,
            ),
        )


def test_train_explicit_rust_stage1_raises(fake_rust):
    # explicit --selfplay-engine rust + stage 1 is a caller error: the exe
    # always plays network self-play with root-mean z (no stage-1 rollouts,
    # no alternative value target), so the request is rejected up front
    with pytest.raises(ValueError, match=r"--selfplay-engine rust.*--stage 1.*--value-target"):
        alphazero.train(
            alphazero.TrainCfg(
                games=2,
                sims=2,
                eval_every=100,
                eval_games=1,
                eval_sims=32,
                lr=1e-4,
                steps_per_game=1,
                channels=8,
                blocks=1,
                save=False,
                quiet=True,
                stage=1,
                selfplay_engine="rust",
                round_games=2,
            )
        )
    assert fake_rust == []  # no round ever dispatched to the exe


def test_train_auto_stage1_downgrades_to_python(fake_rust):
    # rust resolved via auto + stage 1: silently falls back to the python
    # paths (which do support stage 1), announced with a reason event — the
    # rust announcement itself still fired first
    events = []
    model = alphazero.train(
        alphazero.TrainCfg(
            games=2,
            sims=2,
            eval_every=100,
            eval_games=1,
            eval_sims=32,
            lr=1e-4,
            steps_per_game=1,
            channels=8,
            blocks=1,
            save=False,
            quiet=True,
            replay_cap=10,
            stage=1,
            selfplay_engine="auto",
            round_games=2,
        ),
        hooks=alphazero.TrainHooks(on_event=events.append),
    )
    assert model is not None
    assert fake_rust == []  # the round loop never dispatched to the exe
    engines = [e for e in events if e["type"] == "selfplay_engine"]
    assert engines[0] == {"type": "selfplay_engine", "engine": "rust", "workers": 1}
    assert engines[1] == {
        "type": "selfplay_engine",
        "engine": "python",
        "reason": "stage/value_target unsupported by rust path",
    }
    assert [e["game"] for e in events if e["type"] == "game"] == [1, 2]  # python path ran


def test_auto_downgrades_to_python_without_exe(monkeypatch):
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: None)
    assert rustselfplay.resolve_engine("auto", None) == ("python", None)


def test_main_parses_selfplay_flags():
    # alphazero.main argv parse accepts the new flags (argparse SystemExit=2
    # on garbage is the negative case)
    with pytest.raises(SystemExit):
        alphazero.main(["train", "--selfplay-engine", "bogus"])
