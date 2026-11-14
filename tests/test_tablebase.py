# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Persisted Normal tablebase (data/tt_normal.bin): the committed file
matches a fresh minimax solve, the loader validates and falls back, and
solver_move stays optimal when backed by the file."""

import hashlib
import os
import random

import pytest

import ai
from ai import (_TABLEBASE_PATH, _board_key, _build_tablebase_dfs,
                _read_tablebase_file, build_tablebase, solver_move,
                write_tablebase_file)
from game import NormalGame, apply_move

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def dfs_table():
    return _build_tablebase_dfs()


def _child_board(g, m):
    board = g.board[:]
    board[m] = g.current
    return board


def test_committed_file_matches_a_fresh_solve(dfs_table):
    raw = open(_TABLEBASE_PATH, 'rb').read()
    assert len(raw) == 3 ** 9
    table = _read_tablebase_file()
    assert table == dfs_table            # same keys, same values
    filled = {k for k in range(3 ** 9) if raw[k] != 0xFF}
    assert filled == set(dfs_table)      # sentinel exactly on unreachable slots
    assert table.get(0) == 0             # empty board: theoretical draw
    assert len(table) == 5478            # literature figure for legal 3x3 states


def test_loader_falls_back_to_dfs_without_file(dfs_table, monkeypatch, tmp_path):
    monkeypatch.setattr(ai, '_TABLEBASE', None)
    monkeypatch.setattr(ai, '_TABLEBASE_PATH', str(tmp_path / 'absent.bin'))
    assert build_tablebase() == dfs_table


def test_corrupt_or_truncated_file_is_rejected(dfs_table, monkeypatch, tmp_path):
    raw = bytearray(open(_TABLEBASE_PATH, 'rb').read())

    bad = tmp_path / 'bad_byte.bin'
    raw[40] = 0x7F                       # neither a value code nor the sentinel
    bad.write_bytes(bytes(raw))
    monkeypatch.setattr(ai, '_TABLEBASE', None)
    monkeypatch.setattr(ai, '_TABLEBASE_PATH', str(bad))
    assert _read_tablebase_file() is None
    assert build_tablebase() == dfs_table

    trunc = tmp_path / 'truncated.bin'
    trunc.write_bytes(bytes(raw[:100]))
    monkeypatch.setattr(ai, '_TABLEBASE_PATH', str(trunc))
    assert _read_tablebase_file() is None


def test_write_is_idempotent_and_matches_committed(tmp_path):
    p1, p2 = tmp_path / 'a.bin', tmp_path / 'b.bin'
    write_tablebase_file(str(p1))
    write_tablebase_file(str(p2))
    b1 = p1.read_bytes()
    assert b1 == p2.read_bytes()
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(
        open(_TABLEBASE_PATH, 'rb').read()).hexdigest()


def test_solver_move_is_optimal_on_random_positions(dfs_table):
    """Mover value = max over children of (-child value); whichever move
    solver_move picks must realise it (draws included, value 0)."""
    rng = random.Random(20261114)
    table = _read_tablebase_file()
    checked = 0
    for _ in range(200):
        g = NormalGame()
        for _ in range(rng.randint(0, 8)):
            if g.is_over():
                break
            apply_move(g, rng.choice(g.legal_moves()))
        if g.is_over():
            continue
        lm = g.legal_moves()
        best = max(-table[_board_key(_child_board(g, m))] for m in lm)
        mv = solver_move(g.clone())
        assert -table[_board_key(_child_board(g, mv))] == best, g.board
        checked += 1
    assert checked > 50
