# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bridge to the Rust self-play exe (sba-core, D55 Phase A).

The exe owns search + inference and writes one safetensors record per game;
this module owns the Python side of that contract: locating the binary,
loading its records into the replay format, and (Task 8) running rounds
with a watchdog. The exe never talks to torch; Python never touches the tree.
"""

import glob
import os

import numpy as np
import torch


def load_records(dir_path):
    """Read every g*.safetensors in dir_path into replay-ready tuples.

    Returns [] for a missing/empty dir — callers treat that as "nothing
    landed", not an error (a crashed round may legitimately have written 0).
    """
    from safetensors.numpy import load_file

    games = []
    for p in sorted(glob.glob(os.path.join(dir_path, "g*.safetensors"))):
        d = load_file(p)
        states = np.ascontiguousarray(d["states"]).reshape(-1, 4, 9, 9)
        policy = d["policy"].reshape(-1, 81)
        z = d["z"].reshape(-1)
        games.append(
            (
                [torch.as_tensor(np.ascontiguousarray(s)) for s in states],
                [np.asarray(p, dtype=np.float32) for p in policy],
                [float(v) for v in z],
            )
        )
    return games
