# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""dev.py — every development command in one place.

The toolchain spans Windows Python, a WSL mingw cross-compiler and WSL
cargo; each of those has exactly ONE invocation shape that works from this
machine (wsl.exe joins arguments, MSYS rewrites captured POSIX paths, ruff
lives only in .venv). The recipes below encode the shapes that work — run:

    python dev.py              # list tasks
    python dev.py <task> [...] # run one (extra args forwarded)

All subprocess calls are argv-list form (shell=False); nothing is ever
interpolated into a shell string.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
WSL_ROOT = "/mnt/e/Project/SBA"  # repo root inside WSL
TARGET = "x86_64-pc-windows-gnu"


def wsl_cargo(args):
    """Cargo inside WSL, invoked by its absolute login-PATH location as a
    pure argv call (see src/tools/build.py for why bash -lc forwarding dies)."""
    cargo = subprocess.run(
        ["wsl", "bash", "-lc", "command -v cargo"], capture_output=True, text=True
    ).stdout.strip()
    if not cargo:
        raise SystemExit("cargo not found in WSL — install rustup there")
    cmd = ["wsl", cargo, *args]
    print("+", cmd[1], *cmd[2:])
    return subprocess.run(cmd, cwd=ROOT).returncode


def crate_task(crate, cargo_args):
    """Workspace-aware (D60): cargo runs at the ROOT workspace manifest and
    the crate is selected with `-p <crate>` — the shared lockfile and the
    root target/ apply. `-p` must sit BEFORE the `--` passthrough separator
    (args after it go to rustc, not cargo)."""
    if "--" in cargo_args:
        dash = cargo_args.index("--")
        pre, post = cargo_args[:dash], cargo_args[dash:]
        return wsl_cargo([*pre, "-p", crate, *post])
    return wsl_cargo([*cargo_args, "-p", crate])


def wsl_gcc_check():
    """The CI C-warnings gate, mirrored locally: -Werror on all five C files."""
    files = ["sba_ultt.c", "sba_rollout.c", "sba_mcts.c", "sba_rave.c", "sba_minimax.c"]
    for f in files:
        r = subprocess.run(
            [
                "wsl",
                "gcc",
                "-std=c11",
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-D__stdcall=",
                "-Isrc/engines",
                f"{WSL_ROOT}/src/engines/{f}",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            print(f"{f}: FAIL\n{r.stderr[-600:]}")
            return 1
    print("C gate: green (5 files, -Wall -Wextra -Werror)")
    return 0


def _rust_tests(_args):
    rc = crate_task("sba_game", ["test"])
    if rc:
        return rc
    # the cpu backend needs no GPU, so the whole suite runs in WSL
    return crate_task("sba-core", ["test", "--no-default-features", "--features", "cpu"])


def venv(args):
    cmd = [VENV_PY, *args]
    print("+", " ".join(os.path.relpath(p, ROOT) if os.path.isabs(p) else p for p in cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def task_build(_args):
    """All natives: sba_engine.dll (C) + sba_game.dll (Rust state layer),
    plus the sba-core bench binary (Phase A has no DLL yet)."""
    rc = venv(["src/tools/build.py"])
    if rc:
        return rc
    return crate_task("sba-core", ["build", "--release", "--target", TARGET])


def task_complexity(_args):
    """Complexity REPORT, warn-only until the GamePage split (H2): findings
    are printed (rcs ignored, exit 0 always) like the old pip-audit report
    pattern. Two halves — ruff C901 on src/ui (max-complexity 10, see
    pyproject [tool.ruff.lint.mccabe]) and src/tools/func_len.py's AST scan
    for functions over 50 lines (def line to block end) across src/."""
    venv(["-m", "ruff", "check", "--select", "C901", "src/ui/"])
    venv(["src/tools/func_len.py", "src"])
    return 0


TASKS = {
    "build": ("compile every native (C DLL + Rust cdylib + sba-core bench)", task_build),
    "test": (
        "pytest suite (args forwarded: dev.py test tests/test_ai.py -q)",
        lambda a: venv(["-m", "pytest", "tests", "-q", *a]),
    ),
    "lint": ("ruff check", lambda a: venv(["-m", "ruff", "check", "."])),
    "mypy": (
        "mypy on the two FFI boundary modules (--check-untyped-defs; scope in pyproject)",
        lambda a: venv(
            [
                "-m",
                "mypy",
                "--check-untyped-defs",
                "src/engines/cengine.py",
                "src/az/rustbridge.py",
                *a,
            ]
        ),
    ),
    "complexity": (
        "C901 + >50-line function report (src/ui C901, src/ length; warn-only, exit 0)",
        task_complexity,
    ),
    "format": (
        "ruff format --check (same gate as CI)",
        lambda a: venv(["-m", "ruff", "format", "--check", "."]),
    ),
    "format-write": ("apply ruff formatting", lambda a: venv(["-m", "ruff", "format", "."])),
    "fmt": (
        "cargo fmt --all --check (the CI gate; format-write-safe, no crate to pick)",
        lambda a: wsl_cargo(["fmt", "--all", "--", "--check", *a]),
    ),
    "fmt-write": ("apply rustfmt across the workspace", lambda a: wsl_cargo(["fmt", "--all", *a])),
    "clippy": (
        "cargo clippy -D warnings on sba-game",
        lambda a: crate_task(
            "sba_game", ["clippy", "--release", "--target", TARGET, "--", "-D", "warnings"]
        ),
    ),
    "clippy-core": (
        "cargo clippy -D warnings on sba-core (CI Linux gate, default wgpu)",
        lambda a: crate_task("sba-core", ["clippy", "--release", "--", "-D", "warnings"]),
    ),
    "cgate": ("C warnings gate (-Wall -Wextra -Werror, mirrors CI)", lambda a: wsl_gcc_check()),
    "bench-selfplay": (
        "self-play A/B: mp.Pool path vs rust exe (pos/s, speedup)",
        lambda a: venv(["src/tools/bench_selfplay.py", *a]),
    ),
    "coverage": (
        "pytest with coverage; the floor lives in pyproject [tool.coverage.report]",
        lambda a: venv(["-m", "pytest", "tests", "-q", "--cov", *a]),
    ),
    "self-test": (
        "headless suite through the real entry point",
        lambda a: venv(["sba.py", "--self-test"]),
    ),
    "test-rust": (
        "cargo test: sba-game (default) + sba-core (cpu backend, no GPU)",
        _rust_tests,
    ),
}


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("tasks:")
        for name, (desc, _) in TASKS.items():
            print(f"  {name:<13} {desc}")
        return 0
    name, *rest = argv
    if name not in TASKS:
        print(f"unknown task: {name} — run `python dev.py` for the list")
        return 2
    return TASKS[name][1](rest)


if __name__ == "__main__":
    sys.exit(main())
