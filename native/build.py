# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the SBA native engines: native/sba_engine.dll (D27) and — when a
Rust toolchain exists — native/sba_game.dll, the D48 state-layer cdylib.

C modules: sba_ultt.c (shared rules) + sba_rollout.c (plain/pattern) +
sba_mcts.c (full tree search) + sba_minimax.c (normal solve / ultimate
alpha-beta / flat MCTS). One DLL exports everything.

Compiler search order:
1. a Windows gcc/clang on PATH (native build)
2. WSL's mingw-w64 cross compiler (x86_64-w64-mingw32-gcc), if WSL exists —
   installs via `pacman -S mingw-w64-gcc` on an Arch WSL

Rust (sba-game crate): cargo on PATH builds the host target; otherwise WSL's
cargo builds the x86_64-pc-windows-gnu target when its std is installed
(`rustup target add x86_64-pc-windows-gnu`). No cargo -> skipped with a
note; the Python game stays the fallback.

Usage:  python native/build.py
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sba_engine.dll")
SOURCES = ["sba_ultt.c", "sba_rollout.c", "sba_mcts.c", "sba_rave.c", "sba_minimax.c"]

RUST_DIR = os.path.join(HERE, "sba-game")
RUST_MANIFEST = os.path.join(RUST_DIR, "Cargo.toml")
# cargo names the cdylib sba_game.dll (Windows) / libsba_game.so (Linux);
# both ship as native/sba_game.dll|so next to sba_engine.dll
RUST_PRODUCED = "sba_game.dll" if os.name == "nt" else "libsba_game.so"
RUST_LIB = "sba_game.dll" if os.name == "nt" else "sba_game.so"


def build_rust():
    """cargo build the sba_game cdylib (D48). Best-effort: without cargo
    the step is skipped and the Python game remains the fallback."""
    if not os.path.isfile(RUST_MANIFEST):
        return 0
    cargo = shutil.which("cargo")
    if cargo:
        cmd = [cargo, "build", "--release", "--manifest-path", RUST_MANIFEST]
        produced = os.path.join(RUST_DIR, "target", "release", RUST_PRODUCED)
    else:
        # WSL route: only worth it for a Windows dll, which needs the
        # x86_64-pc-windows-gnu std installed in the WSL rustup
        try:
            have = subprocess.run(
                ["wsl", "bash", "-lc", "command -v cargo"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            have = None
        if not (have and have.returncode == 0 and have.stdout.strip()):
            print(
                "cargo not found — sba_game (Rust state layer) skipped; "
                "the Python game stays the fallback"
            )
            return 0
        check = subprocess.run(
            [
                "wsl",
                "bash",
                "-lc",
                "rustup target list --installed | grep -q x86_64-pc-windows-gnu",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check.returncode != 0:
            print(
                "WSL cargo lacks the x86_64-pc-windows-gnu std — sba_game "
                "skipped (`rustup target add x86_64-pc-windows-gnu` in WSL "
                "enables it)"
            )
            return 0
        # wsl.exe JOINS its arguments before exec, which destroys `bash -lc
        # script arg` positional passing — invoke cargo by its absolute
        # login-PATH location instead: pure argv, no shell string, nothing
        # to splice (Mimosa: command injection)
        cargo_path = have.stdout.strip()
        cmd = [
            "wsl",
            cargo_path,
            "build",
            "--release",
            "--target",
            "x86_64-pc-windows-gnu",
            "--manifest-path",
            wsl_path(RUST_MANIFEST),
        ]
        produced = os.path.join(
            RUST_DIR, "target", "x86_64-pc-windows-gnu", "release", RUST_PRODUCED
        )
    print("building rust:", " ".join(cmd) if cmd[0] != "wsl" else cmd[1])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    out = os.path.join(HERE, RUST_LIB)
    shutil.copyfile(produced, out)
    print(f"built: {out} ({os.path.getsize(out)} bytes)")
    return 0


def find_compiler():
    for exe in ("gcc", "clang", "cc"):
        path = shutil.which(exe)
        if path:
            return [path]
    try:
        r = subprocess.run(
            ["wsl", "x86_64-w64-mingw32-gcc", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0:
            return ["wsl", "x86_64-w64-mingw32-gcc"]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def wsl_path(win_path):
    """Translate a Windows path into the WSL /mnt/<drive> form."""
    drive, rest = os.path.splitdrive(os.path.abspath(win_path))
    if not drive:
        # os.path.splitdrive("/home/x") == ("", "/home/x") on POSIX
        raise RuntimeError("wsl_path() needs a Windows drive path, got %r" % win_path)
    return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")


def main():
    for src in SOURCES:
        if not os.path.isfile(os.path.join(HERE, src)):
            print(f"missing source: {src}")
            return 1
    rc = build_rust()
    if rc:
        return rc
    cc = find_compiler()
    if cc is None:
        print(
            "no C compiler found — install gcc on Windows, or run "
            "`pacman -S mingw-w64-gcc` inside WSL (Arch)"
        )
        return 1
    objs = []
    # Linux-hosted gcc (incl. CI) doesn't know __stdcall; Windows targets get
    # a static link so the DLL carries its own libgcc/pthread runtime.
    target_windows = (cc[0] == "wsl") or sys.platform == "win32"
    extra = []
    if not target_windows:
        extra.append("-D__stdcall=")
    elif cc[0] != "wsl":
        extra.append("-static")
    for src in SOURCES:
        obj = src.replace(".c", ".o")
        src_path, obj_path = os.path.join(HERE, src), os.path.join(HERE, obj)
        if cc[0] == "wsl":
            src_path, obj_path = wsl_path(src_path), wsl_path(obj_path)
        cmd = cc + ["-O2", "-fPIC", "-Wall", "-Wextra", "-c", "-o", obj_path, src_path] + extra
        print("compiling:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        # Compiler warnings used to be swallowed on a successful build
        # (capture_output + only printing stderr on failure). Surface them;
        # the CI lint job additionally enforces -Werror via -fsyntax-only.
        if r.stderr.strip():
            print(r.stderr.rstrip())
        if r.returncode != 0:
            return 1
        objs.append(obj)
    out = OUT
    if cc[0] == "wsl":
        out = wsl_path(OUT)
    obj_paths = [
        (wsl_path(os.path.join(HERE, o)) if cc[0] == "wsl" else os.path.join(HERE, o)) for o in objs
    ]
    cmd = cc + ["-O2", "-shared", "-o", out] + extra + obj_paths
    print("linking:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    print(f"built: {OUT} ({os.path.getsize(OUT)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
