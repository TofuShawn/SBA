# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the SBA native engine into native/sba_engine.dll (D27).

Modules: sba_ultt.c (shared rules) + sba_rollout.c (plain/pattern) +
sba_mcts.c (full tree search) + sba_minimax.c (normal solve / ultimate
alpha-beta / flat MCTS). One DLL exports everything.

Compiler search order:
1. a Windows gcc/clang on PATH (native build)
2. WSL's mingw-w64 cross compiler (x86_64-w64-mingw32-gcc), if WSL exists —
   installs via `pacman -S mingw-w64-gcc` on an Arch WSL

Usage:  python native/build.py
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sba_engine.dll')
SOURCES = ['sba_ultt.c', 'sba_rollout.c', 'sba_mcts.c', 'sba_minimax.c']


def find_compiler():
    for exe in ('gcc', 'clang', 'cc'):
        path = shutil.which(exe)
        if path:
            return [path]
    try:
        r = subprocess.run(
            ['wsl', 'x86_64-w64-mingw32-gcc', '--version'],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return ['wsl', 'x86_64-w64-mingw32-gcc']
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def wsl_path(win_path):
    """Translate a Windows path into the WSL /mnt/<drive> form."""
    drive, rest = os.path.splitdrive(os.path.abspath(win_path))
    return '/mnt/' + drive[0].lower() + rest.replace('\\', '/')


def main():
    for src in SOURCES:
        if not os.path.isfile(os.path.join(HERE, src)):
            print(f'missing source: {src}')
            return 1
    cc = find_compiler()
    if cc is None:
        print('no C compiler found — install gcc on Windows, or run '
              '`pacman -S mingw-w64-gcc` inside WSL (Arch)')
        return 1
    objs = []
    for src in SOURCES:
        obj = src.replace('.c', '.o')
        src_path, obj_path = os.path.join(HERE, src), os.path.join(HERE, obj)
        if cc[0] == 'wsl':
            src_path, obj_path = wsl_path(src_path), wsl_path(obj_path)
        cmd = cc + ['-O2', '-c', '-o', obj_path, src_path]
        print('compiling:', ' '.join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
        objs.append(obj)
    out = OUT
    if cc[0] == 'wsl':
        out = wsl_path(OUT)
    cmd = cc + ['-O2', '-shared', '-o', out] + [
        (wsl_path(os.path.join(HERE, o)) if cc[0] == 'wsl'
         else os.path.join(HERE, o)) for o in objs]
    print('linking:', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    print(f'built: {OUT} ({os.path.getsize(OUT)} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
