# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the C rollout kernel into native/sba_rollout.dll (D25).

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
SRC = os.path.join(HERE, 'sba_rollout.c')
OUT = os.path.join(HERE, 'sba_rollout.dll')


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
    if not os.path.isfile(SRC):
        print(f'missing source: {SRC}')
        return 1
    cc = find_compiler()
    if cc is None:
        print('no C compiler found — install gcc on Windows, or run '
              '`pacman -S mingw-w64-gcc` inside WSL (Arch)')
        return 1
    src, out = SRC, OUT
    if cc[0] == 'wsl':
        src, out = wsl_path(SRC), wsl_path(OUT)
    cmd = cc + ['-O2', '-shared', '-o', out, src]
    print('compiling:', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    print(f'built: {OUT} ({os.path.getsize(OUT)} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
