# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the SBA desktop app (PySide6 + vendored SiliconUI).

Build (from the repo root, with PyInstaller installed):
    pyinstaller SBA.spec --noconfirm

The output lands in dist/SBA/ (onedir). The default build is desktop-only
(PySide6 + vendored SiliconUI); the NiceGUI web UI is opt-in, matching the
project's D1 design. To also bundle it (the desktop's "Enable NiceGUI Web UI"
switch then works), build with:
    set SBA_WITH_WEB=1
    pyinstaller SBA.spec --noconfirm

torch/numpy are intentionally excluded: the packaged build's AlphaZero option
falls back to MCTS (ai.alphazero_move catches the missing import).
"""

import os

ROOT = os.path.abspath(SPECPATH)
WITH_WEB = os.environ.get('SBA_WITH_WEB', '0') == '1'

excludes = ['torch', 'numpy', 'pytest', 'matplotlib', 'scipy',
            'IPython', 'jupyter']
if not WITH_WEB:
    excludes += ['webui', 'nicegui', 'uvicorn', 'fastapi', 'starlette',
                 'websockets', 'aiofiles', 'anyio']

datas = [
    (os.path.join(ROOT, 'sba.toml'), '.'),
    (os.path.join(ROOT, 'static'), 'static'),
    (os.path.join(ROOT, 'vendor', 'siui'), 'vendor/siui'),
]

# Ship the trained Ultimate checkpoint as the bundled default model. Users can
# replace it by dropping a new az_ultimate.pt next to the executable.
model_pt = os.path.join(ROOT, 'models', 'az_ultimate.pt')
if os.path.exists(model_pt):
    datas.append((model_pt, 'models'))

binaries = []
hiddenimports = []
if WITH_WEB:
    try:
        from PyInstaller.utils.hooks import collect_all
        d, b, h = collect_all('nicegui')
        datas += d
        binaries = b
        hiddenimports += h
    except Exception as e:  # noqa: BLE001 - surface the failure at build time
        raise SystemExit(f'SBA_WITH_WEB=1 but bundling nicegui failed: {e}')

a = Analysis(
    [os.path.join(ROOT, 'SBA.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SBA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='SBA',
)
