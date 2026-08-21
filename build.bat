@echo off
REM Build the SBA desktop app with PyInstaller (onedir output in dist\SBA\).
REM Optional: set SBA_WITH_WEB=1 first to bundle the NiceGUI web UI.
setlocal
python -m pip install --quiet pyinstaller
python -m PyInstaller SBA.spec --noconfirm --clean
echo.
echo Build finished. Run dist\SBA\SBA.exe
endlocal
