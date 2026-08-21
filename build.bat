@echo off
REM Build the SBA desktop app with PyInstaller.
REM   build.bat            -> onefile: dist\SBA.exe (single portable file)
REM   build.bat onedir     -> onedir:  dist\SBA\SBA.exe + _internal folder
REM Optional: set SBA_WITH_WEB=1 first to bundle the NiceGUI web UI.
setlocal
python -m pip install --quiet pyinstaller
if /I "%1"=="onedir" (
    set SBA_ONEFILE=0
) else (
    set SBA_ONEFILE=1
)
python -m PyInstaller SBA.spec --noconfirm --clean
echo.
if /I "%1"=="onedir" (
    echo Build finished. Copy the whole dist\SBA\ folder (exe + _internal).
) else (
    echo Build finished. dist\SBA.exe is a single self-contained file.
)
endlocal
