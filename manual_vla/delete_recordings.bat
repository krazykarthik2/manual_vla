@echo off
cd /d "%~dp0"
echo Deleting demonstration dataset recordings...
if exist "data\demos" (
    del /q "data\demos\*.npz" 2>nul
    echo All .npz demo recordings deleted.
) else (
    echo data\demos directory does not exist.
)
pause
