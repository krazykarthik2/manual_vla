@echo off
cd /d "%~dp0"
echo Deleting trained model checkpoints...
if exist "models" (
    del /q "models\*.pth" 2>nul
    echo Checkpoints in models/ deleted.
)
if exist "vla_model.pth" (
    del /q "vla_model.pth" 2>nul
    echo vla_model.pth deleted.
)
echo Model cleanup complete.
pause
