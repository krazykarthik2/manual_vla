@echo off
cd /d "%~dp0"
echo Deleting demonstration dataset recordings...

if exist "data\demonstrations" (
    del /q "data\demonstrations\*.npz" 2>nul
    echo Deleted all demos from data\demonstrations\
)
if exist "data\demos" (
    del /q "data\demos\*.npz" 2>nul
    echo Deleted all demos from data\demos\
)

echo Done! All demonstration recordings have been completely deleted.
