@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Compiling papers using Python (fpdf2)...
echo ============================================================

python generate_papers.py

echo ============================================================
echo Done. PDFs are in papers\ folder.
echo ============================================================

endlocal
