@echo off
setlocal
cd /d "%~dp0"

:: Ensure tex folder exists
if not exist "papers\\tex" (
    echo Error: papers\tex folder missing.
    exit /b 1
)

:: Compile manual_vla.tex
pdflatex -interaction=nonstopmode -output-directory=papers\\tex papers\\tex\\manual_vla.tex
move /Y "papers\\tex\\manual_vla.pdf" "papers\\manual_vla.pdf"

:: Compile smolvla_dobot.tex
pdflatex -interaction=nonstopmode -output-directory=papers\\tex papers\\tex\\smolvla_dobot.tex
move /Y "papers\\tex\\smolvla_dobot.pdf" "papers\\smolvla_dobot.pdf"

:: Compile full_smolvla_dobot.tex
pdflatex -interaction=nonstopmode -output-directory=papers\\tex papers\\tex\\full_smolvla_dobot.tex
move /Y "papers\\tex\\full_smolvla_dobot.pdf" "papers\\full_smolvla_dobot.pdf"

echo ------------------------------------------------------------
echo All PDFs compiled and placed in papers\ folder.
echo ------------------------------------------------------------

endlocal
