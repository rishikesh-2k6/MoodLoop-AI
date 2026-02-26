@echo off
REM ───────────────────────────────────────────────────────
REM  MoodLoop AI — Quick Run Script (Windows)
REM  Generates one video with default settings.
REM ───────────────────────────────────────────────────────

echo.
echo   MoodLoop AI — Starting pipeline...
echo.

cd /d "%~dp0"
python main.py %*

echo.
echo   Done! Check the output/ folder for your video.
pause
