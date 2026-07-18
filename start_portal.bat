@echo off
title Document Renamer
cd /d "%~dp0"
echo Starting the Document Renamer portal...
python -m pip install -q -r requirements.txt
python app.py
pause
