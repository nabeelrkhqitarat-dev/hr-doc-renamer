@echo off
title Agrico Renamer - Local AI Worker
cd /d "%~dp0"

rem ====== EDIT THESE TWO LINES ======
set PORTAL_URL=https://hr-doc-renamer.onrender.com
set WORKER_SECRET=CHANGE_ME
rem ==================================

python local_worker.py
pause
