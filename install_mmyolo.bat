@echo off
REM ============================================================
REM  Cài mmyolo vào môi trường ow-ovd
REM  Chạy file này bằng cách double-click từ File Explorer
REM ============================================================
set PYTHONNOUSERSITE=1
pushd %~dp0third_party\mmyolo
D:\miniconda\envs\ow-ovd\python.exe setup.py develop --no-deps
popd
echo.
echo [OK] mmyolo installed!
pause
