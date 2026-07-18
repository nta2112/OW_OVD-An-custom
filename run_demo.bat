@echo off
REM ══════════════════════════════════════════════════════════════════
REM  OW-OVD Pest Detection Demo - Launcher
REM  Chỉnh các biến dưới đây theo đường dẫn thực tế của bạn
REM ══════════════════════════════════════════════════════════════════

REM Python từ conda env ow-ovd (tránh conflict user site-packages)
set PYTHON=D:\miniconda\envs\ow-ovd\python.exe
REM Tắt user-level site-packages để tránh conflict
set PYTHONNOUSERSITE=1
REM Ép buộc mmcv dùng chế độ pure Python không cần C++ extensions
set MMCV_WITH_OPS=0

set CONFIG=configs/open_world/mowod/custom/ip102_t1.py
set CHECKPOINT=C:\Users\HP\Downloads\OW-OVD_checkpoint\t1_best.pth
set DEVICE=cpu

REM Nếu có GPU CUDA: set DEVICE=cuda:0

echo.
echo  ====================================================
echo   OW-OVD Pest Detection Demo
echo   Python: %PYTHON%
echo  ====================================================
echo   Config:     %CONFIG%
echo   Checkpoint: %CHECKPOINT%
echo   Device:     %DEVICE%
echo  ====================================================
echo.

%PYTHON% demo_app.py ^
    --config "%CONFIG%" ^
    --checkpoint "%CHECKPOINT%" ^
    --device %DEVICE% ^
    --port 7860

pause

