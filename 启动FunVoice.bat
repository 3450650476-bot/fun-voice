@echo off
chcp 65001 >nul
title Fun-Voice Dubbing Workbench
cd /d "%~dp0"

:menu
cls
echo.
echo  ==============================================
echo    Fun-Voice Video Dubbing Workbench
echo  ==============================================
echo    [1] Start Web UI (opens browser)
echo    [2] Environment check (Python/GPU/models)
echo    [3] Repair environment (uv sync)
echo    [4] Adapt GPU (torch cu128/cu126/cpu)
echo    [5] Fetch missing models (whisper/Qwen3-TTS/ffmpeg)
echo    [0] Exit
echo  ==============================================
echo.
set /p choice=Enter your choice: 

if "%choice%"=="1" goto start_ui
if "%choice%"=="2" goto check_env
if "%choice%"=="3" goto fix_env
if "%choice%"=="4" goto adapt_gpu
if "%choice%"=="5" goto fetch_models
if "%choice%"=="0" exit /b
echo.
echo Invalid choice, please retry.
pause
goto menu

:start_ui
echo.
echo Starting Fun-Voice ... (first model load may take 1-2 min, keep this window open)
echo Hint: if the page shows an error after restart, press Ctrl+F5 in browser to refresh.
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Virtual env .venv not found. Run option [3] first.
    pause
    goto menu
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m app.server
echo.
echo Server stopped.
pause
goto menu

:check_env
echo.
echo ===== Python environment =====
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'.'); import torch, faster_whisper, gradio; print('Python  :', sys.version.split()[0]); print('torch   :', torch.__version__); print('CUDA    :', 'yes' if torch.cuda.is_available() else 'NO'); print('whisper :', faster_whisper.__version__); print('gradio  :', gradio.__version__)"
if errorlevel 1 (
    echo.
    echo   [ERROR] Python check failed - .venv may be broken.
    echo           Run option [3] Repair environment to fix it.
)
echo.
echo ===== Model assets =====
if exist "..\models\MDX_Net_Models\MDX23C-8KFFT-InstVoc_HQ.ckpt" (echo   [OK] MDX23C vocal separator) else (echo   [MISS] MDX23C vocal separator)
if exist "..\models\faster-whisper" (echo   [OK] faster-whisper ASR) else (echo   [MISS] faster-whisper ASR)
if exist "..\..\HuggingFace\models\Qwen3-TTS-12Hz-1.7B-Base" (echo   [OK] Qwen3-TTS voice clone) else (echo   [MISS] Qwen3-TTS voice clone)
if exist "..\models\ffmpeg\bin\ffmpeg.exe" (echo   [OK] ffmpeg tool) else (echo   [MISS] ffmpeg tool)
echo.
echo ===== Optional: SoX =====
where sox >nul 2>nul
if errorlevel 1 (
    echo   [INFO] sox not in PATH - qwen_tts prints a startup warning.
    echo          It is harmless（12Hz flow does not use it）.
    echo          Install SoX（sox.sourceforge.net）and add to PATH to silence it.
) else (
    echo   [OK] sox found
)
echo.
echo [DONE] Environment check.
echo.
pause
goto menu

:adapt_gpu
echo.
echo Detecting GPU and adapting torch tier (cu128 / cu126 / cpu) ...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tools\adapt_gpu.py
) else (
    tools\uv\uv.exe run --no-project tools\adapt_gpu.py
)
echo.
pause
goto menu

:fetch_models
echo.
echo Detecting and downloading missing models (faster-whisper / Qwen3-TTS / ffmpeg) ...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tools\fetch_models.py
) else (
    tools\uv\uv.exe run --no-project tools\fetch_models.py
)
echo.
pause
goto menu

:fix_env
echo.
echo Repairing environment (uv sync, cached so usually fast) ...
if not exist "tools\uv\uv.exe" (
    echo.
    echo [ERROR] tools\uv\uv.exe not found.
    pause
    goto menu
)
"./tools/uv/uv.exe" sync
if errorlevel 1 (
    echo.
    echo [ERROR] Environment repair failed, check output above.
) else (
    echo.
    echo [DONE] Environment synced.
)
pause
goto menu
