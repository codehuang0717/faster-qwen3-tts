@echo off
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "REPO_DIR=%~dp0"
set "VOICES_JSON=%~dp0voices.json"

cd /d "%REPO_DIR%"
"%VENV_PYTHON%" examples\openai_server.py ^
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base ^
    --voices "%VOICES_JSON%" ^
    --port 7017 ^
    --max-seq-len 1024 ^
    --chunk-size 10 ^
    --xvec-only ^
    --warmup
pause
