@echo off
title LiveKit TTS Transport Test Launcher
echo ==========================================
echo   LiveKit TTS 链路测试一键启动器
echo ==========================================

echo [1/2] 正在后台启动 TTS API 服务 (Port 7017)...
start "Faster-Qwen3-TTS-API" cmd /c ".\run_api.bat"

echo 等待 TTS 服务加载模型和预热 (15秒)...
timeout /t 15 /nobreak > nul

echo [2/2] 正在启动 LiveKit 测试 Agent (Port 8089)...
echo 请在浏览器访问: http://localhost:8089
".venv\Scripts\python.exe" livekit_tts_test.py

pause
