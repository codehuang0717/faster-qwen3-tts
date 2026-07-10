# Windows Deployment Notes

## 1. Clone

```powershell
git clone https://github.com/codehuang0717/faster-qwen3-tts.git
cd faster-qwen3-tts
```

## 2. Install Dependencies

Run the setup script from the repository root:

```powershell
.\setup_windows.bat
```

The script creates `.venv`, installs CUDA PyTorch, installs this package with API dependencies, and downloads the Qwen3-TTS models.

## 3. Start The API

```powershell
.\run_api.bat
```

The API listens on:

```text
http://localhost:7017/v1/audio/speech
```

## 4. Optional LiveKit Test

```powershell
.\run_livekit_test.bat
```

Then open:

```text
http://localhost:8089
```

## Dependency Summary

Core dependencies come from `pyproject.toml`:

- `qwen-tts`
- `transformers`
- `torch`
- `numpy`
- `soundfile`
- `huggingface-hub`

API dependencies come from the `demo` extra:

- `fastapi`
- `uvicorn[standard]`
- `python-multipart`

Project-specific test dependencies:

- `httpx`
- `livekit`

## Notes

- `.venv` is intentionally not committed to GitHub; each machine should create its own environment.
- `voices.json` uses `yingxue.wav` as a relative path, so keep `voices.json` and `yingxue.wav` in the repository root.
- If CUDA is not detected, install or update your NVIDIA driver and use the CUDA-enabled PyTorch wheel from the setup script.
