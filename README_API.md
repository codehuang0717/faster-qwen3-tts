# Faster-Qwen3-TTS API Reference

This document provides technical details for the OpenAI-compatible Text-to-Speech API exposed by this service.

## API Endpoint

- **Base URL**: `http://localhost:7017` (Default)
- **Path**: `/v1/audio/speech`
- **Method**: `POST`

## Request Format

The API follows the OpenAI TTS specification.

### Headers
| Header | Value |
| :--- | :--- |
| `Content-Type` | `application/json` |

### JSON Body Fields
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model` | `string` | Yes | Use `tts-1` or the specific model name. |
| `input` | `string` | Yes | The text to be synthesized into speech. |
| `voice` | `string` | No | The voice ID (e.g., `alloy`, `echo`). Defaults to the primary reference audio. |
| `response_format` | `string` | No | `wav`, `pcm` (Default: `wav`). |
| `speed` | `float` | No | Playback speed (Default: `1.0`). |

## Response Format

The API returns a binary stream of the generated audio.

- **Content-Type**: `audio/wav` or `audio/pcm` depending on the request.
- **Streaming**: Supports Chunked Transfer Encoding. Audio is yielded as soon as the first segment is generated.

## Usage Examples

### Python (using `requests`)
```python
import requests

response = requests.post(
    "http://localhost:7017/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello world, this is a streaming test.",
        "voice": "alloy",
        "response_format": "wav"
    },
    stream=True
)

if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=1024):
        # Process audio chunk (e.g., play or save to file)
        pass
```

### cURL
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello world.",
    "voice": "alloy"
  }' \
  --output speech.wav
```

## Performance & Latency
- **Time to First Audio (TTFA)**: ~1-2 seconds on RTX 2060 after initial CUDA graph capture.
- **Real-Time Factor (RTF)**: > 1.0 (Generation is significantly faster than playback).
- **First Request Warmup**: The very first request after server startup will take ~10 seconds to warm up CUDA graphs. Subsequent requests will be near-instant.

## Server Management
- **Startup**: Run `.\run_api.bat` from the root directory.
- **Stop**: Press `Ctrl+C` in the terminal window.
