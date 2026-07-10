# Faster-Qwen3-TTS API 接入指南 (OpenAI 兼容)

该服务提供了一个兼容 OpenAI TTS 协议的流式语音合成接口，支持多音色切换和实时音频流返回。

## 1. 基础信息
- **接口地址**: `http://localhost:7017/v1/audio/speech`
- **请求方法**: `POST`
- **数据格式**: `application/json`
- **输出音频参数**:
    - **采样率 (Sample Rate)**: `24000 Hz` (24kHz)
    - **声道**: 单声道 (Mono)
    - **位深**: 16-bit (PCM_S16LE)

## 2. 请求参数 (JSON)

| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `model` | `string` | 是 | 固定填写 `tts-1` 或模型全称 |
| `input` | `string` | 是 | 需要转语音的文本内容 |
| `voice` | `string` | 是 | 音色名称。当前可用：`alloy`, `yingxue` (均已开启 **Auto** 混读模式) |
| `response_format` | `string` | 否 | `wav` (默认), `pcm`, `mp3` |
| `speed` | `float` | 否 | 语速，默认为 `1.0` (目前忽略，后续版本支持) |

## 3. 响应说明
- **成功**: 返回 `200 OK` 及其音频内容的 **二进制流 (Binary Stream)**。
- **失败**: 返回 `4xx` 或 `5xx` 状态码及 JSON 格式的错误详情。
- **流式处理**: 接口支持 `Transfer-Encoding: chunked`。数据会随着语音合成的进度实时分片发送（Streaming）。

## 4. 调用示例 (Python)

```python
import requests

def generate_speech(text, voice="yingxue", output_file="output.wav"):
    url = "http://localhost:7017/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav"
    }
    
    # 使用 stream=True 以支持流式接收
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            # 这里每拿到一个 chunk 都可以进行实时处理（如播放）
                            f.write(chunk)
                print(f"音频已保存至: {output_file}")
            else:
                print(f"请求失败: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"网络异常: {e}")

# 使用示例
if __name__ == "__main__":
    generate_speech("这就是流式响应的测试效果。")
```

## 5. 开发者注意事项
1.  **首字延迟 (TTFA)**: 建议客户端使用流式读取（Chunked-read）以实现最低延迟。
2.  **重采样建议**: 服务端输出为 **24kHz**。如果接入 LiveKit (48kHz)，请在客户端添加重采样逻辑。
3.  **混读模式**: 我已将 `yingxue` 和 `alloy` 音色配置为 `Auto` 语言模式，支持中英文自动识别与混读。
4.  **音色切换**: 该配置由服务器端的 `voices.json` 管理。
3.  **连接超时**: 建议客户端设置 30s-60s 的读取超时，因为复杂的文本可能需要更长的生成时间。
4.  **CUDA 加速**: 该接口利用 CUDA Graphs 技术，除首次冷启动请求外，后续请求均可实现毫秒级响应。
