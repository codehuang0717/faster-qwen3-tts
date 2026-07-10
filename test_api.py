import requests
import json

url = "http://localhost:7017/v1/audio/speech"
payload = {
    "model": "tts-1",
    "input": "这就是我的声音克隆效果，你觉得好听吗？",
    "voice": "yingxue",
    "response_format": "wav"
}
headers = {
    "Content-Type": "application/json"
}

print(f"Sending request to {url} with voice 'yingxue'...")
try:
    response = requests.post(url, json=payload, headers=headers, stream=True)
    if response.status_code == 200:
        with open("test_output_yingxue.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Success! Audio saved to test_output_yingxue.wav")
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
