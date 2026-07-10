"""
LiveKit TTS Transport Test
==========================
Tests the LiveKit audio transport by:
1. Browser sends text via REST → treated as "LLM response"
2. Agent calls TTS API to generate PCM stream
3. Agent pushes audio via LiveKit AudioSource (identical to livekit_worker.py)
4. Browser receives audio via LiveKit WebRTC and plays it
If browser audio is smooth → problem is in SIP/Asterisk
If browser audio is choppy → problem is in LiveKit AudioSource or TTS
"""
import asyncio
import json
import time
import logging
import numpy as np
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from livekit import rtc, api
# ── Configuration ──
LIVEKIT_URL = "ws://34.58.12.77:7880"
LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"
TTS_URL = "http://localhost:7017/v1/audio/speech"
TTS_VOICE = "yingxue"
ROOM_NAME = "tts-test"
JITTER_BUFFER_MS = 800
HTTP_PORT = 8089
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit-tts-test")
# ── Shared state ──
text_queue: asyncio.Queue = None
agent_room: rtc.Room = None
# ── FastAPI App ──
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
@app.get("/")
async def index():
    with open("livekit_test.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
@app.get("/token")
async def get_token():
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("browser-user")
        .with_name("Browser User")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME, can_subscribe=True, can_publish_data=True))
    )
    return JSONResponse({
        "token": token.to_jwt(),
        "url": LIVEKIT_URL,
        "room": ROOM_NAME,
    })
@app.post("/synthesize")
async def synthesize(request: dict):
    text = request.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    await text_queue.put(text)
    logger.info(f"Queued text: {text[:50]}...")
    return JSONResponse({"status": "queued"})
# ── LiveKit Agent Logic ──
async def push_tts_audio(room: rtc.Room, text: str):
    """Call TTS API, push PCM audio via AudioSource — mirrors livekit_worker.py exactly."""
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("tts", source)
    publication = await room.local_participant.publish_track(track)
    logger.info(f"Published audio track for: {text[:40]}...")
    try:
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": TTS_VOICE,
            "response_format": "pcm",
        }
        start_time = time.perf_counter()
        first_chunk_logged = False
        jitter_buffer = b""
        jitter_threshold = int(24000 * (JITTER_BUFFER_MS / 1000) * 2)  # bytes
        playback_started = False
        total_samples = 0
        playback_start_time = 0
        # Jitter tracking: record each TTS chunk arrival time
        chunk_arrival_times = []
        frame_push_times = []
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", TTS_URL, json=payload, timeout=30.0) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    logger.error(f"TTS error {resp.status_code}: {err[:200]}")
                    return
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if not chunk:
                        continue
                    chunk_arrival_times.append(time.perf_counter())
                    if not first_chunk_logged:
                        ttfb_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(f"TTS TTFB: {ttfb_ms:.0f}ms")
                        first_chunk_logged = True
                        # Send metric to browser via data channel
                        try:
                            await room.local_participant.publish_data(
                                json.dumps({"type": "ttfb", "value": round(ttfb_ms)}).encode(),
                                reliable=True,
                            )
                        except Exception:
                            pass
                    jitter_buffer += chunk
                    if not playback_started and len(jitter_buffer) >= jitter_threshold:
                        playback_started = True
                        playback_start_time = time.time()
                    if playback_started:
                        audio_np = np.frombuffer(jitter_buffer, dtype=np.int16)
                        jitter_buffer = b""
                        # Push in 480-sample (20ms) frames — same as livekit_worker.py
                        for i in range(0, len(audio_np), 480):
                            samples = audio_np[i : i + 480]
                            if len(samples) > 0:
                                frame = rtc.AudioFrame(
                                    data=samples.tobytes(),
                                    sample_rate=24000,
                                    num_channels=1,
                                    samples_per_channel=len(samples),
                                )
                                await source.capture_frame(frame)
                                total_samples += len(samples)
                                frame_push_times.append(time.perf_counter())
                        await asyncio.sleep(0.01)
        # Flush remaining jitter buffer
        if jitter_buffer:
            if not playback_started:
                playback_started = True
                playback_start_time = time.time()
            audio_np = np.frombuffer(jitter_buffer, dtype=np.int16)
            for i in range(0, len(audio_np), 480):
                samples = audio_np[i : i + 480]
                if len(samples) > 0:
                    frame = rtc.AudioFrame(
                        data=samples.tobytes(),
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=len(samples),
                    )
                    await source.capture_frame(frame)
                    total_samples += len(samples)
        # Wait for audio to drain (same logic as livekit_worker.py)
        if total_samples > 0 and playback_started:
            total_duration = total_samples / 24000
            elapsed = time.time() - playback_start_time
            remaining = total_duration - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining + 0.3)
        total_ms = (time.perf_counter() - start_time) * 1000
        audio_dur = total_samples / 24000 if total_samples > 0 else 0.01
        # Calculate TTS chunk jitter (variation in chunk arrival intervals)
        tts_jitter_ms = 0
        chunk_intervals = []
        if len(chunk_arrival_times) > 1:
            chunk_intervals = [
                (chunk_arrival_times[i+1] - chunk_arrival_times[i]) * 1000
                for i in range(len(chunk_arrival_times) - 1)
            ]
            avg_interval = sum(chunk_intervals) / len(chunk_intervals)
            tts_jitter_ms = sum(abs(ci - avg_interval) for ci in chunk_intervals) / len(chunk_intervals)
        # Calculate frame push jitter (variation in AudioSource.capture_frame intervals)
        push_jitter_ms = 0
        if len(frame_push_times) > 1:
            push_intervals = [
                (frame_push_times[i+1] - frame_push_times[i]) * 1000
                for i in range(len(frame_push_times) - 1)
            ]
            avg_push = sum(push_intervals) / len(push_intervals)
            push_jitter_ms = sum(abs(pi - avg_push) for pi in push_intervals) / len(push_intervals)
        logger.info(
            f"Done — Total: {total_ms:.0f}ms, Audio: {audio_dur:.2f}s, RTF: {total_ms/1000/audio_dur:.3f}, "
            f"TTS Chunk Jitter: {tts_jitter_ms:.1f}ms (chunks={len(chunk_arrival_times)}), "
            f"Push Jitter: {push_jitter_ms:.2f}ms (frames={len(frame_push_times)})"
        )
        try:
            await room.local_participant.publish_data(
                json.dumps({
                    "type": "done",
                    "total_ms": round(total_ms),
                    "audio_s": round(audio_dur, 2),
                    "tts_chunk_jitter_ms": round(tts_jitter_ms, 1),
                    "push_jitter_ms": round(push_jitter_ms, 2),
                    "tts_chunks": len(chunk_arrival_times),
                    "push_frames": len(frame_push_times),
                }).encode(),
                reliable=True,
            )
        except Exception:
            pass
    except Exception as e:
        logger.error(f"push_tts_audio error: {e}", exc_info=True)
    finally:
        await room.local_participant.unpublish_track(publication.sid)
        logger.info("Unpublished audio track.")
async def agent_worker():
    """Background task: join LiveKit room, consume text queue, push TTS audio."""
    global agent_room
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("tts-agent")
        .with_name("TTS Agent")
        .with_grants(api.VideoGrants(
            room_join=True, room=ROOM_NAME, room_admin=True,
            can_publish=True, can_subscribe=True, can_publish_data=True,
        ))
    )
    room = rtc.Room()
    agent_room = room
    await room.connect(LIVEKIT_URL, token.to_jwt())
    logger.info(f"Agent joined LiveKit room: {ROOM_NAME}")
    while True:
        text = await text_queue.get()
        logger.info(f"Processing text: {text[:60]}...")
        await push_tts_audio(room, text)
@app.on_event("startup")
async def startup():
    global text_queue
    text_queue = asyncio.Queue()
    asyncio.create_task(agent_worker())
    logger.info(f"Server ready at http://localhost:{HTTP_PORT}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT)