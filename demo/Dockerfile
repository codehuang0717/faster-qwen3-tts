FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_SIZE=5
ENV ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV DEMO_DEFAULT_BACKEND=ggml
ENV DEMO_GGML_QUANT=BF16
ENV DEMO_WEB_ONLY=1
ENV DEMO_REQUIRE_LOGIN=1
ENV USAGE_BUCKET_MOUNT_PATH=/data
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio \
    && python -m pip install "qwentts-cpp-python==0.3.0+cu128" \
        -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu128 \
    && python -m pip install "faster-qwen3-tts[demo] @ git+https://github.com/andimarafioti/faster-qwen3-tts.git@ggml-backend-integration" \
    && python -m pip install -r requirements.txt

EXPOSE 7860
CMD ["python3", "server.py", "--host", "0.0.0.0"]
