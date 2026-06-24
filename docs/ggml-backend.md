# qwentts.cpp GGML Backend

This repo keeps `faster-qwen3-tts` as the user-facing package and adds
`qwentts-cpp-python` as an optional native runtime package.

## Package Layout

```text
faster-qwen3-tts
  Existing Torch/CUDA-graph backend
  Optional GGML adapter in faster_qwen3_tts.ggml_backend

qwentts-cpp-python
  Pure Python ctypes wrapper over qwentts.cpp's C ABI
  Platform wheel packaging for libqwen + libggml

qwentts.cpp
  Pascal's C++/GGML implementation, built separately with CMake
```

The main package stays installable without native binaries:

```bash
pip install faster-qwen3-tts
```

The GGML backend is opt-in:

```bash
pip install "faster-qwen3-tts[ggml]"
```

For local wrapper development, clone the wrapper repo beside this checkout and
install it in editable mode:

```bash
git clone https://github.com/andimarafioti/qwentts-cpp-python ../qwentts-cpp-python
pip install -e ../qwentts-cpp-python
```

Development build with local native libraries:

```bash
cd ../qwentts-cpp-python
python scripts/build_native.py --source /path/to/qwentts.cpp --backend cuda --clean
pip install -e .
```

## Python Usage

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    backend="ggml",
    quant="BF16",
)

audio_list, sr = model.generate_voice_design(
    text="Welcome to the show.",
    instruct="Warm, confident narrator with slight British accent",
    language="English",
)
```

Local GGUF paths are supported:

```python
model = FasterQwen3TTS.from_pretrained(
    "unused",
    backend="ggml",
    gguf_talker_path="qwen-talker-1.7b-voicedesign-BF16.gguf",
    gguf_codec_path="qwen-tokenizer-12hz-BF16.gguf",
    qwentts_library_path="/path/to/libqwen.so",
)
```

CLI usage:

```bash
faster-qwen3-tts --backend ggml --quant BF16 design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "Warm, confident narrator" \
  --text "Welcome to the show." \
  --language English \
  --output out.wav
```

Cached qwentts.cpp references are supported for base voice cloning:

```python
audio_list, sr = model.generate_voice_clone(
    text="Cached speaker and RVQ latents avoid reference audio encoding.",
    language="English",
    ref_spk="freeman.spk",
    ref_rvq="freeman.rvq",
    ref_text="The transcript for the cached reference audio.",
)
```

Use `ref_spk` by itself for speaker-only conditioning. Use `ref_spk` +
`ref_rvq` + `ref_text` for cached ICL conditioning. Raw `ref_audio` and cached
references are mutually exclusive.

## Current ABI Gaps

The qwentts.cpp C ABI is already enough for buffered and streaming
synthesis, voice cloning, CustomVoice, and VoiceDesign. These gaps remain
before treating the backend as full parity:

- no C ABI for creating `.spk` / `.rvq` files from `ref_audio` inside Python
- no `non_streaming_mode` switch
- base-model `instruct` is rejected by qwentts.cpp
- KV-cache length is fixed in qwentts.cpp
- raw reference audio must reach the ABI as mono float 24 kHz

The public GGUF model repo used by the resolver is
`Serveurperso/Qwen3-TTS-GGUF`.

## TTFA Profiling

The GGML adapter attaches a `ggml_profile` snapshot to the first streamed
chunk's `timing` dict. It includes Python-visible boundaries such as ctypes
parameter packing, lock wait, native `qt_synthesize()` entry, first native
audio callback, callback copy/queue cost, and first Python yield.

Run the focused diagnostic with:

```bash
python benchmarks/profile_ggml_ttfa.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --mode custom \
  --speaker aiden \
  --quant BF16 \
  --cache-dir .cache/qwentts \
  --local-files-only \
  --chunk-sizes 2,4,8,16
```

With a `libqwen` that includes native profiling markers, the same command also
prints native phase splits for prompt build, first talker prefill, first code
predictor step, first emit entry, and first codec decode. Use `--warmup-runs 0`
to inspect true first-request latency; use the default warmup to inspect
steady-state server latency after CUDA/GGML graphs and prefix caches are hot.

For a direct Torch-vs-GGML comparison, use the benchmark entry point:

```bash
MODEL_SIZE=1.7B MODE=custom QUANT=BF16 ./benchmark.sh backends
MODEL_SIZE=0.6B QUANT=BF16 ./benchmark.sh backend-base
```

The legacy CUDA-graph-only benchmarks still run with `./benchmark.sh`.

## Wheel Build

Build native libraries into the wrapper package:

```bash
cd ../qwentts-cpp-python
python scripts/build_native.py --source /path/to/qwentts.cpp --backend cuda --clean
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu130 python -m build --wheel
```

The build script copies `libqwen` and `libggml*` into
`src/qwentts_cpp/lib/`. The package marks itself as a binary distribution
so the wheel receives a platform tag instead of `py3-none-any`.

The first public release should be CUDA-first. `faster-qwen3-tts` already
requires CUDA for its existing fast path, so a CPU-only native wheel would be
surprising and not very useful for the main package audience.

For public CUDA wheels, use the manual GitHub Actions workflow:

```text
andimarafioti/qwentts-cpp-python:.github/workflows/wheels.yml
```

The workflow builds both Linux x86_64 and Linux aarch64 CUDA wheels as artifacts.
It installs the CUDA toolkit on GitHub-hosted runners and compiles explicit CUDA
architectures, so no GPU is required at build time. Download the artifacts and
test them locally on representative CUDA machines before upload.

Local reproduction mirrors the workflow:

```bash
cd ../qwentts-cpp-python
python scripts/build_native.py \
  --source third_party/qwentts.cpp \
  --backend cuda \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja \
  --cmake-arg="-DCMAKE_CUDA_ARCHITECTURES=75-virtual;80-virtual;86-real;89-real;120a-real;121a-real"
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu130 python -m build --wheel
```

Recommended first wheel targets:

- Linux x86_64 CUDA from GitHub Actions
- Linux aarch64 CUDA from GitHub Actions
- CPU wheels only as a later fallback package or dev artifact

CUDA-linked `libqwen` depends on CUDA runtime libraries and an NVIDIA driver at
runtime. That is acceptable for `faster-qwen3-tts[ggml]`, but it means CPU and
CUDA artifacts should not be uploaded with the same package name, version, and
platform compatibility tag unless the wheel contains a loader that can select
between both backends. For local wheelhouses, set a build tag so artifacts do
not overwrite each other:

```bash
# CUDA artifact
python scripts/build_native.py --source /path/to/qwentts.cpp --backend cuda --clean
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu130 python -m build --wheel

# Optional CPU dev artifact
python scripts/build_native.py --source /path/to/qwentts.cpp --backend cpu --clean
QWENTTS_CPP_WHEEL_BUILD_TAG=1cpu python -m build --wheel
```

For PyPI, CUDA-only under `qwentts-cpp-python` is coherent with this project:
pip will choose the platform wheel automatically, and users choose only the
`faster-qwen3-tts[ggml]` extra. CPU support can be a separate distribution or a
future combined wheel if there is demand.
