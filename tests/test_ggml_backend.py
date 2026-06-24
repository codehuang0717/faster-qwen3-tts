import sys
import types

import numpy as np
import pytest

from faster_qwen3_tts.ggml_backend import GGMLQwen3TTS


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def synthesize(self, **kwargs):
        self.calls.append(("synthesize", kwargs))
        return np.array([0.0, 0.25], dtype=np.float32), 24000

    def stream(self, *, codec_chunk_sec, **kwargs):
        self.calls.append(("stream", {"codec_chunk_sec": codec_chunk_sec, **kwargs}))
        yield np.array([0.0, 0.25], dtype=np.float32), 24000

    def load_rvq_codes(self, path):
        self.calls.append(("load_rvq_codes", {"path": path}))
        return np.arange(8, dtype=np.int32).reshape(4, 2)

    def speaker_names(self):
        return ["aiden", "vivian"]


@pytest.fixture
def qwentts_cpp_stub(monkeypatch):
    module = types.SimpleNamespace(
        QwenTTS=object,
        load_speaker_embedding=lambda _path: np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    monkeypatch.setitem(sys.modules, "qwentts_cpp", module)
    return module


def test_cached_speaker_only_forwards_spk_embedding(qwentts_cpp_stub):
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    audio_list, sr = model.generate_voice_clone(
        text="hello",
        language="English",
        ref_spk="speaker.spk",
        xvec_only=True,
    )

    assert sr == 24000
    np.testing.assert_array_equal(audio_list[0], np.array([0.0, 0.25], dtype=np.float32))
    _name, kwargs = runtime.calls[0]
    np.testing.assert_array_equal(
        kwargs["ref_spk_emb"],
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert kwargs["ref_text"] is None
    assert "ref_audio_24k" not in kwargs
    assert "ref_codes" not in kwargs


def test_cached_icl_forwards_spk_rvq_and_ref_text(qwentts_cpp_stub):
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    model.generate_voice_clone(
        text="hello",
        language="English",
        ref_spk="speaker.spk",
        ref_rvq="reference.rvq",
        ref_text="reference transcript",
    )

    assert runtime.calls[0] == ("load_rvq_codes", {"path": "reference.rvq"})
    _name, kwargs = runtime.calls[1]
    np.testing.assert_array_equal(kwargs["ref_codes"], np.arange(8, dtype=np.int32).reshape(4, 2))
    assert kwargs["ref_text"] == "reference transcript"


def test_cached_streaming_forwards_adapter_timing(qwentts_cpp_stub):
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    chunk, sr, timing = next(
        model.generate_voice_clone_streaming(
            text="hello",
            language="English",
            ref_spk_emb=np.ones(3, dtype=np.float32),
            chunk_size=4,
        )
    )

    assert sr == 24000
    np.testing.assert_array_equal(chunk, np.array([0.0, 0.25], dtype=np.float32))
    assert timing["adapter_prepare_ms"] >= 0.0
    _name, kwargs = runtime.calls[0]
    assert kwargs["codec_chunk_sec"] == pytest.approx(4 / 12.5)
    np.testing.assert_array_equal(kwargs["ref_spk_emb"], np.ones(3, dtype=np.float32))


def test_cached_references_reject_raw_audio_mix(qwentts_cpp_stub):
    model = GGMLQwen3TTS(_FakeRuntime())

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.generate_voice_clone(
            text="hello",
            language="English",
            ref_audio="reference.wav",
            ref_spk="speaker.spk",
        )


def test_cached_rvq_requires_ref_text(qwentts_cpp_stub):
    model = GGMLQwen3TTS(_FakeRuntime())

    with pytest.raises(ValueError, match="ref_text is required"):
        model.generate_voice_clone(
            text="hello",
            language="English",
            ref_spk="speaker.spk",
            ref_rvq="reference.rvq",
        )


def test_ggml_speaker_listing_uses_runtime():
    model = GGMLQwen3TTS(_FakeRuntime())

    assert model.get_supported_speakers() == ["aiden", "vivian"]
