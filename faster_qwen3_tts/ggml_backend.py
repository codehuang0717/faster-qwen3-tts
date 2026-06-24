"""Optional GGML/qwentts.cpp backend adapter.

This module keeps the public faster-qwen3-tts API shape while delegating
generation to the separately packaged qwentts.cpp C ABI wrapper.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_QWEN_FRAME_RATE = 24000 / 1920


def _require_qwentts_cpp():
    try:
        from qwentts_cpp import QwenTTS, load_speaker_embedding
    except ImportError as exc:
        raise ImportError(
            "backend='ggml' requires the optional qwentts-cpp-python package. "
            "Install that package first, then retry with backend='ggml'."
        ) from exc
    return QwenTTS, load_speaker_embedding


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int = 24000) -> np.ndarray:
    if src_sr == dst_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = audio.shape[0] / float(src_sr)
    dst_n = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _load_ref_audio_24k(
    ref_audio: Union[str, Path],
    *,
    append_silence: bool = True,
    silence_secs: float = 0.5,
) -> np.ndarray:
    audio, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if append_silence and silence_secs > 0:
        audio = np.concatenate([audio, np.zeros(int(sr * silence_secs), dtype=np.float32)])
    return _resample_linear(np.asarray(audio, dtype=np.float32), int(sr), 24000)


class GGMLQwen3TTS:
    """FasterQwen3TTS-compatible wrapper backed by qwentts.cpp/GGML."""

    sample_rate = 24000

    def __init__(self, runtime):
        self.runtime = runtime

    def get_supported_speakers(self) -> list[str]:
        if not hasattr(self.runtime, "speaker_names"):
            return []
        return list(self.runtime.speaker_names())

    @classmethod
    def from_gguf(
        cls,
        talker_path: Union[str, Path],
        codec_path: Union[str, Path],
        *,
        library_path: Optional[Union[str, Path]] = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ) -> "GGMLQwen3TTS":
        QwenTTS, _load_speaker_embedding = _require_qwentts_cpp()
        runtime = QwenTTS(
            talker_path=talker_path,
            codec_path=codec_path,
            library_path=library_path,
            use_fa=use_fa,
            clamp_fp16=clamp_fp16,
        )
        return cls(runtime)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        quant: str = "BF16",
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        library_path: Optional[Union[str, Path]] = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ) -> "GGMLQwen3TTS":
        QwenTTS, _load_speaker_embedding = _require_qwentts_cpp()
        runtime = QwenTTS.from_pretrained(
            model_name,
            quant=quant,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            library_path=library_path,
            use_fa=use_fa,
            clamp_fp16=clamp_fp16,
        )
        return cls(runtime)

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        xvec_only: bool = False,
        non_streaming_mode: Optional[bool] = None,
        append_silence: bool = True,
        instruct: Optional[str] = None,
        ref_spk: Optional[Union[str, Path]] = None,
        ref_rvq: Optional[Union[str, Path]] = None,
        ref_spk_emb: Optional[np.ndarray] = None,
        ref_codes: Optional[np.ndarray] = None,
        voice_clone_prompt=None,
    ) -> Tuple[list, int]:
        if voice_clone_prompt is not None:
            raise NotImplementedError(
                "The GGML backend cannot consume torch voice_clone_prompt objects; "
                "use ref_spk/ref_rvq cached qwentts.cpp latents instead."
            )
        if instruct:
            raise NotImplementedError("qwentts.cpp currently rejects instruct for base voice-clone models")
        ref_kwargs, _adapter_prepare_ms = self._resolve_clone_reference(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
            ref_spk=ref_spk,
            ref_rvq=ref_rvq,
            ref_spk_emb=ref_spk_emb,
            ref_codes=ref_codes,
        )
        audio, sr = self.runtime.synthesize(
            text=text,
            lang=language,
            **ref_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return [audio], sr

    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        xvec_only: bool = False,
        non_streaming_mode: Optional[bool] = None,
        append_silence: bool = True,
        parity_mode: bool = False,
        instruct: Optional[str] = None,
        ref_spk: Optional[Union[str, Path]] = None,
        ref_rvq: Optional[Union[str, Path]] = None,
        ref_spk_emb: Optional[np.ndarray] = None,
        ref_codes: Optional[np.ndarray] = None,
        voice_clone_prompt=None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        if voice_clone_prompt is not None:
            raise NotImplementedError(
                "The GGML backend cannot consume torch voice_clone_prompt objects; "
                "use ref_spk/ref_rvq cached qwentts.cpp latents instead."
            )
        if instruct:
            raise NotImplementedError("qwentts.cpp currently rejects instruct for base voice-clone models")
        ref_kwargs, adapter_prepare_ms = self._resolve_clone_reference(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
            ref_spk=ref_spk,
            ref_rvq=ref_rvq,
            ref_spk_emb=ref_spk_emb,
            ref_codes=ref_codes,
        )
        yield from self._stream_runtime(
            text=text,
            lang=language,
            **ref_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            adapter_prepare_ms=adapter_prepare_ms,
        )

    def _resolve_clone_reference(
        self,
        *,
        ref_audio: Optional[Union[str, Path]],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
        ref_spk: Optional[Union[str, Path]],
        ref_rvq: Optional[Union[str, Path]],
        ref_spk_emb: Optional[np.ndarray],
        ref_codes: Optional[np.ndarray],
    ) -> Tuple[dict, float]:
        adapter_start = time.perf_counter()
        has_cached_ref = any(
            value is not None for value in (ref_spk, ref_rvq, ref_spk_emb, ref_codes)
        )
        if ref_audio is not None and has_cached_ref:
            raise ValueError(
                "ref_audio is mutually exclusive with cached qwentts.cpp references "
                "(ref_spk/ref_rvq/ref_spk_emb/ref_codes)."
            )
        if ref_spk is not None and ref_spk_emb is not None:
            raise ValueError("Use either ref_spk or ref_spk_emb, not both.")
        if ref_rvq is not None and ref_codes is not None:
            raise ValueError("Use either ref_rvq or ref_codes, not both.")
        if xvec_only and (ref_rvq is not None or ref_codes is not None):
            raise ValueError("ref_rvq/ref_codes require ICL mode; set xvec_only=False.")

        if ref_audio is not None:
            ref_audio_24k = _load_ref_audio_24k(ref_audio, append_silence=append_silence)
            effective_ref_text = "" if xvec_only else ref_text
            return {
                "ref_audio_24k": ref_audio_24k,
                "ref_text": effective_ref_text or None,
            }, (time.perf_counter() - adapter_start) * 1000

        if not has_cached_ref:
            raise ValueError("Voice cloning requires ref_audio or cached ref_spk/ref_spk_emb.")

        spk_emb = self._load_cached_speaker(ref_spk, ref_spk_emb)
        codes = self._load_cached_codes(ref_rvq, ref_codes)
        if codes is not None and not ref_text:
            raise ValueError("ref_text is required when using ref_rvq/ref_codes.")

        ref_kwargs = {
            "ref_spk_emb": spk_emb,
            "ref_text": ref_text if codes is not None else None,
        }
        if codes is not None:
            ref_kwargs["ref_codes"] = codes
        return ref_kwargs, (time.perf_counter() - adapter_start) * 1000

    def _load_cached_speaker(
        self,
        ref_spk: Optional[Union[str, Path]],
        ref_spk_emb: Optional[np.ndarray],
    ) -> np.ndarray:
        if ref_spk_emb is not None:
            spk_emb = np.ascontiguousarray(ref_spk_emb, dtype=np.float32).reshape(-1)
        elif ref_spk is not None:
            _QwenTTS, load_speaker_embedding = _require_qwentts_cpp()
            spk_emb = load_speaker_embedding(ref_spk)
        else:
            raise ValueError("ref_spk/ref_spk_emb is required for cached voice cloning.")

        if spk_emb.size == 0:
            raise ValueError("ref_spk_emb must not be empty.")
        return spk_emb

    def _load_cached_codes(
        self,
        ref_rvq: Optional[Union[str, Path]],
        ref_codes: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if ref_codes is not None:
            return np.ascontiguousarray(ref_codes, dtype=np.int32)
        if ref_rvq is None:
            return None
        load_rvq_codes: Optional[Callable[..., np.ndarray]] = getattr(
            self.runtime,
            "load_rvq_codes",
            None,
        )
        if load_rvq_codes is None:
            raise NotImplementedError(
                "Cached RVQ references require qwentts-cpp-python >= 0.1.0a1 "
                "and qwentts.cpp ABI v2."
            )
        return load_rvq_codes(ref_rvq)

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        audio, sr = self.runtime.synthesize(
            text=text,
            lang=language,
            speaker=speaker,
            instruct=instruct or None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return [audio], sr

    def generate_custom_voice_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        adapter_start = time.perf_counter()
        yield from self._stream_runtime(
            text=text,
            lang=language,
            speaker=speaker,
            instruct=instruct or None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            adapter_prepare_ms=(time.perf_counter() - adapter_start) * 1000,
        )

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        audio, sr = self.runtime.synthesize(
            text=text,
            lang=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return [audio], sr

    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        adapter_start = time.perf_counter()
        yield from self._stream_runtime(
            text=text,
            lang=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            adapter_prepare_ms=(time.perf_counter() - adapter_start) * 1000,
        )

    def _stream_runtime(self, *, chunk_size: int, adapter_prepare_ms: float = 0.0, **kwargs):
        chunk_sec = max(1, int(chunk_size)) / _QWEN_FRAME_RATE
        start = time.perf_counter()
        last = start
        for idx, (chunk, sr) in enumerate(
            self.runtime.stream(codec_chunk_sec=chunk_sec, **kwargs)
        ):
            now = time.perf_counter()
            native_profile = getattr(self.runtime, "last_stream_profile", None)
            yield chunk, sr, {
                "chunk_index": idx,
                "decode_ms": (now - last) * 1000,
                "total_ms": (now - start) * 1000,
                "adapter_prepare_ms": adapter_prepare_ms if idx == 0 else 0.0,
                "ggml_profile": dict(native_profile) if native_profile else None,
                "is_final": False,
            }
            last = now
