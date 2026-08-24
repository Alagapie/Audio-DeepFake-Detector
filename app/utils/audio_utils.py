import io
import logging
import subprocess

import librosa
import numpy as np
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)

_ffmpeg_available: bool | None = None


def check_ffmpeg() -> bool:
    global _ffmpeg_available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        _ffmpeg_available = True
        logger.info("ffmpeg found — all audio formats supported")
    except (FileNotFoundError, subprocess.CalledProcessError):
        _ffmpeg_available = False
        logger.warning(
            "ffmpeg not found. Only WAV/FLAC supported. "
            "Install ffmpeg for MP3/M4A/OGG Opus support: "
            "https://ffmpeg.org/download.html"
        )
    return _ffmpeg_available


def _convert_with_ffmpeg(data: bytes, target_sr: int) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-ar", str(target_sr), "-ac", "1", "-f", "wav", "pipe:1"],
        input=data, capture_output=True, check=True,
    )
    audio, _ = sf.read(io.BytesIO(result.stdout))
    return audio


async def load_audio_from_bytes(data: bytes, target_sr: int = settings.sample_rate) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(io.BytesIO(data))
    except Exception:
        if _ffmpeg_available is True:
            logger.info("soundfile failed, falling back to ffmpeg")
            audio = _convert_with_ffmpeg(data, target_sr)
            return audio.astype(np.float32), target_sr
        raise RuntimeError(
            "Unsupported audio format. Install ffmpeg (https://ffmpeg.org/download.html) "
            "for MP3/M4A/OGG Opus support, or use WAV/FLAC."
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32), target_sr


async def preprocess_pipeline(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != settings.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    if len(audio) > settings.max_audio_samples:
        audio = audio[:settings.max_audio_samples]
    return audio.astype(np.float32)
