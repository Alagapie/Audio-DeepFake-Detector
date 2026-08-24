import numpy as np

from app.config import settings


async def preprocess_audio_bytes(data: bytes) -> np.ndarray:
    from app.utils.audio_utils import load_audio_from_bytes, preprocess_pipeline
    audio, sr = await load_audio_from_bytes(data, target_sr=settings.sample_rate)
    audio = await preprocess_pipeline(audio, sr)
    return audio
