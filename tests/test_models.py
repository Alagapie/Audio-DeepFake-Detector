import numpy as np
import pytest

from app.config import settings
from app.utils.audio_utils import preprocess_pipeline


@pytest.mark.asyncio
async def test_preprocess_pipeline(dummy_audio):
    audio, sr = dummy_audio
    result = await preprocess_pipeline(audio, sr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert np.max(np.abs(result)) <= 1.0


@pytest.mark.asyncio
async def test_preprocess_pipeline_truncate():
    sr = settings.sample_rate
    long_audio = np.ones(sr * 120)
    result = await preprocess_pipeline(long_audio, sr)
    assert len(result) == settings.max_audio_samples
