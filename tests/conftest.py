import numpy as np
import pytest

from app.config import settings
from app.utils.audio_utils import preprocess_pipeline


@pytest.fixture
def dummy_wav_bytes():
    sr = settings.sample_rate
    t = np.linspace(0, 2, int(sr * 2), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def dummy_audio():
    sr = settings.sample_rate
    t = np.linspace(0, 2, int(sr * 2), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t), sr
