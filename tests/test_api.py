import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.config import settings


@pytest.fixture
def dummy_wav_bytes():
    sr = settings.sample_rate
    t = np.linspace(0, 1, int(sr * 1), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_health():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


@pytest.mark.asyncio
async def test_detect_unsupported_format():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.mp4", b"fake", "video/mp4")}
        resp = await client.post("/api/v1/detect", files=files)
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_detect(dummy_wav_bytes):
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.wav", dummy_wav_bytes, "audio/wav")}
        params = {"xai_enabled": False}
        resp = await client.post("/api/v1/detect", files=files, params=params)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("bona-fide", "spoof")
        assert 0.0 <= data["confidence"] <= 1.0
