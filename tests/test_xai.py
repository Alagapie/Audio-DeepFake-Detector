import numpy as np

from app.xai.attention_capturer import process_attentions, extract_frame_attention
from app.xai.temporal_mapper import map_attention_to_timeline, frames_to_time


def _make_dummy_attentions(num_layers=6, num_heads=12, time_frames=50):
    return [np.random.randn(num_heads, time_frames, time_frames).astype(np.float32) for _ in range(num_layers)]


def test_process_attentions_shape():
    attns = _make_dummy_attentions(time_frames=32)
    result = process_attentions(attns)
    assert len(result) == len(attns) + 1
    assert result[-1].shape[0] == 32


def test_extract_frame_attention():
    attns = _make_dummy_attentions(time_frames=16)
    result = extract_frame_attention(attns, layer_idx=0)
    assert result.shape[0] == 16


def test_temporal_mapping():
    fake_scores = [np.random.randn(20) for _ in range(3)]
    timeline = map_attention_to_timeline(fake_scores, sample_rate=16000)
    assert len(timeline) == 20
    assert "start_ms" in timeline[0]
    assert "attention_score" in timeline[0]
    assert 0.0 <= timeline[0]["attention_score"] <= 1.0


def test_frames_to_time():
    t = frames_to_time(10, stride=320, sample_rate=16000)
    assert t == 200.0
