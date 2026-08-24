import numpy as np

from app.config import settings


def map_attention_to_timeline(
    layer_maps: list[np.ndarray],
    sample_rate: int = settings.sample_rate,
) -> list[dict]:
    feat_encoder_stride = 320
    n_frames = layer_maps[0].shape[0]
    frame_duration_ms = (feat_encoder_stride / sample_rate) * 1000.0

    aggregate = layer_maps[-1]
    aggregate = (aggregate - aggregate.min()) / (aggregate.max() - aggregate.min() + 1e-8)

    timeline = []
    for i in range(n_frames):
        start_ms = round(i * frame_duration_ms, 1)
        end_ms = round((i + 1) * frame_duration_ms, 1)
        timeline.append({
            "frame": i,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "attention_score": float(aggregate[i]),
        })

    return timeline


def frames_to_time(frame_idx: int, stride: int = 320, sample_rate: int = settings.sample_rate) -> float:
    return (frame_idx * stride) / sample_rate * 1000.0
