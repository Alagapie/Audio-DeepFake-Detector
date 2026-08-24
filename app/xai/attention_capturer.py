import numpy as np


def _attention_rollout(attentions: list[np.ndarray]) -> np.ndarray:
    seq_len = attentions[0].shape[-1]
    rollout = np.eye(seq_len)
    for layer_attn in attentions:
        A = layer_attn.mean(axis=0)
        A = 0.5 * A + 0.5 * np.eye(seq_len)
        A = A / A.sum(axis=-1, keepdims=True)
        rollout = A @ rollout
    return rollout


def process_attentions(attentions: list[np.ndarray]) -> list[np.ndarray]:
    layer_scores = []
    for attn in attentions:
        attn = attn.mean(axis=0)
        attn = attn.mean(axis=0)
        layer_scores.append(attn)

    rollout = _attention_rollout(attentions)
    importance = rollout.mean(axis=0)
    layer_scores.append(importance)

    return layer_scores


def extract_frame_attention(attentions: list[np.ndarray], layer_idx: int = -1) -> np.ndarray:
    if layer_idx == -1:
        rollout = _attention_rollout(attentions)
        return rollout.mean(axis=0)
    attn = attentions[layer_idx]
    attn = attn.mean(axis=0)
    attn = attn.mean(axis=0)
    return attn
