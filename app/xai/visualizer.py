import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_heatmap(timeline: list[dict], width: int = 12, height: int = 4) -> str:
    scores = [t["attention_score"] for t in timeline]
    times = [t["start_ms"] / 1000.0 for t in timeline]

    fig, ax = plt.subplots(figsize=(width, height))
    ax.fill_between(times, scores, alpha=0.4, color="crimson")
    ax.plot(times, scores, color="darkred", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Attention Score")
    ax.set_title("Wav2Vec2 Attention — Temporal Saliency")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_per_layer_heatmap(
    attentions: list[np.ndarray],
    timeline: list[dict],
    num_layers: int = 4,
) -> dict[str, str]:
    times = [t["start_ms"] / 1000.0 for t in timeline]
    result = {}
    num_avail = len(attentions)

    for i in range(min(num_layers, num_avail)):
        layer_idx = num_avail - num_layers + i if num_layers > 0 else i
        if layer_idx < 0:
            layer_idx = i
        attn = attentions[layer_idx].mean(axis=0).mean(axis=-1)
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.fill_between(times, attn, alpha=0.4, color="steelblue")
        ax.plot(times, attn, color="navy", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Attn")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        result[f"layer_{layer_idx}"] = base64.b64encode(buf.read()).decode("utf-8")

    return result
