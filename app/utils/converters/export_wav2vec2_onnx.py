import logging

import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from app.config import settings

logger = logging.getLogger(__name__)


def export_wav2vec2_onnx():
    logger.info(f"Loading model: {settings.wav2vec2_model_id}")
    processor = AutoFeatureExtractor.from_pretrained(settings.wav2vec2_model_id)
    model = AutoModelForAudioClassification.from_pretrained(settings.wav2vec2_model_id)
    model.eval()

    dummy_input = processor([0.0] * settings.sample_rate, sampling_rate=settings.sample_rate, return_tensors="pt", padding=True, truncation=True).input_values

    torch.onnx.export(
        model,
        dummy_input,
        str(settings.weights_dir / "wav2vec2.onnx"),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    torch.save(model.state_dict(), settings.wav2vec2_pytorch_path)
    logger.info(f"Exported to {settings.weights_dir / 'wav2vec2.onnx'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_wav2vec2_onnx()
