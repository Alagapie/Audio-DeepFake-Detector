import logging

import torch

from app.config import settings
from app.models.aasist_pipeline import AASISTModel

logger = logging.getLogger(__name__)


def export_aasist_onnx():
    logger.info("Building AASIST model")
    model = AASISTModel()
    if settings.aasist_pytorch_path.exists():
        state = torch.load(settings.aasist_pytorch_path, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("Loaded pretrained weights")
    model.eval()

    dummy_input = torch.randn(1, 64600)
    torch.onnx.export(
        model,
        dummy_input,
        str(settings.weights_dir / "aasist.onnx"),
        input_names=["input"],
        output_names=["embedding", "logits"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Exported to {settings.weights_dir / 'aasist.onnx'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_aasist_onnx()
