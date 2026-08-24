import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous notebook-friendly wrappers
# ---------------------------------------------------------------------------


class Wav2Vec2Model:
    """Synchronous Wav2Vec2 wrapper for notebook/interactive use.

    Usage:
        w2v = Wav2Vec2Model()
        w2v.load()
        conf, emb = w2v.infer(audio_np)           # (confidence, 768-dim)
        conf, emb = w2v.infer_with_embedding(audio_np)  # (confidence, 768-dim)
        conf, emb, attns = w2v.get_attentions(audio_np) # + list of attention maps
    """

    def __init__(self):
        self._processor = None
        self._model = None

    def load(self, model_id: str = "Vansh180/deepfake-audio-wav2vec2"):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        logger.info(f"Loading {model_id}")
        self._processor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model = AutoModelForAudioClassification.from_pretrained(model_id)
        self._model.eval()

    def to(self, device: torch.device | str):
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def infer(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        if audio.ndim == 2:
            audio = audio[0]
        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, 1].item())
        return confidence, logits[0].numpy()

    @torch.no_grad()
    def infer_with_embedding(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        if audio.ndim == 2:
            audio = audio[0]
        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
        outputs = self._model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, 1].item())
        return confidence, hidden.numpy()

    @torch.no_grad()
    def get_attentions(self, audio: np.ndarray) -> tuple[float, np.ndarray, list[np.ndarray]]:
        if audio.ndim == 2:
            audio = audio[0]
        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
        outputs = self._model(**inputs, output_attentions=True, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0).numpy()
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, 1].item())
        attentions = [a.squeeze(0).numpy() for a in outputs.attentions]
        return confidence, hidden, attentions


class AASISTModelWrapper:
    """Synchronous AASIST wrapper for notebook/interactive use.

    Usage:
        aasist = AASISTModelWrapper()
        aasist.load("weights/aasist_backbone.pt")
        conf, emb = aasist.infer(audio_np)                # (confidence, 2-dim logits)
        conf, emb = aasist.infer_with_embedding(audio_np) # (confidence, 160-dim)
    """

    def __init__(self):
        self._model = None

    def load(self, weights_path: str | Path = "weights/aasist_backbone.pt"):
        from app.models.aasist_pipeline import AASISTModel
        self._model = AASISTModel()
        path = Path(weights_path)
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            self._model.load_state_dict(state, strict=False)
        else:
            logger.warning(f"AASIST weights not found at {path}, using random init")
        self._model.eval()

    def to(self, device: torch.device | str):
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def infer(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        audio = self._pad_or_truncate(audio)
        x = torch.from_numpy(audio).float().unsqueeze(0)
        _, logits = self._model(x)
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, 1].item())
        return confidence, logits[0].numpy()

    @torch.no_grad()
    def infer_with_embedding(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        audio = self._pad_or_truncate(audio)
        x = torch.from_numpy(audio).float().unsqueeze(0)
        last_hidden, logits = self._model(x)
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, 1].item())
        return confidence, last_hidden[0].numpy()

    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio[0]
        if len(audio) > 64600:
            return audio[:64600]
        if len(audio) < 64600:
            return np.pad(audio, (0, 64600 - len(audio)))
        return audio


class MetaClassifierWrapper:
    """Synchronous meta-classifier wrapper for notebook/interactive use.

    Usage:
        meta = MetaClassifierWrapper()
        meta.load("weights/meta_mlp.pt")
        score = meta.predict(w2v_emb, aasist_emb)  # 768-dim + 160-dim -> float
    """

    def __init__(self):
        self._model = None
        self._is_loaded = False

    def load(self, weights_path: str | Path = "weights/meta_mlp.pt"):
        from app.models.meta_classifier import MetaMLP
        path = Path(weights_path)
        if path.exists():
            self._model = MetaMLP()
            state = torch.load(path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()
            self._is_loaded = True
            logger.info(f"Meta-classifier loaded from {path}")
        else:
            logger.warning(f"No meta-classifier at {path}")

    def to(self, device: torch.device | str):
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def predict(self, w2v_emb: np.ndarray | torch.Tensor, aasist_emb: np.ndarray | torch.Tensor) -> float:
        if not self._is_loaded:
            raise RuntimeError("Meta-classifier not loaded. Call .load() first.")
        if isinstance(w2v_emb, np.ndarray):
            w2v_emb = torch.from_numpy(w2v_emb)
        if isinstance(aasist_emb, np.ndarray):
            aasist_emb = torch.from_numpy(aasist_emb)
        x = torch.cat([w2v_emb, aasist_emb], dim=0).unsqueeze(0)
        return float(self._model(x).item())

    @torch.no_grad()
    def predict_proba(self, w2v_emb, aasist_emb, w2v_conf=0.5, aasist_conf=0.5) -> float:
        if not self._is_loaded or w2v_emb is None or aasist_emb is None:
            return (w2v_conf + aasist_conf) / 2.0
        return self.predict(w2v_emb, aasist_emb)


# ---------------------------------------------------------------------------
# Convenience: load all models at once
# ---------------------------------------------------------------------------

def load_all(
    w2v_model_id: str = "Vansh180/deepfake-audio-wav2vec2",
    aasist_weights: str | Path = "weights/aasist_backbone.pt",
    meta_weights: str | Path = "weights/meta_mlp.pt",
    device: str | torch.device | None = None,
) -> tuple[Wav2Vec2Model, AASISTModelWrapper, MetaClassifierWrapper]:
    """Load all three models into Python variables. Returns (w2v, aasist, meta).

    Notebook usage:
        from app.notebook_utils import load_all
        w2v, aasist, meta = load_all(device="cpu")

        # Single inference
        conf_w2v, emb_w2v = w2v.infer_with_embedding(audio)
        conf_aasist, emb_aasist = aasist.infer_with_embedding(audio)
        final_conf = meta.predict(emb_w2v, emb_aasist)
    """
    if device is None:
        device = "cpu"

    w2v = Wav2Vec2Model()
    w2v.load(w2v_model_id)
    w2v.to(device)

    aasist = AASISTModelWrapper()
    aasist.load(aasist_weights)
    aasist.to(device)

    meta = MetaClassifierWrapper()
    meta.load(meta_weights)
    meta.to(device)

    return w2v, aasist, meta
