import logging

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.config import settings

logger = logging.getLogger(__name__)


class Wav2Vec2Pipeline:
    def __init__(self):
        self._name = "wav2vec2"
        self._model: torch.nn.Module | None = None
        self._processor: AutoFeatureExtractor | None = None

    @property
    def name(self) -> str:
        return self._name

    async def load(self) -> None:
        logger.info(f"Loading Wav2Vec2 from {settings.wav2vec2_model_id}")
        self._processor = AutoFeatureExtractor.from_pretrained(settings.wav2vec2_model_id)
        self._model = AutoModelForAudioClassification.from_pretrained(
            settings.wav2vec2_model_id,
            attn_implementation="eager",
        )
        self._model.eval()
        logger.info("Wav2Vec2 model loaded")

    async def unload(self) -> None:
        self._model = None
        self._processor = None

    async def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="np", padding=True, truncation=True, max_length=len(audio))
        return inputs.input_values.astype(np.float32)

    async def infer(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Wav2Vec2 model not loaded")
        if audio.ndim == 2:
            audio = audio[0]
        inputs = self._processor(audio, sampling_rate=settings.sample_rate, return_tensors="pt", padding=True, truncation=True, max_length=len(audio))
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs[0, 1].item())
        embedding = logits[0].numpy().astype(np.float64)
        return confidence, embedding

    async def infer_with_embedding(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Wav2Vec2 model not loaded")
        if audio.ndim == 2:
            audio = audio[0]
        inputs = self._processor(audio, sampling_rate=settings.sample_rate, return_tensors="pt", padding=True, truncation=True, max_length=len(audio))
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            pooled = hidden.mean(dim=1).squeeze(0)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs[0, 1].item())
        return confidence, pooled.numpy()

    async def get_attentions(self, audio: np.ndarray) -> tuple[float, np.ndarray, list[np.ndarray]]:
        if self._model is None:
            raise RuntimeError("Wav2Vec2 model not loaded")
        if audio.ndim == 2:
            audio = audio[0]
        inputs = self._processor(audio, sampling_rate=settings.sample_rate, return_tensors="pt", padding=True, truncation=True, max_length=len(audio))
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            embedding = hidden.mean(dim=1).squeeze(0).numpy()
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs[0, 1].item())
            attn_tuple = outputs.attentions
            if attn_tuple is None:
                raise RuntimeError("Wav2Vec2 did not return attentions. Ensure attn_implementation='eager'.")
            attentions = [a.squeeze(0).numpy() for a in attn_tuple]
        return confidence, embedding, attentions
