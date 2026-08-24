import logging
import time

import numpy as np

from app.config import settings
from app.inference.preprocessor import preprocess_audio_bytes
from app.models.aasist_pipeline import AASISTPipeline
from app.models.meta_classifier import MetaClassifier
from app.models.wav2vec2_pipeline import Wav2Vec2Pipeline

logger = logging.getLogger(__name__)


class DetectionResult:
    def __init__(self, prediction: str, confidence: float, scores: dict[str, float], process_time_ms: float):
        self.prediction = prediction
        self.confidence = confidence
        self.scores = scores
        self.process_time_ms = process_time_ms


class XaiResult:
    def __init__(self, detection: DetectionResult, attention_maps: dict):
        self.detection = detection
        self.attention_maps = attention_maps


class InferenceOrchestrator:
    def __init__(self, wav2vec2: Wav2Vec2Pipeline, aasist: AASISTPipeline, meta_classifier: MetaClassifier):
        self._w2v = wav2vec2
        self._aasist = aasist
        self._meta = meta_classifier

    async def detect(self, audio_bytes: bytes, xai_enabled: bool = False) -> DetectionResult | XaiResult:
        start = time.perf_counter()
        audio = await preprocess_audio_bytes(audio_bytes)

        w2v_confidence, w2v_emb = await self._w2v.infer_with_embedding(audio)

        aasist_confidence = 0.5
        aasist_emb: np.ndarray | None = None
        try:
            aasist_confidence, aasist_emb = await self._aasist.infer_with_embedding(audio)
        except Exception as e:
            logger.warning(f"AASIST inference failed: {e}")

        ensemble_conf = await self._meta.predict_proba(w2v_emb, aasist_emb, w2v_confidence, aasist_confidence)
        elapsed = (time.perf_counter() - start) * 1000.0

        det = DetectionResult(
            prediction="spoof" if ensemble_conf > 0.5 else "bona-fide",
            confidence=ensemble_conf,
            scores={"wav2vec2": w2v_confidence, "aasist": aasist_confidence, "ensemble": ensemble_conf},
            process_time_ms=round(elapsed, 1),
        )

        if xai_enabled:
            return await self._detect_with_xai(audio, det, start)
        return det

    async def _detect_with_xai(self, audio: np.ndarray, det: DetectionResult, start: float) -> XaiResult:
        from app.xai.attention_capturer import process_attentions
        from app.xai.temporal_mapper import map_attention_to_timeline
        from app.xai.visualizer import generate_heatmap

        _, _, attentions = await self._w2v.get_attentions(audio)
        layer_maps = process_attentions(attentions)
        timeline = map_attention_to_timeline(layer_maps)
        heatmap_b64 = generate_heatmap(timeline)

        return XaiResult(
            detection=det,
            attention_maps={
                "overall_heatmap": heatmap_b64,
                "layer_count": len(attentions),
                "timeline": timeline,
            },
        )
