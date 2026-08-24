from pydantic import BaseModel


class DetectResponse(BaseModel):
    request_id: str
    prediction: str
    confidence: float
    scores: dict[str, float]
    process_time_ms: float


class XaiAttentionMap(BaseModel):
    overall_heatmap: str
    layer_count: int
    timeline: list[dict]


class XaiResponse(DetectResponse):
    attention_maps: XaiAttentionMap | None = None


class HealthResponse(BaseModel):
    status: str
    wav2vec2_loaded: bool
    aasist_loaded: bool
    meta_classifier_loaded: bool
    device: str
    uptime_seconds: float
