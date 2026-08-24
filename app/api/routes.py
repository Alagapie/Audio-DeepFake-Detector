import time
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.schemas import DetectResponse, HealthResponse, XaiResponse
from app.config import logger, settings
from app.dependencies import _components, get_orchestrator
from app.inference.orchestrator import InferenceOrchestrator, XaiResult
from app.utils.pdf_report import REPORTS, generate_report

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health():
    from app.api.lifespan import _start_time

    comps = _components
    uptime = time.time() - _start_time if _start_time else 0.0
    return HealthResponse(
        status="ok",
        wav2vec2_loaded=comps is not None,
        aasist_loaded=comps is not None and comps.aasist._model is not None,
        meta_classifier_loaded=comps is not None and comps.meta_classifier.is_loaded,
        device=settings.device,
        uptime_seconds=round(uptime, 1),
    )


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    orch: InferenceOrchestrator = Depends(get_orchestrator),
):
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext and f".{ext}" not in settings.supported_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")

    data = await file.read()
    if len(data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    result = await orch.detect(data, xai_enabled=True)
    rid = str(uuid.uuid4())

    if isinstance(result, XaiResult):
        resp = XaiResponse(
            request_id=rid,
            prediction=result.detection.prediction,
            confidence=result.detection.confidence,
            scores=result.detection.scores,
            process_time_ms=result.detection.process_time_ms,
            attention_maps=result.attention_maps,
        )
    else:
        resp = DetectResponse(
            request_id=rid,
            prediction=result.prediction,
            confidence=result.confidence,
            scores=result.scores,
            process_time_ms=result.process_time_ms,
        )

    attn_maps = getattr(resp, "attention_maps", None)
    if hasattr(attn_maps, "model_dump"):
        attn_maps = attn_maps.model_dump()

    pdf_bytes = generate_report(
        request_id=rid,
        prediction=resp.prediction,
        confidence=resp.confidence,
        scores=resp.scores,
        process_time_ms=resp.process_time_ms,
        attention_maps=attn_maps,
    )
    REPORTS[rid] = pdf_bytes

    return {
        **resp.model_dump(),
        "report_download_link": f"/api/v1/report/{rid}",
    }


@router.get("/report/{request_id}")
async def download_report(request_id: str):
    pdf_bytes = REPORTS.get(request_id)
    if pdf_bytes is None:
        raise HTTPException(status_code=404, detail="Report not found or expired")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="deepfake_report_{request_id[:8]}.pdf"'},
    )
