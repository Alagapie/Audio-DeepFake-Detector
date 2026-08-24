# Audio Deepfake Detector

Production-grade audio deepfake & voice clone detection system. Upload an audio file and get a **bonafide / spoof** verdict with confidence scores, per-model breakdowns, explainable AI heatmaps, and a downloadable PDF report.

## How It Works

The system fuses two complementary models through a trained meta-classifier:

| Component | Role |
|-----------|------|
| **Wav2Vec2** (`Vansh180/deepfake-audio-wav2vec2`) | Self-supervised speech representations fine-tuned for deepfake detection |
| **AASIST** | Spectro-temporal graph attention network specialized in anti-spoofing |
| **Meta MLP** | Stacked classifier that combines both model outputs into a final verdict |
| **XAI module** | Attention rollout heatmaps + temporal timeline showing *which* parts of the audio triggered the decision |

## API Endpoints

Base URL: `http://localhost:8000/api/v1`

### `GET /health`
Model loading status, device, and uptime.

### `POST /detect`
Analyze an audio file (`.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, max 50 MB):

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@sample_audio.wav"
```

Response:

```json
{
  "request_id": "3f8a...",
  "prediction": "spoof",
  "confidence": 0.97,
  "scores": { "wav2vec2": 0.95, "aasist": 0.91, "meta_mlp": 0.97 },
  "process_time_ms": 812.4,
  "attention_maps": {
    "overall_heatmap": "<base64 PNG>",
    "layer_count": 4,
    "timeline": [ ... ]
  },
  "report_download_link": "/api/v1/report/3f8a..."
}
```

### `GET /report/{request_id}`
Download the auto-generated PDF analysis report.

## Quickstart

```bash
# 1. Create environment (Python 3.10+)
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights in ./weights/
#    - wav2vec2_pytorch.pt   (auto-downloaded on first run if missing)
#    - aasist_backbone.pt    (auto-downloaded from clovaai/aasist)
#    - meta_mlp.pt           (trained meta classifier)

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive Swagger docs: `http://localhost:8000/docs`

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Mounts `./weights` into the container, exposes port `8000`, includes health checks and 4 GB memory limit.

## Configuration

All settings can be overridden via environment variables with the `ADFD_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADFD_DEVICE` | `cpu` | Torch device (`cuda` supported) |
| `ADFD_MAX_UPLOAD_SIZE_MB` | `50` | Max upload size |
| `ADFD_MAX_AUDIO_SECONDS` | `300` | Max audio length |
| `ADFD_XAI_ENABLED_DEFAULT` | `false` | XAI heatmaps on by default |
| `ADFD_LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
├── app/
│   ├── api/              # FastAPI routes, schemas, lifespan
│   ├── inference/        # Orchestrator + preprocessing pipeline
│   ├── models/           # Wav2Vec2, AASIST, meta classifier wrappers
│   ├── xai/              # Attention capturer, temporal mapper, visualizer
│   └── utils/            # Audio I/O, ONNX exporters, PDF reports, downloads
├── docker/               # Dockerfile + compose
├── research_and_training/# Kaggle/standalone meta-MLP training scripts
├── tests/                # Pytest suite (API, models, XAI)
└── weights/              # Model artifacts (gitignored except .gitkeep)
```

## Testing

```bash
pytest tests/ -v
```
