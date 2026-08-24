import logging
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sample_rate: int = 16000
    max_audio_seconds: int = 300
    max_audio_samples: int = sample_rate * max_audio_seconds

    aasist_nb_samp: int = 64600

    device: str = "cpu"

    weights_dir: Path = Path("weights")

    wav2vec2_model_id: str = "Vansh180/deepfake-audio-wav2vec2"
    wav2vec2_pytorch_path: Path = weights_dir / "wav2vec2_pytorch.pt"

    aasist_pytorch_path: Path = weights_dir / "aasist_backbone.pt"
    aasist_download_url: str = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"

    meta_classifier_path: Path = weights_dir / "meta_mlp.pt"

    xai_enabled_default: bool = False
    xai_layers_to_visualize: int = 4

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    max_upload_size_mb: int = 50

    log_level: str = "INFO"

    model_config = {"env_prefix": "ADFD_"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


settings = Settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio_deepfake")
