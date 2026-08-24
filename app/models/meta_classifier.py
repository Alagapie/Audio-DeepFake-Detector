import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MetaMLP(nn.Module):
    def __init__(self, w2v_dim: int = 768, aasist_dim: int = 160, joint_dim: int = 64):
        super().__init__()
        self.w2v_branch = nn.Sequential(
            nn.Linear(w2v_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.aasist_branch = nn.Sequential(
            nn.Linear(aasist_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, joint_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(joint_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        w2v = x[:, :768]
        aasist = x[:, 768:]
        w2v_out = self.w2v_branch(w2v)
        aasist_out = self.aasist_branch(aasist)
        fused = torch.cat([w2v_out, aasist_out], dim=-1)
        return self.classifier(fused)


class MetaClassifier:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model: MetaMLP | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                f"Meta-classifier weights not found at {self.model_path}. "
                "Falling back to score averaging. "
                "Train one with research_and_training/kaggle_mlp_trainer.py"
            )
            self._model = None
            return
        self._model = MetaMLP()
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        try:
            self._model.load_state_dict(state)
        except RuntimeError as e:
            logger.warning(
                f"Failed to load meta-classifier weights (arch mismatch?): {e}. "
                "Falling back to score averaging. "
                "Retrain with research_and_training/kaggle_mlp_trainer.py"
            )
            self._model = None
            return
        self._model.eval()
        logger.info(f"Meta-classifier loaded from {self.model_path} (dual-branch: W2V2 768→128 + AASIST 160→64 → 192→64→1)")

    async def predict_proba(
        self,
        wav2vec2_embedding: torch.Tensor | None,
        aasist_embedding: torch.Tensor | None,
        wav2vec2_confidence: float = 0.5,
        aasist_confidence: float = 0.5,
    ) -> float:
        if not self.is_loaded or wav2vec2_embedding is None or aasist_embedding is None:
            ensemble = (wav2vec2_confidence + aasist_confidence) / 2.0
            return ensemble

        with torch.no_grad():
            w2v = torch.as_tensor(wav2vec2_embedding)
            aasist = torch.as_tensor(aasist_embedding)
            x = torch.cat([w2v, aasist], dim=0).unsqueeze(0)
            score = float(self._model(x).item())
        return score

    def save(self, path: Path | None = None) -> None:
        dest = path or self.model_path
        if self._model is None:
            raise ValueError("No trained model to save")
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), dest)
        logger.info(f"Meta-classifier saved to {dest}")
