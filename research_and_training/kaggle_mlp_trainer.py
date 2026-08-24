import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.aasist_pipeline import AASISTPipeline
from app.models.meta_classifier import MetaMLP
from app.models.wav2vec2_pipeline import Wav2Vec2Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("meta_trainer")


class AudioEmbeddingDataset(Dataset):
    def __init__(self, csv_path: str, max_samples: int | None = None):
        self.pairs: list[tuple[str, float]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append((row["path"], float(row.get("label", 0))))
        if max_samples:
            self.pairs = self.pairs[:max_samples]
        logger.info(f"Loaded {len(self.pairs)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.pairs[idx]


def extract_embeddings(
    dataset: AudioEmbeddingDataset,
    w2v: Wav2Vec2Pipeline,
    aasist_pipeline: AASISTPipeline,
    device: torch.device,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    import os

    import librosa
    import soundfile as sf

    all_embs: list[torch.Tensor] = []
    all_labels: list[float] = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting embeddings"):
        batch_paths = dataset.pairs[i : i + batch_size]
        for path, label in batch_paths:
            try:
                audio_np, sr = sf.read(path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != settings.sample_rate:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=settings.sample_rate)
                peak = np.max(np.abs(audio_np))
                if peak > 0:
                    audio_np = audio_np / peak
                audio_np = audio_np[:settings.max_audio_samples].astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

            try:
                _, w2v_emb = w2v.infer_with_embedding(audio_np)
                _, aasist_emb = aasist_pipeline.infer_with_embedding(audio_np)
            except Exception as e:
                logger.warning(f"Failed to extract embeddings for {path}: {e}")
                continue

            combined = np.concatenate([w2v_emb, aasist_emb])
            all_embs.append(torch.from_numpy(combined).float().to(device))
            all_labels.append(label)

    logger.info(f"Extracted {len(all_embs)} embeddings")
    return torch.stack(all_embs), torch.tensor(all_labels, dtype=torch.float32, device=device)


class EmbeddingDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
    device: torch.device,
):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            feats, labels = feats.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = float("inf")
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats, labels = feats.to(device), labels.to(device).unsqueeze(1)
                    preds = model(feats)
                    val_loss += criterion(preds, labels).item()
            avg_val = val_loss / len(val_loader)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), "best_meta_mlp.pt")
                logger.info(f"Epoch {epoch}: saved best model (val_loss={avg_val:.6f})")

        scheduler.step()
        logger.info(f"Epoch {epoch}: train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with columns: path,label")
    parser.add_argument("--val-data", help="Optional validation CSV")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=str(settings.meta_classifier_path))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading Wav2Vec2...")
    w2v = Wav2Vec2Pipeline()
    w2v.load()

    logger.info("Loading AASIST...")
    aasist_pipeline = AASISTPipeline()
    aasist_pipeline.load()

    w2v_to_device = getattr(w2v._model, "to", None)
    if w2v_to_device:
        w2v._model = w2v._model.to(device)
    if aasist_pipeline._model is not None:
        aasist_pipeline._model = aasist_pipeline._model.to(device)

    logger.info(f"Extracting embeddings from {args.data}...")
    train_ds = AudioEmbeddingDataset(args.data, max_samples=args.max_samples)
    X, y = extract_embeddings(train_ds, w2v, aasist_pipeline, device, batch_size=8)

    train_dataset = EmbeddingDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_data:
        val_ds = AudioEmbeddingDataset(args.val_data, max_samples=args.max_samples)
        X_val, y_val = extract_embeddings(val_ds, w2v, aasist_pipeline, device, batch_size=8)
        val_dataset = EmbeddingDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = MetaMLP().to(device)
    logger.info(f"Training MetaMLP: {sum(p.numel() for p in model.parameters()):,} params")

    train(model, train_loader, val_loader, args.epochs, args.lr, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Final model saved to {output_path}")
    print(f"\nCopy this file to {settings.meta_classifier_path} for inference:\n  cp {output_path} {settings.meta_classifier_path}")


if __name__ == "__main__":
    main()
