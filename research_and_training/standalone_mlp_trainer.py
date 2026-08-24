"""
Standalone Meta-Classifier Trainer for ASVspoof 2019.

Trains a dual-branch MetaMLP (W2V2 768→128 + AASIST 160→64 → 192→64→1) on combined embeddings.
Output: meta_mlp.pt — copy this into the main project's weights/ directory.

Usage:
    pip install torch transformers librosa soundfile tqdm
    python standalone_mlp_trainer.py \
        --train-csv /path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt \
        --eval-csv /path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
        --audio-dir /path/to/ASVspoof2019/LA/ASVspoof2019_LA_train/flac \
        --eval-audio-dir /path/to/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac \
        --output meta_mlp.pt

Expected CSV format (ASVspoof 2019 protocol .trl.txt):
    SPEAKER AUDIO_FILE_NAME - - LABEL
where LABEL is "spoof" or "bonafide".
"""

import argparse
import csv
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("standalone_trainer")

# ---------------------------------------------------------------------------
# Configuration constants (mirror main project)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 60
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
AASIST_NB_SAMP = 64600

# ---------------------------------------------------------------------------
# AASIST architecture (copy of official clovaai/aasist, MIT License)
# ---------------------------------------------------------------------------


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.sample_rate = sample_rate
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        self.mel = filbandwidthsf
        hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.register_buffer("hsupp", hsupp)
        band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * hsupp.numpy() / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * hsupp.numpy() / self.sample_rate)
            band_pass[i, :] = torch.Tensor(np.hamming(self.kernel_size)) * torch.Tensor(hHigh - hLow)
        self.register_buffer("band_pass", band_pass)

    def forward(self, x, mask=False):
        filters = self.band_pass.clone().to(x.device)
        if mask:
            A = random.randint(0, 20)
            A0 = random.randint(0, filters.shape[0] - A)
            filters[A0:A0 + A, :] = 0
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, padding=self.kernel_size // 2, bias=None, groups=1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x):
        x = self.input_drop(x)
        nb = x.size(1)
        x_e = x.unsqueeze(2).expand(-1, -1, nb, -1)
        att_map = torch.tanh(self.att_proj(x_e * x_e.transpose(1, 2)))
        att_map = F.softmax(torch.matmul(att_map, self.att_weight).squeeze(-1) / self.temp, dim=-1)
        out = self.proj_with_att(torch.bmm(att_map, x)) + self.proj_without_att(x)
        out = self.act(self.bn(out.view(-1, out.size(-1))).view(out.size(0), -1, out.size(-1)))
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        for name in ("att_weight11", "att_weight22", "att_weight12", "att_weightM"):
            p = nn.Parameter(torch.FloatTensor(out_dim, 1))
            nn.init.xavier_normal_(p)
            setattr(self, name, p)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x1, x2 = self.proj_type1(x1), self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        x_e = x.unsqueeze(2) * x.unsqueeze(1)
        att_map = torch.tanh(self.att_proj(x_e))
        board = torch.zeros(x.size(0), x.size(1), x.size(1), 1, device=x.device)
        board[:, :n1, :n1] = torch.matmul(att_map[:, :n1, :n1], self.att_weight11)
        board[:, n1:, n1:] = torch.matmul(att_map[:, n1:, n1:], self.att_weight22)
        board[:, :n1, n1:] = torch.matmul(att_map[:, :n1, n1:], self.att_weight12)
        board[:, n1:, :n1] = torch.matmul(att_map[:, n1:, :n1], self.att_weight12)
        att_map = F.softmax(board.squeeze(-1) / self.temp, dim=-1)
        master_e = x * master
        master_att = F.softmax(
            torch.matmul(torch.tanh(self.att_projM(master_e)), self.att_weightM).squeeze(-1) / self.temp, dim=-1
        )
        master = self.proj_with_attM(torch.bmm(master_att.unsqueeze(1), x)) + self.proj_without_attM(master)
        out = self.proj_with_att(torch.bmm(att_map, x)) + self.proj_without_att(x)
        out = self.act(self.bn(out.view(-1, out.size(-1))).view(out.size(0), -1, out.size(-1)))
        return out.narrow(1, 0, n1), out.narrow(1, n1, n2), master


class GraphPool(nn.Module):
    def __init__(self, k, in_dim, p=0.3):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p)

    def forward(self, h):
        scores = torch.sigmoid(self.proj(self.drop(h)))
        _, n, f = h.size()
        n_keep = max(int(n * self.k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, f))


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3))
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1))
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        out = self.conv1(x if self.first else self.selu(self.bn1(x)))
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class AASISTModel(nn.Module):
    """Exact official AASIST architecture (MIT)."""

    def __init__(self):
        super().__init__()
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.7, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]
        self.conv_time = SincConv(out_channels=filts[0], kernel_size=128)
        self.first_bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            ResidualBlock(filts[1], first=True),
            ResidualBlock(filts[2]),
            ResidualBlock(filts[3]),
            ResidualBlock(filts[4]),
            ResidualBlock(filts[4]),
            ResidualBlock(filts[4]),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0])
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0])
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1])
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1])
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1])
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1])
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        e = self.encoder(x)
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))
        bs = x.size(0)
        m1 = self.master1.expand(bs, -1, -1)
        m2 = self.master2.expand(bs, -1, -1)
        t1, s1, m1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=m1)
        s1, t1 = self.pool_hS1(s1), self.pool_hT1(t1)
        ta, sa, ma = self.HtrgGAT_layer_ST12(t1, s1, master=m1)
        t1, s1, m1 = t1 + ta, s1 + sa, m1 + ma
        t2, s2, m2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=m2)
        s2, t2 = self.pool_hS2(s2), self.pool_hT2(t2)
        ta, sa, ma = self.HtrgGAT_layer_ST22(t2, s2, master=m2)
        t2, s2, m2 = t2 + ta, s2 + sa, m2 + ma
        for v in (t1, t2, s1, s2, m1, m2):
            v = self.drop_way(v)
        out_T = torch.max(t1, t2)
        out_S = torch.max(s1, s2)
        master = torch.max(m1, m2)
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return last_hidden, output


# ---------------------------------------------------------------------------
# Meta-MLP classifier
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ASVspoof 2019 protocol file parser
# ---------------------------------------------------------------------------

def load_asvspoof_csv(protocol_path: str, audio_dir: str, max_samples: int | None = None):
    """
    Parse ASVspoof 2019 .trl.txt protocol file.

    Format:
        SPEAKER_ID AUDIO_FILE_NAME - - LABEL
    where LABEL is "bonafide" or "spoof".
    Returns list of (full_audio_path, label_float).
    """
    pairs = []
    audio_dir = Path(audio_dir)
    with open(protocol_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            audio_id = parts[1]
            label_str = parts[4].lower()
            label = 1.0 if label_str == "spoof" else 0.0
            pairs.append((str(audio_dir / f"{audio_id}.flac"), label))
    if max_samples:
        pairs = pairs[:max_samples]
    logger.info(f"Loaded {len(pairs)} samples from {protocol_path}")
    return pairs


# ---------------------------------------------------------------------------
# Audio loading & embedding extraction
# ---------------------------------------------------------------------------

def load_and_preprocess(path: str) -> np.ndarray | None:
    """Load audio file, convert to mono, resample to 16 kHz, normalize peak to 1.0."""
    try:
        import librosa
        import soundfile as sf
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        audio = audio[:MAX_AUDIO_SAMPLES].astype(np.float32)
        return audio
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


@torch.no_grad()
def extract_embeddings(
    pairs: list,
    w2v_model,
    w2v_processor,
    aasist_model,
    device: torch.device,
    batch_audio: int = 4,
    desc: str = "Extracting",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 928-dim combined embeddings (768 W2V + 160 AASIST) for all samples."""
    all_embs, all_labels = [], []
    for path, label in tqdm(pairs, desc=desc):
        audio = load_and_preprocess(path)
        if audio is None:
            continue

        # Wav2Vec2 embedding (768-dim)
        inputs = w2v_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = w2v_model(**inputs, output_hidden_states=True)
        w2v_emb = outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()

        # AASIST embedding (160-dim)
        aasist_audio = audio.copy()
        if len(aasist_audio) > AASIST_NB_SAMP:
            aasist_audio = aasist_audio[:AASIST_NB_SAMP]
        elif len(aasist_audio) < AASIST_NB_SAMP:
            aasist_audio = np.pad(aasist_audio, (0, AASIST_NB_SAMP - len(aasist_audio)))
        a_tensor = torch.from_numpy(aasist_audio).float().unsqueeze(0).to(device)
        aasist_hidden, _ = aasist_model(a_tensor)
        aasist_emb = aasist_hidden.squeeze(0).cpu().numpy()

        combined = np.concatenate([w2v_emb, aasist_emb])
        all_embs.append(torch.from_numpy(combined).float())
        all_labels.append(label)

    X = torch.stack(all_embs).to(device)
    y = torch.tensor(all_labels, dtype=torch.float32, device=device)
    logger.info(f"Extracted {len(all_embs)} embeddings, shape={list(X.shape)}")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(feats)
        loss = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device).unsqueeze(1)
        preds = model(feats)
        total_loss += criterion(preds, labels).item()
        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    output_path,
):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        metrics = f"train_loss={train_loss:.6f}"
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            metrics += f"  val_loss={val_loss:.6f}  val_acc={val_acc:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_path)
                logger.info(f"Epoch {epoch}: saved best model (val_loss={val_loss:.6f}, acc={val_acc:.4f})")
        else:
            torch.save(model.state_dict(), output_path)

        logger.info(f"Epoch {epoch}/{epochs}: {metrics}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Meta-MLP trainer for ASVspoof 2019.\n"
        "Downloads Wav2Vec2 from HF Hub automatically. Requires pre-downloaded AASIST weights."
    )
    parser.add_argument("--train-csv", required=True, help="ASVspoof train protocol .trl.txt")
    parser.add_argument("--eval-csv", help="ASVspoof eval protocol .trl.txt (optional validation)")
    parser.add_argument("--audio-dir", required=True, help="Directory containing train FLAC files")
    parser.add_argument("--eval-audio-dir", help="Directory containing eval FLAC files")
    parser.add_argument("--aasist-weights", default="aasist_backbone.pt", help="Path to AASIST backbone weights")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for debugging")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="meta_mlp.pt", help="Output path for trained MLP weights")
    parser.add_argument("--w2v-model-id", default="Vansh180/deepfake-audio-wav2vec2", help="HF model ID for Wav2Vec2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load Wav2Vec2
    logger.info(f"Loading Wav2Vec2: {args.w2v_model_id}")
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    w2v_processor = AutoFeatureExtractor.from_pretrained(args.w2v_model_id)
    w2v_model = AutoModelForAudioClassification.from_pretrained(args.w2v_model_id)
    w2v_model = w2v_model.to(device)
    w2v_model.eval()
    logger.info("Wav2Vec2 loaded")

    # Load AASIST
    aasist_path = Path(args.aasist_weights)
    logger.info(f"Loading AASIST from {aasist_path}")
    aasist_model = AASISTModel()
    if aasist_path.exists():
        state = torch.load(aasist_path, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        aasist_model.load_state_dict(state, strict=False)
    else:
        logger.warning(f"AASIST weights not found at {aasist_path}, using random init")
    aasist_model = aasist_model.to(device)
    aasist_model.eval()
    logger.info("AASIST loaded")

    # Load data
    logger.info("Loading training data...")
    train_pairs = load_asvspoof_csv(args.train_csv, args.audio_dir, args.max_samples)
    logger.info(f"Extracting train embeddings ({len(train_pairs)} samples)...")
    X_train, y_train = extract_embeddings(
        train_pairs, w2v_model, w2v_processor, aasist_model, device, desc="Train embeddings"
    )

    val_loader = None
    if args.eval_csv and args.eval_audio_dir:
        logger.info("Loading eval data...")
        eval_pairs = load_asvspoof_csv(args.eval_csv, args.eval_audio_dir, args.max_samples)
        logger.info(f"Extracting eval embeddings ({len(eval_pairs)} samples)...")
        X_eval, y_eval = extract_embeddings(
            eval_pairs, w2v_model, w2v_processor, aasist_model, device, desc="Eval embeddings"
        )
        val_loader = DataLoader(EmbeddingDataset(X_eval, y_eval), batch_size=args.batch_size)

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    # Train
    model = MetaMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MetaMLP: {n_params:,} parameters, dual-branch: W2V2 768->128 + AASIST 160->64 -> 192->64->1")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train(model, train_loader, val_loader, args.epochs, args.lr, device, output_path)

    # Save final
    torch.save(model.state_dict(), output_path)
    logger.info(f"Final model saved to {output_path.resolve()}")
    print(f"\nDone. Copy '{output_path}' to the main project's weights/ directory.")


if __name__ == "__main__":
    main()
