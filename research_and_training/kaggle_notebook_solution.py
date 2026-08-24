"""
Self-contained meta-classifier trainer for ASVspoof 2019 on Kaggle.
Copy-paste cells into your notebook. No imports from app.* needed.
"""

# ============================================================
# CELL 1 — Imports & Spark (match your existing setup)
# ============================================================
import os
import glob
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("ASVspoof-Audio-Pipeline") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# ============================================================
# CELL 2 — Paths (adjust to your Kaggle dataset mount)
# ============================================================
BASE_PATH = "/kaggle/input/asvpoof-2019-dataset/LA/LA"

LA_TRAIN_AUDIO = os.path.join(BASE_PATH, "ASVspoof2019_LA_train", "flac")
LA_DEV_AUDIO   = os.path.join(BASE_PATH, "ASVspoof2019_LA_dev", "flac")
LA_EVAL_AUDIO  = os.path.join(BASE_PATH, "ASVspoof2019_LA_eval", "flac")

TRAIN_PROTOCOL = os.path.join(
    BASE_PATH,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)
DEV_PROTOCOL = os.path.join(
    BASE_PATH,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.dev.trl.txt"
)

# Where to save the trained MLP
OUTPUT_PATH = "/kaggle/working/meta_mlp.pt"

# AASIST weights — upload aasist_backbone.pt as a Kaggle dataset or
# place at this path. If missing, the model uses random init.
AASIST_WEIGHTS = "/kaggle/input/aasist-weights/aasist_backbone.pt"

SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = SAMPLE_RATE * 60
AASIST_NB_SAMP = 64600

# ============================================================
# CELL 3 — AASIST architecture (official clovaai/aasist, MIT)
# ============================================================

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

# ============================================================
# CELL 4 — Meta-MLP classifier
# ============================================================

class MetaMLP(nn.Module):
    def __init__(self, w2v_dim=768, aasist_dim=160, joint_dim=64):
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

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        w2v = x[:, :768]
        aasist = x[:, 768:]
        w2v_out = self.w2v_branch(w2v)
        aasist_out = self.aasist_branch(aasist)
        fused = torch.cat([w2v_out, aasist_out], dim=-1)
        return self.classifier(fused)

# ============================================================
# CELL 5 — Load Wav2Vec2 + AASIST (internet required for W2V2)
# ============================================================

# --- Wav2Vec2 + AASIST (both on CPU) ---
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

device = torch.device("cpu")
print("Device:", device)

print("Loading Wav2Vec2: Vansh180/deepfake-audio-wav2vec2")
w2v_processor = AutoFeatureExtractor.from_pretrained("Vansh180/deepfake-audio-wav2vec2")
w2v_model = AutoModelForAudioClassification.from_pretrained("Vansh180/deepfake-audio-wav2vec2")
w2v_model = w2v_model.to(device)
w2v_model.eval()
print("Wav2Vec2 loaded")

print(f"Loading AASIST from {AASIST_WEIGHTS}")
aasist_model = AASISTModel()
if os.path.exists(AASIST_WEIGHTS):
    state = torch.load(AASIST_WEIGHTS, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    aasist_model.load_state_dict(state, strict=False)
    print("AASIST loaded with pretrained weights")
else:
    print("WARNING: AASIST weights not found — using random init (poor performance)")
aasist_model = aasist_model.to(device)
aasist_model.eval()

# ============================================================
# CELL 6 — Extract 928-dim embeddings (W2V2 768 + AASIST 160)
# ============================================================

def extract_embedding(audio_path):
    """Load audio, return (768-dim w2v_emb, 160-dim aasist_emb, label)."""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio = audio[:MAX_AUDIO_SAMPLES].astype(np.float32)

    # Wav2Vec2 embedding
    max_len = min(len(audio), SAMPLE_RATE * 30)
    inputs = w2v_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = w2v_model(**inputs, output_hidden_states=True)
        w2v_emb = outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()

    # AASIST embedding (always on CPU)
    a = audio.copy()
    if len(a) > AASIST_NB_SAMP:
        a = a[:AASIST_NB_SAMP]
    elif len(a) < AASIST_NB_SAMP:
        a = np.pad(a, (0, AASIST_NB_SAMP - len(a)))
    a_tensor = torch.from_numpy(a).float().unsqueeze(0)
    with torch.no_grad():
        aasist_hidden, _ = aasist_model(a_tensor)
        aasist_emb = aasist_hidden.squeeze(0).numpy()

    return np.concatenate([w2v_emb, aasist_emb])

# Load protocol
protocol_cols = ["speaker_id", "file_id", "dash", "system_id", "label"]
train_df = pd.read_csv(TRAIN_PROTOCOL, sep=r"\s+", header=None, names=protocol_cols)
train_df["audio_path"] = train_df["file_id"].apply(lambda x: os.path.join(LA_TRAIN_AUDIO, f"{x}.flac"))
train_df["label_num"] = (train_df["label"] == "spoof").astype(float)

# Subsample to 10K (8K spoof + 2K bonafide) for faster training
spoof_df = train_df[train_df["label"] == "spoof"].sample(8000, random_state=42)
bonafide_df = train_df[train_df["label"] == "bonafide"].sample(2000, random_state=42)
train_df = pd.concat([spoof_df, bonafide_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Training samples: {len(train_df)}")
print(train_df["label"].value_counts())

# Extract embeddings (~10K samples)
X_list, y_list = [], []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting embeddings"):
    try:
        emb = extract_embedding(row["audio_path"])
        X_list.append(emb)
        y_list.append(row["label_num"])
    except Exception as e:
        print(f"Failed on {row['file_id']}: {e}")

if len(X_list) == 0:
    raise RuntimeError("All samples failed — check the errors above.")
X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"Embeddings shape: {X.shape}, Labels: {y.shape} ({y.sum():.0f} spoof / {(len(y)-y.sum()):.0f} bonafide)")

# Save embeddings for later reuse
np.save("/kaggle/working/embeddings_X.npy", X)
np.save("/kaggle/working/embeddings_y.npy", y)
print("Embeddings saved to /kaggle/working/")

# ============================================================
# CELL 7 — Train Meta-MLP
# ============================================================

class EmbeddingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Split train/val
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=64)

model = MetaMLP().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_val_loss = float("inf")
epochs = 20

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(feats), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device).unsqueeze(1)
            preds = model(feats)
            val_loss += criterion(preds, labels).item()
            correct += ((preds > 0.5).float() == labels).sum().item()
            total += labels.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUTPUT_PATH)

    print(f"Epoch {epoch:2d}/{epochs}  train_loss={train_loss/len(train_loader):.6f}  val_loss={val_loss:.6f}  val_acc={val_acc:.4f}")

torch.save(model.state_dict(), OUTPUT_PATH)
print(f"\nDone! Model saved to {OUTPUT_PATH}")

# ============================================================
# CELL 8 — Download the trained meta_mlp.pt
# ============================================================
from IPython.display import FileLink
FileLink(OUTPUT_PATH)
