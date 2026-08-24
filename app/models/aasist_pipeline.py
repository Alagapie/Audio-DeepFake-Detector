import logging
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings

logger = logging.getLogger(__name__)


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
        master_att = F.softmax(torch.matmul(torch.tanh(self.att_projM(master_e)), self.att_weightM).squeeze(-1) / self.temp, dim=-1)
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


class AASISTPipeline:
    def __init__(self):
        self._name = "aasist"
        self._model: AASISTModel | None = None

    @property
    def name(self) -> str:
        return self._name

    async def load(self) -> None:
        path = settings.aasist_pytorch_path
        logger.info(f"Loading AASIST from {path}")
        self._model = AASISTModel()
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            self._model.load_state_dict(state, strict=False)
        else:
            logger.warning(f"AASIST weights not found at {path}, using random init")
        self._model.eval()
        logger.info("AASIST model loaded")

    async def unload(self) -> None:
        self._model = None

    async def infer(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        if self._model is None:
            raise RuntimeError("AASIST model not loaded")
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if audio.shape[1] > settings.aasist_nb_samp:
            audio = audio[:, :settings.aasist_nb_samp]
        elif audio.shape[1] < settings.aasist_nb_samp:
            audio = np.pad(audio, ((0, 0), (0, settings.aasist_nb_samp - audio.shape[1])))
        x = torch.from_numpy(audio).float()
        with torch.no_grad():
            _, logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs[0, 1].item())
        return confidence, logits[0].numpy()

    async def infer_with_embedding(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        if self._model is None:
            raise RuntimeError("AASIST model not loaded")
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if audio.shape[1] > settings.aasist_nb_samp:
            audio = audio[:, :settings.aasist_nb_samp]
        elif audio.shape[1] < settings.aasist_nb_samp:
            audio = np.pad(audio, ((0, 0), (0, settings.aasist_nb_samp - audio.shape[1])))
        x = torch.from_numpy(audio).float()
        with torch.no_grad():
            last_hidden, logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs[0, 1].item())
        return confidence, last_hidden[0].numpy()
