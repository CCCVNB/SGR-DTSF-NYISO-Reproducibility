# -*- coding: utf-8 -*-
"""
Dual-scale baseline experiments for two-and-a-half-year H24-to-P24 load forecasting.

Chronological split:
    Train      : 2016-01-01 <= t < 2017-01-01
    Validation : 2017-01-01 <= t < 2017-07-01
    Test       : 2017-07-01 <= t < 2018-07-01

Main changes relative to the original PG-DTSF implementation:
1) A strong dual-resolution base predictor uses both fast load dynamics and slow context.
2) The slow vector is encoded by an MLP instead of a length-one GRU.
3) The uncertainty head is trained with a heteroscedastic Gaussian objective.
4) Current difficulty and historical error memory have consistent, separate representations.
5) A horizon-wise gate and adaptive residual scale control semantic correction.
6) The semantic residual head is zero-initialized so training starts from the strong base predictor.
7) Multiple diagnostic, ablation, regularization, and eta-sensitivity experiment groups are supported.
8) Evaluation is streaming and memory-safe for million-sample datasets.

This version includes dual-scale variants and the fusion comparators reported in the paper.
The script consumes previously prepared per-zone arrays. Raw-data preprocessing and dataset construction are intentionally outside this repository.
"""

import os
import json
import time
import math
import random
import argparse
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Configuration
# =========================================================

@dataclass
class Config:
    # ---- Root containing already-prepared model inputs ----
    save_root: str = r"./data/processed"

    # ---- Multi-year range and chronological split ----
    start_date: str = "2016-01-01"
    end_date_exclusive: str = "2018-07-01"
    train_start: str = "2016-01-01"
    train_end_exclusive: str = "2017-01-01"
    val_start: str = "2017-01-01"
    val_end_exclusive: str = "2017-07-01"
    test_start: str = "2017-07-01"
    test_end_exclusive: str = "2018-07-01"

    # ---- H24 -> P24 setting at 5-min resolution ----
    hist_len: int = 288
    pred_len: int = 288

    # ---- Training ----
    seeds: Tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- Model dimensions ----
    fast_dim: int = 3
    slow_dim: int = 13
    hidden_dim: int = 128
    fusion_dim: int = 128
    zone_emb_dim: int = 16
    dropout: float = 0.1
    n_heads: int = 4
    shared_dim: int = 48
    specific_dim: int = 48
    error_state_dim: int = 8

    # ---- PG-DTSF loss weights ----
    fast_loss_weight: float = 0.2
    align_loss_weight: float = 5e-4
    sep_loss_weight: float = 1e-4
    unc_loss_weight: float = 1e-4
    rampup_warmup: int = 3
    rampup_len: int = 5
    error_memory_momentum: float = 0.95
    difficulty_gamma: float = 0.5
    semantic_residual_scale: float = 0.5

    # ---- PG-DTSF-v2 objective and adaptive correction ----
    base_loss_weight: float = 0.05
    residual_loss_weight: float = 0.10
    hetero_unc_loss_weight: float = 0.01
    correction_l2_weight: float = 1e-4
    v2_align_loss_weight: float = 1e-5
    v2_sep_loss_weight: float = 1e-5
    eta_min: float = 0.05
    eta_max: float = 1.00
    eta_init: float = 0.30

    # ---- Run control ----
    run_experiments: bool = True
    save_predictions: bool = False
    max_train_samples: int = 0  # 0 means full; useful for debug
    max_eval_samples: int = 0   # 0 means full
    sample_mode: str = "uniform"  # first | uniform | random, used only when max_*_samples > 0
    experiment_tag: str = "pg_dtsf_v2_multigroup"
    resume: bool = False
    use_amp: bool = False

    @property
    def save_dir(self) -> str:
        return os.path.join(
            self.save_root,
            "two_half_year_201601_201807_per_zone_scalers_H288_P288"
        )

    @property
    def run_dir(self) -> str:
        return os.path.join(
            self.save_dir,
            f"H288_P288_{self.experiment_tag}"
        )


VALID_ZONES = [
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL",
    "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"
]




# =========================================================
# 2. General utilities
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def zone_safe_name(zone: str) -> str:
    return zone.replace(".", "").replace(" ", "_")
















def load_target_scalers(save_dir: str) -> Dict[str, Tuple[float, float]]:
    scaler_dir = os.path.join(save_dir, "scalers_by_zone")
    out = {}
    for zone in VALID_ZONES:
        zsafe = zone_safe_name(zone)
        zdir = os.path.join(scaler_dir, zsafe)
        mean_path = os.path.join(zdir, "target_mean.npy")
        scale_path = os.path.join(zdir, "target_scale.npy")
        if os.path.exists(mean_path) and os.path.exists(scale_path):
            out[zone] = (float(np.load(mean_path)[0]), float(np.load(scale_path)[0]))
    return out


# =========================================================
# 3. Data processing
# =========================================================























# =========================================================
# 4. Dataset and metrics
# =========================================================

class ZoneWindowDataset(Dataset):
    def __init__(self, cfg: Config, split: str, max_samples: int = 0):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg = cfg
        self.split = split
        self.base_dir = os.path.join(cfg.save_dir, f"{split}_by_zone")
        self.scalers = load_target_scalers(cfg.save_dir)
        Xf_list, Xs_list, Y_list, Z_list, T_list, Mean_list, Scale_list = [], [], [], [], [], [], []
        for zid, zone in enumerate(VALID_ZONES):
            zsafe = zone_safe_name(zone)
            xfp = os.path.join(self.base_dir, f"{split}_Xf_{zsafe}.npy")
            xsp = os.path.join(self.base_dir, f"{split}_Xs_{zsafe}.npy")
            yp = os.path.join(self.base_dir, f"{split}_Y_{zsafe}.npy")
            tp = os.path.join(self.base_dir, f"{split}_T_{zsafe}.npy")
            if not (os.path.exists(xfp) and os.path.exists(xsp) and os.path.exists(yp)):
                continue
            Xf = np.load(xfp).astype(np.float32)
            Xs = np.load(xsp).astype(np.float32)
            Y = np.load(yp).astype(np.float32)
            T = np.load(tp, allow_pickle=True) if os.path.exists(tp) else np.arange(Xf.shape[0])
            if max_samples and max_samples > 0 and Xf.shape[0] > max_samples:
                if cfg.sample_mode == "uniform":
                    idx = np.linspace(0, Xf.shape[0] - 1, num=max_samples, dtype=np.int64)
                elif cfg.sample_mode == "random":
                    rng = np.random.default_rng(2026 + zid)
                    idx = np.sort(rng.choice(Xf.shape[0], size=max_samples, replace=False))
                else:
                    idx = np.arange(max_samples, dtype=np.int64)
                Xf = Xf[idx]
                Xs = Xs[idx]
                Y = Y[idx]
                T = T[idx]
            mean, scale = self.scalers[zone]
            Xf_list.append(Xf)
            Xs_list.append(Xs)
            Y_list.append(Y)
            Z_list.append(np.full((Xf.shape[0],), zid, dtype=np.int64))
            T_list.append(T)
            Mean_list.append(np.full((Xf.shape[0],), mean, dtype=np.float32))
            Scale_list.append(np.full((Xf.shape[0],), scale, dtype=np.float32))
        if not Xf_list:
            raise RuntimeError(f"No samples found for split={split} under {self.base_dir}")
        self.Xf = np.concatenate(Xf_list, axis=0)
        self.Xs = np.concatenate(Xs_list, axis=0)
        self.Y = np.concatenate(Y_list, axis=0)
        self.Z = np.concatenate(Z_list, axis=0)
        self.T = np.concatenate(T_list, axis=0)
        self.target_mean = np.concatenate(Mean_list, axis=0)
        self.target_scale = np.concatenate(Scale_list, axis=0)
        print(f"[DATASET] {split}: Xf={self.Xf.shape}, Xs={self.Xs.shape}, Y={self.Y.shape}")

    def __len__(self):
        return self.Xf.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x_fast": torch.from_numpy(self.Xf[idx]),
            "x_slow": torch.from_numpy(self.Xs[idx]),
            "y": torch.from_numpy(self.Y[idx]),
            "zone": torch.tensor(self.Z[idx], dtype=torch.long),
            "target_mean": torch.tensor(self.target_mean[idx], dtype=torch.float32),
            "target_scale": torch.tensor(self.target_scale[idx], dtype=torch.float32),
            "timestamp": str(self.T[idx])
        }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def inverse_target(y_scaled: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # y_scaled: [B,H], mean/scale: [B]
    return y_scaled * scale.unsqueeze(-1) + mean.unsqueeze(-1)


class OnlineRegressionMetrics:
    """Memory-safe streaming accumulator for MAE, RMSE, MAPE and R2.

    Only scalar float64 statistics are retained. Large prediction arrays are never
    converted to full-size float64 copies.
    """

    def __init__(self, eps: float = 1.0):
        self.eps = float(eps)
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_ape = 0.0
        self.sum_true = 0.0
        self.sum_true_sq = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred64 = np.asarray(pred, dtype=np.float64).reshape(-1)
        true64 = np.asarray(true, dtype=np.float64).reshape(-1)
        if pred64.size != true64.size:
            raise ValueError(f"Prediction/target size mismatch: {pred64.size} vs {true64.size}")
        if pred64.size == 0:
            return
        err = pred64 - true64
        abs_err = np.abs(err)
        self.n += int(true64.size)
        self.sum_abs += float(np.sum(abs_err, dtype=np.float64))
        self.sum_sq += float(np.sum(err * err, dtype=np.float64))
        self.sum_ape += float(np.sum(abs_err / np.maximum(np.abs(true64), self.eps), dtype=np.float64))
        self.sum_true += float(np.sum(true64, dtype=np.float64))
        self.sum_true_sq += float(np.sum(true64 * true64, dtype=np.float64))

    def compute(self) -> Dict[str, float]:
        if self.n <= 0:
            return {"MAE": float("nan"), "RMSE": float("nan"),
                    "MAPE": float("nan"), "R2": float("nan")}
        mae = self.sum_abs / self.n
        mse = self.sum_sq / self.n
        rmse = math.sqrt(max(mse, 0.0))
        mape = self.sum_ape / self.n * 100.0
        ss_tot = self.sum_true_sq - (self.sum_true * self.sum_true) / self.n
        r2 = 1.0 - self.sum_sq / (ss_tot + 1e-12)
        return {"MAE": float(mae), "RMSE": float(rmse),
                "MAPE": float(mape), "R2": float(r2)}


def compute_metrics_np(pred: np.ndarray, true: np.ndarray, eps: float = 1.0,
                       row_chunk_size: int = 4096) -> Dict[str, float]:
    """Memory-safe NumPy metrics evaluated in row chunks."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    meter = OnlineRegressionMetrics(eps=eps)
    if pred.ndim == 0:
        meter.update(pred.reshape(1), true.reshape(1))
    elif pred.ndim == 1:
        for start in range(0, pred.shape[0], row_chunk_size):
            end = min(start + row_chunk_size, pred.shape[0])
            meter.update(pred[start:end], true[start:end])
    else:
        for start in range(0, pred.shape[0], row_chunk_size):
            end = min(start + row_chunk_size, pred.shape[0])
            meter.update(pred[start:end], true[start:end])
    return meter.compute()


def persistence_metrics(ds: ZoneWindowDataset, row_chunk_size: int = 4096) -> Dict[str, float]:
    """Previous-day persistence metrics without materializing full inverse-scaled arrays."""
    meter = OnlineRegressionMetrics(eps=1.0)
    n = len(ds)
    for start in range(0, n, row_chunk_size):
        end = min(start + row_chunk_size, n)
        pred_scaled = ds.Xf[start:end, :, 0]
        true_scaled = ds.Y[start:end]
        mean = ds.target_mean[start:end, None]
        scale = ds.target_scale[start:end, None]
        pred_orig = pred_scaled * scale + mean
        true_orig = true_scaled * scale + mean
        meter.update(pred_orig, true_orig)
    return meter.compute()


# =========================================================
# 5. Building blocks
# =========================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1, layers: int = 2):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(max(1, layers - 1)):
            mods += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_ch)
        self.pad = pad

    def forward(self, x):  # [B,C,L]
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]
        y = self.dropout(F.gelu(y))
        y = self.norm(y)
        return y + self.proj(x)


class FastGRUEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.drop(F.gelu(self.proj(x)))
        out, h = self.gru(z)
        out = self.norm(out)
        h = self.norm(h[-1])
        return out, h


# =========================================================
# 6. Models
# =========================================================

class BaseForecastModel(nn.Module):
    def __init__(self, cfg: Config, name: str):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.pred_len = cfg.pred_len
        self.num_zones = len(VALID_ZONES)
        self.zone_emb = nn.Embedding(self.num_zones, cfg.zone_emb_dim)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        raise NotImplementedError

    def compute_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor, epoch: int = 0):
        return F.mse_loss(out["pred"], y), {"pred_loss": float(F.mse_loss(out["pred"], y).detach().cpu())}

    def update_memory(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor):
        return


class BaselineRNN(BaseForecastModel):
    def __init__(self, cfg: Config, rnn_type: str = "gru"):
        super().__init__(cfg, rnn_type)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.in_proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.rnn = rnn_cls(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = F.gelu(self.in_proj(x_fast))
        out, h = self.rnn(z)
        if isinstance(h, tuple):
            h = h[0]
        h = self.norm(h[-1])
        e = self.zone_emb(zone)
        pred = self.head(torch.cat([h, e], dim=-1))
        return {"pred": pred}


class BaselineTCN(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "tcn")
        self.in_proj = nn.Conv1d(cfg.fast_dim, cfg.hidden_dim, 1)
        self.blocks = nn.Sequential(
            TemporalConvBlock(cfg.hidden_dim, cfg.hidden_dim, 3, 1, cfg.dropout),
            TemporalConvBlock(cfg.hidden_dim, cfg.hidden_dim, 3, 2, cfg.dropout),
            TemporalConvBlock(cfg.hidden_dim, cfg.hidden_dim, 3, 4, cfg.dropout),
        )
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = x_fast.transpose(1, 2)
        z = self.in_proj(z)
        z = self.blocks(z)
        h = z[:, :, -1]
        e = self.zone_emb(zone)
        pred = self.head(torch.cat([h, e], dim=-1))
        return {"pred": pred}


class BaselineTransformer(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "transformer")
        self.proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, cfg.hist_len + 8)
        layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.hidden_dim * 4,
                                           dropout=cfg.dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = self.pos(self.proj(x_fast))
        z = self.encoder(z)
        h = z.mean(dim=1)
        pred = self.head(torch.cat([h, self.zone_emb(zone)], dim=-1))
        return {"pred": pred}


class BaselinePatchTST(BaseForecastModel):
    def __init__(self, cfg: Config, patch_len: int = 16, stride: int = 8):
        super().__init__(cfg, "patchtst")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_dim = patch_len * cfg.fast_dim
        self.proj = nn.Linear(self.patch_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, 200)
        layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.hidden_dim * 4,
                                           dropout=cfg.dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        # x_fast: [B,L,F] -> patches [B,P,patch_len*F]
        patches = x_fast.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B,P,F,patch_len]
        patches = patches.transpose(2, 3).contiguous().flatten(2)
        z = self.pos(self.proj(patches))
        z = self.encoder(z)
        h = z.mean(dim=1)
        pred = self.head(torch.cat([h, self.zone_emb(zone)], dim=-1))
        return {"pred": pred}


class ProbSparseSelfAttention(nn.Module):
    """A compact attention block used in the Informer baseline."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Compact implementation uses full attention for reproducibility.
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(y))
        y = self.ff(x)
        return self.norm2(x + self.drop(y))


class BaselineInformer(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "informer")
        self.proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, cfg.hist_len + 8)
        self.layers = nn.ModuleList([ProbSparseSelfAttention(cfg.hidden_dim, cfg.n_heads, cfg.dropout) for _ in range(2)])
        self.distill = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = self.pos(self.proj(x_fast))
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i == 0:
                z = self.distill(z.transpose(1, 2)).transpose(1, 2)
        h = z.mean(dim=1)
        pred = self.head(torch.cat([h, self.zone_emb(zone)], dim=-1))
        return {"pred": pred}


class BaselineITransformer(BaseForecastModel):
    """Compact iTransformer-style baseline: attention over variate tokens."""
    def __init__(self, cfg: Config):
        super().__init__(cfg, "itransformer")
        self.var_embed = nn.Linear(cfg.hist_len, cfg.hidden_dim)
        self.var_type = nn.Parameter(torch.randn(1, cfg.fast_dim, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.hidden_dim * 4,
                                           dropout=cfg.dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = MLP(cfg.hidden_dim * cfg.fast_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        # [B,L,F] -> [B,F,L], each variable is a token.
        z = x_fast.transpose(1, 2)
        z = self.var_embed(z) + self.var_type
        z = self.encoder(z)
        h = z.flatten(1)
        pred = self.head(torch.cat([h, self.zone_emb(zone)], dim=-1))
        return {"pred": pred}


class BaselineTimeMixer(BaseForecastModel):
    """Compact TimeMixer-style baseline with multi-scale decomposition."""
    def __init__(self, cfg: Config):
        super().__init__(cfg, "timemixer")
        self.scales = [1, 2, 4]
        self.proj = nn.ModuleList([nn.Linear(cfg.fast_dim, cfg.hidden_dim) for _ in self.scales])
        self.mix = MLP(cfg.hidden_dim * len(self.scales) + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def _downsample(self, x, scale):
        if scale == 1:
            return x
        # Average pooling along time.
        b, l, f = x.shape
        cut = (l // scale) * scale
        x = x[:, :cut]
        return x.reshape(b, cut // scale, scale, f).mean(dim=2)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        reps = []
        for scale, proj in zip(self.scales, self.proj):
            xs = self._downsample(x_fast, scale)
            trend = xs.mean(dim=1, keepdim=True)
            seasonal = xs - trend
            h = F.gelu(proj(seasonal)).mean(dim=1) + F.gelu(proj(trend.squeeze(1)))
            reps.append(h)
        h = torch.cat(reps + [self.zone_emb(zone)], dim=-1)
        pred = self.mix(h)
        return {"pred": pred}


class BaselineTimeXer(BaseForecastModel):
    """Compact TimeXer-style baseline: patch temporal tokens with exogenous slow-context token."""
    def __init__(self, cfg: Config, patch_len: int = 16, stride: int = 8):
        super().__init__(cfg, "timexer")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len * cfg.fast_dim, cfg.hidden_dim)
        self.slow_proj = nn.Linear(cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, 200)
        self.cross = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.self_block = ProbSparseSelfAttention(cfg.hidden_dim, cfg.n_heads, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        patches = x_fast.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.transpose(2, 3).contiguous().flatten(2)
        ptok = self.pos(self.patch_proj(patches))
        e = self.zone_emb(zone)
        stok = self.slow_proj(torch.cat([x_slow, e], dim=-1)).unsqueeze(1)
        z, _ = self.cross(ptok, stok, stok, need_weights=False)
        z = self.self_block(ptok + z)
        h = z.mean(dim=1)
        pred = self.head(torch.cat([h, e], dim=-1))
        return {"pred": pred}



# =========================================================
# 6B. Dual-scale versions of standard temporal baselines
# =========================================================

class SlowContextMLP(nn.Module):
    """Encode the daily/weather-calendar slow context vector into a hidden state."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout, layers=2)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x_slow):
        return self.norm(self.net(x_slow))


class DualScaleRNN(BaseForecastModel):
    """GRU/LSTM baseline with explicit fast-history and slow-context fusion."""
    def __init__(self, cfg: Config, rnn_type: str = "gru"):
        super().__init__(cfg, f"dual_{rnn_type}")
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.in_proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.rnn = rnn_cls(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.fast_norm = nn.LayerNorm(cfg.hidden_dim)
        self.slow = SlowContextMLP(cfg)
        self.fuse = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = F.gelu(self.in_proj(x_fast))
        _, h = self.rnn(z)
        if isinstance(h, tuple):
            h = h[0]
        hf = self.fast_norm(h[-1])
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        fused = self.fuse(torch.cat([hf, hs, e], dim=-1))
        pred = self.head(torch.cat([fused, e], dim=-1))
        return {"pred": pred, "fast_repr": hf, "slow_repr": hs}


class DualScaleTCN(BaseForecastModel):
    """TCN baseline with explicit slow-context fusion."""
    def __init__(self, cfg: Config):
        super().__init__(cfg, "dual_tcn")
        self.in_proj = nn.Conv1d(cfg.fast_dim, cfg.hidden_dim, 1)
        self.blocks = nn.Sequential(
            TemporalConvBlock(cfg.hidden_dim, cfg.hidden_dim, 3, 1, cfg.dropout),
            TemporalConvBlock(cfg.hidden_dim, cfg.hidden_dim, 3, 2, cfg.dropout),
            TemporalConvBlock(cfg.hidden_dim, cfg.hidden_dim, 3, 4, cfg.dropout),
        )
        self.slow = SlowContextMLP(cfg)
        self.fuse = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = self.in_proj(x_fast.transpose(1, 2))
        z = self.blocks(z)
        hf = z[:, :, -1]
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        fused = self.fuse(torch.cat([hf, hs, e], dim=-1))
        pred = self.head(torch.cat([fused, e], dim=-1))
        return {"pred": pred, "fast_repr": hf, "slow_repr": hs}


class DualScaleTransformer(BaseForecastModel):
    """Transformer encoder baseline with explicit slow-context fusion."""
    def __init__(self, cfg: Config):
        super().__init__(cfg, "dual_transformer")
        self.proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, cfg.hist_len + 8)
        layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.hidden_dim * 4,
                                           dropout=cfg.dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.slow = SlowContextMLP(cfg)
        self.fuse = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = self.pos(self.proj(x_fast))
        z = self.encoder(z)
        hf = z.mean(dim=1)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        fused = self.fuse(torch.cat([hf, hs, e], dim=-1))
        pred = self.head(torch.cat([fused, e], dim=-1))
        return {"pred": pred, "fast_repr": hf, "slow_repr": hs}


class DualScalePatchTST(BaseForecastModel):
    """PatchTST-style baseline with explicit slow-context fusion."""
    def __init__(self, cfg: Config, patch_len: int = 16, stride: int = 8):
        super().__init__(cfg, "dual_patchtst")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_dim = patch_len * cfg.fast_dim
        self.proj = nn.Linear(self.patch_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, 200)
        layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.hidden_dim * 4,
                                           dropout=cfg.dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.slow = SlowContextMLP(cfg)
        self.fuse = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        patches = x_fast.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.transpose(2, 3).contiguous().flatten(2)
        z = self.pos(self.proj(patches))
        z = self.encoder(z)
        hf = z.mean(dim=1)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        fused = self.fuse(torch.cat([hf, hs, e], dim=-1))
        pred = self.head(torch.cat([fused, e], dim=-1))
        return {"pred": pred, "fast_repr": hf, "slow_repr": hs}


class DualScaleInformer(BaseForecastModel):
    """Informer-style baseline with explicit slow-context fusion."""
    def __init__(self, cfg: Config):
        super().__init__(cfg, "dual_informer")
        self.proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.pos = PositionalEncoding(cfg.hidden_dim, cfg.hist_len + 8)
        self.layers = nn.ModuleList([
            ProbSparseSelfAttention(cfg.hidden_dim, cfg.n_heads, cfg.dropout) for _ in range(2)
        ])
        self.distill = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.slow = SlowContextMLP(cfg)
        self.fuse = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        z = self.pos(self.proj(x_fast))
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i == 0:
                z = self.distill(z.transpose(1, 2)).transpose(1, 2)
        hf = z.mean(dim=1)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        fused = self.fuse(torch.cat([hf, hs, e], dim=-1))
        pred = self.head(torch.cat([fused, e], dim=-1))
        return {"pred": pred, "fast_repr": hf, "slow_repr": hs}


class DualScaleITransformer(BaseForecastModel):
    """iTransformer-style baseline with an additional slow-context token.

    Fast variables are treated as variate tokens as in iTransformer. The slow daily
    weather-calendar vector is encoded as an extra exogenous token and participates
    in the same token-level attention block.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg, "dual_itransformer")
        self.var_embed = nn.Linear(cfg.hist_len, cfg.hidden_dim)
        self.var_type = nn.Parameter(torch.randn(1, cfg.fast_dim, cfg.hidden_dim) * 0.02)
        self.slow = SlowContextMLP(cfg)
        self.slow_type = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.hidden_dim * 4,
                                           dropout=cfg.dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = MLP(cfg.hidden_dim * (cfg.fast_dim + 1) + cfg.zone_emb_dim,
                        cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        # Fast variable tokens: [B,L,F] -> [B,F,L] -> [B,F,H]
        fast_tokens = self.var_embed(x_fast.transpose(1, 2)) + self.var_type
        slow_token = self.slow(x_slow).unsqueeze(1) + self.slow_type
        z = torch.cat([fast_tokens, slow_token], dim=1)
        z = self.encoder(z)
        h = z.flatten(1)
        pred = self.head(torch.cat([h, self.zone_emb(zone)], dim=-1))
        return {"pred": pred, "token_repr": z, "slow_repr": z[:, -1]}

class DirectFusion(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "direct_fusion")
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        _, hf = self.fast(x_fast)
        hs = self.slow(x_slow)
        pred = self.head(torch.cat([hf, hs, self.zone_emb(zone)], dim=-1))
        return {"pred": pred}


class GatedFusion(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "gated_fusion")
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.gate = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        _, hf = self.fast(x_fast)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        g = torch.sigmoid(self.gate(torch.cat([hf, hs, e], dim=-1)))
        h = g * hs + (1.0 - g) * hf
        pred = self.head(torch.cat([h, e], dim=-1))
        return {"pred": pred}


class CrossAttentionFusion(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "cross_attention_fusion")
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow_tok = nn.Linear(cfg.slow_dim, cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.head = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        Hf, hf = self.fast(x_fast)
        st = F.gelu(self.slow_tok(x_slow)).unsqueeze(1)
        q = hf.unsqueeze(1)
        ctx, _ = self.attn(q, st, st, need_weights=False)
        e = self.zone_emb(zone)
        pred = self.head(torch.cat([hf, ctx.squeeze(1), e], dim=-1))
        return {"pred": pred}


class ResidualFusion(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "residual_fusion")
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.fast_head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.res_head = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)
        self.gate = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, 1, cfg.dropout)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        _, hf = self.fast(x_fast)
        e = self.zone_emb(zone)
        base = self.fast_head(torch.cat([hf, e], dim=-1))
        hs = self.slow(x_slow)
        z = torch.cat([hf, hs, e], dim=-1)
        res = self.res_head(z)
        beta = torch.sigmoid(self.gate(z))
        pred = base + 0.5 * beta * res
        return {"pred": pred, "fast_pred": base}


class SharedBottomMTL(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "shared_bottom_mtl")
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.shared = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.heads = nn.ModuleList([MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout) for _ in VALID_ZONES])

    def forward(self, x_fast, x_slow, zone, y_true=None):
        _, hf = self.fast(x_fast)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)
        h = self.shared(torch.cat([hf, hs, e], dim=-1))
        all_preds = torch.stack([head(h) for head in self.heads], dim=1)  # [B,M,H]
        pred = all_preds[torch.arange(zone.shape[0], device=zone.device), zone]
        return {"pred": pred}


class PGDTSFV1(BaseForecastModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg, "pg_dtsf")
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow_proj = nn.Linear(cfg.slow_dim, cfg.hidden_dim)
        self.slow_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.slow_norm = nn.LayerNorm(cfg.hidden_dim)

        self.shared_enc = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.shared_dim, cfg.dropout)
        self.spec_encs = nn.ModuleList([MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.specific_dim, cfg.dropout) for _ in VALID_ZONES])
        self.mem_proj = nn.Linear(cfg.shared_dim + cfg.specific_dim, cfg.hidden_dim)

        self.fast_head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)
        self.unc_head = MLP(cfg.hidden_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

        self.env_proj = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.err_proj = MLP(cfg.error_state_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.query_proj = nn.Linear(cfg.hidden_dim + cfg.zone_emb_dim + cfg.hidden_dim + cfg.hidden_dim, cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.fuse = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim + cfg.hidden_dim + cfg.hidden_dim, cfg.hidden_dim, cfg.fusion_dim, cfg.dropout)
        self.res_head = MLP(cfg.fusion_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)
        self.res_gate = MLP(cfg.hidden_dim + cfg.zone_emb_dim + cfg.hidden_dim + cfg.hidden_dim, cfg.hidden_dim, 1, cfg.dropout)

        self.register_buffer("error_memory", torch.zeros(len(VALID_ZONES), cfg.error_state_dim), persistent=True)

    def _slow_encode(self, x_slow):
        # Current data has one daily slow vector. Use a length-1 sequence for compatibility.
        z = F.gelu(self.slow_proj(x_slow)).unsqueeze(1)
        out, h = self.slow_gru(z)
        out = self.slow_norm(out)
        h = self.slow_norm(h[-1])
        return out, h

    def _specific_encode(self, hs, zone):
        all_specs = torch.stack([enc(hs) for enc in self.spec_encs], dim=1)  # [B,M,D]
        spec = all_specs[torch.arange(zone.shape[0], device=zone.device), zone]
        return spec

    def _difficulty_descriptor(self, fast_pred, unc_logit, last_load):
        dev = torch.abs(fast_pred - last_load.unsqueeze(-1))
        unc = F.softplus(unc_logit)
        stats = torch.stack([
            dev.mean(dim=-1), dev.std(dim=-1), dev[:, 0], dev[:, -1],
            unc.mean(dim=-1), unc.std(dim=-1), unc[:, 0], unc[:, -1]
        ], dim=-1)
        return stats

    def forward(self, x_fast, x_slow, zone, y_true=None):
        Hf, hf = self.fast(x_fast)
        e = self.zone_emb(zone)
        fast_in = torch.cat([hf, e], dim=-1)
        fast_pred = self.fast_head(fast_in)
        unc_logit = self.unc_head(fast_in)
        p = self._difficulty_descriptor(fast_pred, unc_logit, x_fast[:, -1, 0])
        mem = self.error_memory[zone].detach()
        r = self.cfg.difficulty_gamma * p + (1.0 - self.cfg.difficulty_gamma) * mem

        Hs, hs = self._slow_encode(x_slow)
        zc = self.shared_enc(hs)
        zr = self._specific_encode(hs, zone)
        mem_tokens = torch.stack([
            torch.cat([zc, torch.zeros_like(zr)], dim=-1),
            torch.cat([torch.zeros_like(zc), zr], dim=-1)
        ], dim=1)
        kv = self.mem_proj(mem_tokens)  # [B,2,hidden]
        env = self.env_proj(x_slow)
        err = self.err_proj(r)
        q = self.query_proj(torch.cat([hf, e, env, err], dim=-1)).unsqueeze(1)
        ctx, attn_w = self.attn(q, kv, kv, need_weights=True)
        ctx = ctx.squeeze(1)
        fused = self.fuse(torch.cat([ctx, hf, e, env, err], dim=-1))
        res = self.res_head(fused)
        beta = torch.sigmoid(self.res_gate(torch.cat([hf, e, env, err], dim=-1)))
        pred = fast_pred + self.cfg.semantic_residual_scale * beta * res
        return {
            "pred": pred,
            "fast_pred": fast_pred,
            "unc_logit": unc_logit,
            "p_desc": p,
            "zc": zc,
            "zr": zr,
            "attn": attn_w,
            "beta": beta,
        }

    def update_memory(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor):
        with torch.no_grad():
            fp = out.get("fast_pred")
            if fp is None:
                return
            err = torch.abs(fp - y)
            desc = torch.stack([
                err.mean(dim=-1), err.std(dim=-1), err[:, 0], err[:, -1],
                err.mean(dim=-1), err.std(dim=-1), err[:, 0], err[:, -1]
            ], dim=-1)
            for zid in zone.unique():
                mask = zone == zid
                if mask.any():
                    avg = desc[mask].mean(dim=0)
                    self.error_memory[zid] = self.cfg.error_memory_momentum * self.error_memory[zid] + \
                                             (1.0 - self.cfg.error_memory_momentum) * avg

    def _align_loss(self, zc, zone):
        # Simple cross-zone shared-centroid variance loss within a mini-batch.
        centroids = []
        for zid in zone.unique():
            mask = zone == zid
            if mask.sum() > 1:
                centroids.append(zc[mask].mean(dim=0))
        if len(centroids) <= 1:
            return zc.new_tensor(0.0)
        c = torch.stack(centroids, dim=0)
        return ((c - c.mean(dim=0, keepdim=True)) ** 2).mean()

    def _sep_loss(self, zc, zr):
        zc = zc - zc.mean(dim=0, keepdim=True)
        zr = zr - zr.mean(dim=0, keepdim=True)
        cov = zc.T @ zr / max(1, zc.shape[0] - 1)
        return (cov ** 2).mean()

    def _ramp(self, epoch: int):
        if epoch <= self.cfg.rampup_warmup:
            return 0.0
        if epoch >= self.cfg.rampup_warmup + self.cfg.rampup_len:
            return 1.0
        return float(epoch - self.cfg.rampup_warmup) / float(self.cfg.rampup_len)

    def compute_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor, epoch: int = 0):
        pred_loss = F.mse_loss(out["pred"], y)
        fast_loss = F.mse_loss(out["fast_pred"], y)
        align_loss = self._align_loss(out["zc"], zone)
        sep_loss = self._sep_loss(out["zc"], out["zr"])
        unc_loss = (out["unc_logit"] ** 2).mean()
        ramp = self._ramp(epoch)
        loss = pred_loss + self.cfg.fast_loss_weight * fast_loss + \
               ramp * self.cfg.align_loss_weight * align_loss + \
               ramp * self.cfg.sep_loss_weight * sep_loss + \
               self.cfg.unc_loss_weight * unc_loss
        return loss, {
            "pred_loss": float(pred_loss.detach().cpu()),
            "fast_loss": float(fast_loss.detach().cpu()),
            "align_loss": float(align_loss.detach().cpu()),
            "sep_loss": float(sep_loss.detach().cpu()),
            "unc_loss": float(unc_loss.detach().cpu()),
            "ramp": ramp,
        }



# =========================================================
# PG-DTSF-v2 variants and experiment groups
# =========================================================

PG_V2_CORE_FLAGS = {
    "use_decomposition": True,
    "use_retrieval": True,
    "use_error_guidance": True,
    "use_uncertainty": True,
    "use_memory": False,
    "use_residual": True,
    "horizon_gate": True,
    "adaptive_eta": True,
    "use_align": False,
    "use_sep": False,
    "fixed_eta": None,
}


def pg_v2_flags_from_name(method: str) -> Dict[str, Any]:
    """Translate a method name into an explicit PG-DTSF-v2 configuration."""
    name = method.lower()
    flags = dict(PG_V2_CORE_FLAGS)
    if name in ["pg_v2_core", "pg_dtsf_v2"]:
        return flags
    if name == "pg_v2_no_decomposition":
        flags["use_decomposition"] = False
    elif name == "pg_v2_no_retrieval":
        flags["use_retrieval"] = False
    elif name == "pg_v2_no_error_guidance":
        flags["use_error_guidance"] = False
    elif name == "pg_v2_no_uncertainty":
        flags["use_uncertainty"] = False
    elif name in ["pg_v2_no_residual", "pg_v2_base_only"]:
        flags["use_residual"] = False
    elif name == "pg_v2_scalar_gate":
        flags["horizon_gate"] = False
    elif name == "pg_v2_with_memory":
        flags["use_memory"] = True
    elif name == "pg_v2_with_align":
        flags["use_align"] = True
    elif name == "pg_v2_with_sep":
        flags["use_sep"] = True
    elif name == "pg_v2_full":
        flags["use_memory"] = True
        flags["use_align"] = True
        flags["use_sep"] = True
    elif name == "pg_v2_no_decomp_no_retrieval":
        flags["use_decomposition"] = False
        flags["use_retrieval"] = False
    elif name == "pg_v2_no_error_no_uncertainty":
        flags["use_error_guidance"] = False
        flags["use_uncertainty"] = False
    elif name == "pg_v2_no_decomp_no_error":
        flags["use_decomposition"] = False
        flags["use_error_guidance"] = False
    elif name == "pg_v2_no_retrieval_no_error":
        flags["use_retrieval"] = False
        flags["use_error_guidance"] = False
    elif name.startswith("pg_v2_eta_"):
        token = name[len("pg_v2_eta_"):].replace("p", ".")
        try:
            eta = float(token)
        except ValueError as exc:
            raise ValueError(f"Cannot parse fixed eta from method name: {method}") from exc
        if not (0.0 <= eta <= 2.0):
            raise ValueError(f"Fixed eta should be in [0,2], got {eta}")
        flags["adaptive_eta"] = False
        flags["fixed_eta"] = eta
    else:
        raise ValueError(f"Unknown PG-DTSF-v2 variant: {method}")
    return flags


EXPERIMENT_GROUPS = {
    # Minimum-cost diagnosis: compare the old implementation, strong fusion base, and v2 core.
    "diagnostic": [
        "direct_fusion", "pg_dtsf_v1", "pg_v2_core"
    ],
    # Core module ablation required for the revised paper.
    "core_ablation": [
        "direct_fusion",
        "pg_v2_no_decomposition",
        "pg_v2_no_retrieval",
        "pg_v2_no_error_guidance",
        "pg_v2_no_uncertainty",
        "pg_v2_no_residual",
        "pg_v2_core",
    ],
    # Determine whether horizon-wise gating and adaptive eta are necessary.
    "gate_eta": [
        "direct_fusion",
        "pg_v2_eta_0p1",
        "pg_v2_eta_0p3",
        "pg_v2_eta_0p5",
        "pg_v2_eta_0p7",
        "pg_v2_eta_1p0",
        "pg_v2_scalar_gate",
        "pg_v2_core",
    ],
    # Add each potentially risky regularizer one at a time, then combine them.
    "regularization": [
        "direct_fusion",
        "pg_v2_core",
        "pg_v2_with_memory",
        "pg_v2_with_align",
        "pg_v2_with_sep",
        "pg_v2_full",
    ],
    # Compact paper-ready ablation set.
    "paper_ablation": [
        "direct_fusion",
        "pg_v2_no_decomposition",
        "pg_v2_no_retrieval",
        "pg_v2_no_error_guidance",
        "pg_v2_no_residual",
        "pg_v2_core",
    ],
    # Reviewer-requested combined ablations.
    "combined_ablation": [
        "direct_fusion",
        "pg_v2_no_decomp_no_retrieval",
        "pg_v2_no_error_no_uncertainty",
        "pg_v2_no_decomp_no_error",
        "pg_v2_no_retrieval_no_error",
        "pg_v2_core",
    ],
    # Dual-scale versions of standard temporal baselines. These are used to check
    # whether the proposed method still helps when the strong sequence baselines
    # also receive slow meteorological-calendar context.
    "dual_scale_baselines": [
        "gru", "dual_gru",
        "tcn", "dual_tcn",
        "transformer", "dual_transformer",
        "patchtst", "dual_patchtst",
        "informer", "dual_informer",
        "itransformer", "dual_itransformer",
        "direct_fusion",
        "pg_v2_core",
        "pg_v2_with_memory",
    ],

    # Fused PG-DTSF: integrates the high-level v1 prediction-guided semantic routing
    # with the strong ME-PG-DTSF dual-scale base and memory-enhanced residual correction.
    "fused_diagnostic": [
        "direct_fusion", "pg_dtsf_v1", "pg_v2_with_memory", "fused_pg_dtsf"
    ],
    "fused_ablation": [
        "direct_fusion",
        "fused_pg_base_only",
        "fused_pg_no_memory",
        "fused_pg_no_semantic_memory",
        "fused_pg_no_v1_state",
        "fused_pg_no_decomposition",
        "fused_pg_no_uncertainty",
        "fused_pg_dtsf",
    ],
    "fused_vs_dual": [
        "dual_gru", "dual_tcn", "dual_transformer", "dual_patchtst",
        "dual_informer", "dual_itransformer",
        "direct_fusion", "pg_v2_with_memory", "fused_pg_dtsf",
    ],
    "fused_paper": [
        "direct_fusion",
        "pg_dtsf_v1",
        "pg_v2_with_memory",
        "fused_pg_base_only",
        "fused_pg_no_memory",
        "fused_pg_no_semantic_memory",
        "fused_pg_no_v1_state",
        "fused_pg_dtsf",
    ],
}
EXPERIMENT_GROUPS["all"] = list(dict.fromkeys(
    method for group in EXPERIMENT_GROUPS.values() for method in group
))


def resolve_experiment_methods(group: str, explicit_methods: str = "") -> List[str]:
    if explicit_methods.strip():
        return [m.strip().lower() for m in explicit_methods.split(",") if m.strip()]
    group = group.strip().lower()
    if group not in EXPERIMENT_GROUPS:
        raise ValueError(
            f"Unknown experiment_group={group}. Available groups: {sorted(EXPERIMENT_GROUPS)}"
        )
    return list(EXPERIMENT_GROUPS[group])


class PGDTSFV2(BaseForecastModel):
    """Strong dual-resolution base forecast plus structured semantic residual correction.

    The model starts from a Direct-Fusion-like base predictor. The semantic branch is
    zero-initialized and can only improve the base through a bounded horizon-wise correction.
    """

    def __init__(self, cfg: Config, variant_name: str):
        super().__init__(cfg, variant_name)
        self.variant_name = variant_name
        self.flags = pg_v2_flags_from_name(variant_name)

        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.base_head = MLP(
            cfg.hidden_dim * 2 + cfg.zone_emb_dim,
            cfg.hidden_dim,
            cfg.pred_len,
            cfg.dropout,
        )

        # Properly supervised log-variance-like output for forecasting difficulty.
        self.unc_head = MLP(
            cfg.hidden_dim * 2 + cfg.zone_emb_dim,
            cfg.hidden_dim,
            cfg.pred_len,
            cfg.dropout,
        )

        self.shared_enc = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.shared_dim, cfg.dropout)
        self.spec_encs = nn.ModuleList([
            MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.specific_dim, cfg.dropout)
            for _ in VALID_ZONES
        ])
        self.unified_enc = MLP(
            cfg.hidden_dim, cfg.hidden_dim, cfg.shared_dim + cfg.specific_dim, cfg.dropout
        )
        self.mem_proj = nn.Linear(cfg.shared_dim + cfg.specific_dim, cfg.hidden_dim)

        # Current descriptor: 4 deviation statistics + 4 uncertainty statistics.
        # Historical descriptor: 4 actual base-error statistics. They are concatenated,
        # never averaged dimension-by-dimension.
        self.current_error_dim = 8
        self.history_error_dim = 4
        self.total_error_dim = self.current_error_dim + self.history_error_dim
        self.err_norm = nn.LayerNorm(self.total_error_dim)
        self.err_proj = MLP(self.total_error_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.env_proj = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)

        query_dim = cfg.hidden_dim + cfg.zone_emb_dim + cfg.hidden_dim + cfg.hidden_dim
        self.query_proj = nn.Linear(query_dim, cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        correction_state_dim = cfg.hidden_dim * 4 + cfg.zone_emb_dim
        self.fuse = MLP(
            correction_state_dim, cfg.hidden_dim, cfg.fusion_dim, cfg.dropout
        )
        self.res_head = MLP(
            cfg.fusion_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout
        )
        gate_out = cfg.pred_len if self.flags["horizon_gate"] else 1
        self.gate_head = MLP(
            correction_state_dim, cfg.hidden_dim, gate_out, cfg.dropout
        )
        self.eta_head = MLP(
            correction_state_dim, cfg.hidden_dim, gate_out, cfg.dropout
        )

        self.register_buffer(
            "error_memory",
            torch.zeros(len(VALID_ZONES), self.history_error_dim),
            persistent=True,
        )

        # Start exactly from the strong base predictor. This prevents the newly added
        # semantic branch from damaging the base forecast at initialization.
        self._zero_initialize_residual_head()
        self._initialize_eta_bias(cfg.eta_init)

    @staticmethod
    def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
        last = None
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                last = sub
        return last

    def _zero_initialize_residual_head(self) -> None:
        layer = self._last_linear(self.res_head)
        if layer is not None:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _initialize_eta_bias(self, eta_init: float) -> None:
        layer = self._last_linear(self.eta_head)
        if layer is None or layer.bias is None:
            return
        lo, hi = self.cfg.eta_min, self.cfg.eta_max
        ratio = (eta_init - lo) / max(hi - lo, 1e-8)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        bias = math.log(ratio / (1.0 - ratio))
        nn.init.zeros_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    def _specific_encode(self, hs: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
        out = hs.new_zeros((hs.shape[0], self.cfg.specific_dim))
        for zid in zone.unique(sorted=True):
            mask = zone == zid
            out[mask] = self.spec_encs[int(zid.item())](hs[mask])
        return out

    @staticmethod
    def _four_stats(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            x[:, 0],
            x[:, -1],
        ], dim=-1)

    def _current_difficulty(
        self,
        base_pred: torch.Tensor,
        log_var: torch.Tensor,
        last_load: torch.Tensor,
    ) -> torch.Tensor:
        deviation = torch.abs(base_pred - last_load.unsqueeze(-1))
        dev_stats = self._four_stats(deviation)
        if self.flags["use_uncertainty"]:
            predictive_std = torch.exp(0.5 * torch.clamp(log_var, -6.0, 6.0))
            unc_stats = self._four_stats(predictive_std)
        else:
            unc_stats = torch.zeros_like(dev_stats)
        current = torch.cat([dev_stats, unc_stats], dim=-1)
        if not self.flags["use_error_guidance"]:
            current = torch.zeros_like(current)
        return current

    def _semantic_tokens(
        self, hs: torch.Tensor, zone: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.flags["use_decomposition"]:
            zc = self.shared_enc(hs)
            zr = self._specific_encode(hs, zone)
            shared_token = torch.cat([zc, torch.zeros_like(zr)], dim=-1)
            specific_token = torch.cat([torch.zeros_like(zc), zr], dim=-1)
            tokens = torch.stack([shared_token, specific_token], dim=1)
        else:
            unified = self.unified_enc(hs)
            zc = unified[:, :self.cfg.shared_dim]
            zr = unified[:, self.cfg.shared_dim:]
            tokens = unified.unsqueeze(1)
        return self.mem_proj(tokens), zc, zr

    def _eta(self, correction_state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        fixed_eta = self.flags.get("fixed_eta")
        if fixed_eta is not None:
            return torch.full_like(beta, float(fixed_eta))
        if not self.flags["adaptive_eta"]:
            return torch.full_like(beta, float(self.cfg.semantic_residual_scale))
        raw = self.eta_head(correction_state)
        return self.cfg.eta_min + (self.cfg.eta_max - self.cfg.eta_min) * torch.sigmoid(raw)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        _, hf = self.fast(x_fast)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)

        base_state = torch.cat([hf, hs, e], dim=-1)
        base_pred = self.base_head(base_state)
        log_var = self.unc_head(base_state)

        current = self._current_difficulty(
            base_pred, log_var, x_fast[:, -1, 0]
        )
        if self.flags["use_memory"] and self.flags["use_error_guidance"]:
            history = self.error_memory[zone].detach()
        else:
            history = current.new_zeros((current.shape[0], self.history_error_dim))
        error_state = torch.cat([current, history], dim=-1)
        error_embedding = self.err_proj(self.err_norm(error_state))
        env = self.env_proj(x_slow)

        kv, zc, zr = self._semantic_tokens(hs, zone)
        if self.flags["use_retrieval"]:
            query = self.query_proj(
                torch.cat([hf, e, env, error_embedding], dim=-1)
            ).unsqueeze(1)
            context, attn = self.attn(query, kv, kv, need_weights=True)
            context = context.squeeze(1)
        else:
            context = kv.mean(dim=1)
            attn = None

        correction_state = torch.cat(
            [context, hf, hs, error_embedding, e], dim=-1
        )
        fused = self.fuse(correction_state)

        if self.flags["use_residual"]:
            residual = self.res_head(fused)
            beta = torch.sigmoid(self.gate_head(correction_state))
            eta = self._eta(correction_state, beta)
            gated_residual = eta * beta * residual
            pred = base_pred + gated_residual
        else:
            residual = torch.zeros_like(base_pred)
            beta = torch.zeros_like(base_pred)
            eta = torch.zeros_like(base_pred)
            gated_residual = torch.zeros_like(base_pred)
            pred = base_pred

        return {
            "pred": pred,
            "base_pred": base_pred,
            "fast_pred": base_pred,
            "unc_logit": log_var,
            "current_desc": current,
            "history_desc": history,
            "error_state": error_state,
            "zc": zc,
            "zr": zr,
            "attn": attn,
            "residual": residual,
            "beta": beta,
            "eta": eta,
            "gated_residual": gated_residual,
        }

    def update_memory(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor):
        if not self.flags["use_memory"]:
            return
        with torch.no_grad():
            base = out.get("base_pred")
            if base is None:
                return
            error_stats = self._four_stats(torch.abs(base - y))
            for zid in zone.unique(sorted=True):
                mask = zone == zid
                if mask.any():
                    avg = error_stats[mask].mean(dim=0)
                    zid_int = int(zid.item())
                    self.error_memory[zid_int] = (
                        self.cfg.error_memory_momentum * self.error_memory[zid_int]
                        + (1.0 - self.cfg.error_memory_momentum) * avg
                    )

    @staticmethod
    def _align_loss(zc: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
        centroids = []
        for zid in zone.unique(sorted=True):
            mask = zone == zid
            if mask.sum() > 1:
                centroids.append(zc[mask].mean(dim=0))
        if len(centroids) <= 1:
            return zc.new_tensor(0.0)
        c = torch.stack(centroids, dim=0)
        return ((c - c.mean(dim=0, keepdim=True)) ** 2).mean()

    @staticmethod
    def _sep_loss(zc: torch.Tensor, zr: torch.Tensor) -> torch.Tensor:
        zc0 = zc - zc.mean(dim=0, keepdim=True)
        zr0 = zr - zr.mean(dim=0, keepdim=True)
        cov = zc0.T @ zr0 / max(1, zc0.shape[0] - 1)
        return (cov ** 2).mean()

    def _ramp(self, epoch: int) -> float:
        if epoch <= self.cfg.rampup_warmup:
            return 0.0
        if epoch >= self.cfg.rampup_warmup + self.cfg.rampup_len:
            return 1.0
        return float(epoch - self.cfg.rampup_warmup) / float(self.cfg.rampup_len)

    def compute_loss(
        self,
        out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        zone: torch.Tensor,
        epoch: int = 0,
    ):
        pred_loss = F.mse_loss(out["pred"], y)
        base_loss = F.mse_loss(out["base_pred"], y)

        if self.flags["use_residual"]:
            residual_target = (y - out["base_pred"]).detach()
            residual_loss = F.mse_loss(out["gated_residual"], residual_target)
            correction_l2 = out["gated_residual"].pow(2).mean()
        else:
            residual_loss = pred_loss.new_tensor(0.0)
            correction_l2 = pred_loss.new_tensor(0.0)

        if self.flags["use_uncertainty"]:
            log_var = torch.clamp(out["unc_logit"], -6.0, 6.0)
            squared_error = (out["base_pred"].detach() - y).pow(2)
            unc_loss = (torch.exp(-log_var) * squared_error + log_var).mean()
        else:
            unc_loss = pred_loss.new_tensor(0.0)

        align_loss = (
            self._align_loss(out["zc"], zone)
            if self.flags["use_align"] and self.flags["use_decomposition"]
            else pred_loss.new_tensor(0.0)
        )
        sep_loss = (
            self._sep_loss(out["zc"], out["zr"])
            if self.flags["use_sep"] and self.flags["use_decomposition"]
            else pred_loss.new_tensor(0.0)
        )
        ramp = self._ramp(epoch)

        loss = (
            pred_loss
            + self.cfg.base_loss_weight * base_loss
            + self.cfg.residual_loss_weight * residual_loss
            + self.cfg.hetero_unc_loss_weight * unc_loss
            + self.cfg.correction_l2_weight * correction_l2
            + ramp * self.cfg.v2_align_loss_weight * align_loss
            + ramp * self.cfg.v2_sep_loss_weight * sep_loss
        )
        return loss, {
            "pred_loss": float(pred_loss.detach().cpu()),
            "base_loss": float(base_loss.detach().cpu()),
            "residual_loss": float(residual_loss.detach().cpu()),
            "unc_loss": float(unc_loss.detach().cpu()),
            "align_loss": float(align_loss.detach().cpu()),
            "sep_loss": float(sep_loss.detach().cpu()),
            "correction_l2": float(correction_l2.detach().cpu()),
            "beta_mean": float(out["beta"].detach().mean().cpu()),
            "eta_mean": float(out["eta"].detach().mean().cpu()),
            "ramp": ramp,
        }


# =========================================================
# Fused PG-DTSF: v1 originality + ME-PG-DTSF performance skeleton
# =========================================================

HYBRID_PG_CORE_FLAGS = {
    "use_decomposition": True,
    "use_retrieval": True,
    "use_semantic_memory": True,
    "use_v1_prediction_state": True,
    "use_uncertainty": True,
    "use_memory": True,
    "use_residual": True,
    "horizon_gate": True,
    "adaptive_eta": True,
    "use_align": False,
    "use_sep": False,
    "fixed_eta": None,
}


def hybrid_pg_flags_from_name(method: str) -> Dict[str, Any]:
    """Translate a fused PG-DTSF method name into explicit switches."""
    name = method.lower()
    flags = dict(HYBRID_PG_CORE_FLAGS)
    if name in [
        "fused_pg_dtsf", "pg_dtsf_fused", "pg_dtsf_final", "proposed", "hybrid_pg_dtsf"
    ]:
        return flags
    if name in ["fused_pg_base_only", "fused_pg_no_residual"]:
        flags["use_residual"] = False
    elif name == "fused_pg_no_memory":
        flags["use_memory"] = False
    elif name in ["fused_pg_no_semantic_memory", "fused_pg_no_learned_memory"]:
        flags["use_semantic_memory"] = False
    elif name in ["fused_pg_no_v1_state", "fused_pg_no_prediction_state"]:
        flags["use_v1_prediction_state"] = False
    elif name in ["fused_pg_no_decomposition", "fused_pg_no_decomp"]:
        flags["use_decomposition"] = False
    elif name == "fused_pg_no_uncertainty":
        flags["use_uncertainty"] = False
    elif name == "fused_pg_no_retrieval":
        flags["use_retrieval"] = False
    elif name == "fused_pg_scalar_gate":
        flags["horizon_gate"] = False
    elif name == "fused_pg_with_align":
        flags["use_align"] = True
    elif name == "fused_pg_with_sep":
        flags["use_sep"] = True
    elif name == "fused_pg_full_regularized":
        flags["use_align"] = True
        flags["use_sep"] = True
    elif name.startswith("fused_pg_eta_"):
        token = name[len("fused_pg_eta_"):].replace("p", ".")
        try:
            eta = float(token)
        except ValueError as exc:
            raise ValueError(f"Cannot parse eta from method name: {method}") from exc
        flags["adaptive_eta"] = False
        flags["fixed_eta"] = eta
    else:
        raise ValueError(f"Unknown fused PG-DTSF variant: {method}")
    return flags


class FusedPGDTSF(BaseForecastModel):
    """Fused PG-DTSF.

    This version keeps the high-level originality of PG-DTSF-v1—prediction-state-driven
    semantic routing and semantic memory retrieval—while using the ME-PG-DTSF idea of a
    strong dual-scale base forecast plus zero-initialized semantic residual correction.

    Core computation:
        y_hat = y_base + eta_h * beta_h * Delta y_sem

    where the residual is conditioned on:
        fast dynamics, slow meteorological-calendar semantics, zone embedding,
        prediction-state descriptor, historical error memory, and semantic memory tokens.
    """

    def __init__(self, cfg: Config, variant_name: str = "fused_pg_dtsf"):
        super().__init__(cfg, variant_name)
        self.variant_name = variant_name
        self.flags = hybrid_pg_flags_from_name(variant_name)

        # Strong dual-scale base skeleton.
        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.base_head = MLP(
            cfg.hidden_dim * 2 + cfg.zone_emb_dim,
            cfg.hidden_dim,
            cfg.pred_len,
            cfg.dropout,
        )

        # v1-style auxiliary fast-only forecast, used only to describe prediction disagreement
        # and to add a light auxiliary loss. It is not the final base forecast.
        self.fast_head = MLP(
            cfg.hidden_dim + cfg.zone_emb_dim,
            cfg.hidden_dim,
            cfg.pred_len,
            cfg.dropout,
        )

        # Difficulty / reliability predictor trained through heteroscedastic objective.
        self.unc_head = MLP(
            cfg.hidden_dim * 2 + cfg.zone_emb_dim,
            cfg.hidden_dim,
            cfg.pred_len,
            cfg.dropout,
        )

        # v1-style slow semantic decomposition.
        self.shared_enc = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.shared_dim, cfg.dropout)
        self.spec_encs = nn.ModuleList([
            MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.specific_dim, cfg.dropout)
            for _ in VALID_ZONES
        ])
        self.unified_enc = MLP(
            cfg.hidden_dim, cfg.hidden_dim, cfg.shared_dim + cfg.specific_dim, cfg.dropout
        )
        self.semantic_token_proj = nn.Linear(cfg.shared_dim + cfg.specific_dim, cfg.hidden_dim)

        # Learned semantic memory banks: global recurring patterns and zone-conditioned patterns.
        self.num_global_slots = 4
        self.num_zone_slots = 2
        self.global_semantic_memory = nn.Parameter(
            torch.randn(self.num_global_slots, cfg.hidden_dim) * 0.02
        )
        self.zone_semantic_memory = nn.Parameter(
            torch.randn(len(VALID_ZONES), self.num_zone_slots, cfg.hidden_dim) * 0.02
        )

        # Prediction-state descriptor.
        # 4 stats of base-last-load deviation,
        # 4 stats of base-vs-fast disagreement,
        # 4 stats of predictive uncertainty,
        # 4 stats of intra-horizon variation,
        # plus 4 stats of historical base-error memory.
        self.current_state_dim = 16
        self.history_error_dim = 4
        self.total_state_dim = self.current_state_dim + self.history_error_dim
        self.state_norm = nn.LayerNorm(self.total_state_dim)
        self.state_proj = MLP(self.total_state_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.env_proj = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)

        query_dim = cfg.hidden_dim * 4 + cfg.zone_emb_dim  # hf, hs, env, state, zone
        self.query_proj = nn.Linear(query_dim, cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        correction_state_dim = cfg.hidden_dim * 4 + cfg.zone_emb_dim  # ctx,hf,hs,state,e
        self.fuse = MLP(correction_state_dim, cfg.hidden_dim, cfg.fusion_dim, cfg.dropout)
        self.res_head = MLP(cfg.fusion_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)
        gate_out = cfg.pred_len if self.flags["horizon_gate"] else 1
        self.gate_head = MLP(correction_state_dim, cfg.hidden_dim, gate_out, cfg.dropout)
        self.eta_head = MLP(correction_state_dim, cfg.hidden_dim, gate_out, cfg.dropout)

        self.register_buffer(
            "error_memory",
            torch.zeros(len(VALID_ZONES), self.history_error_dim),
            persistent=True,
        )

        self._zero_initialize_residual_head()
        self._initialize_eta_bias(cfg.eta_init)

    @staticmethod
    def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
        last = None
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                last = sub
        return last

    def _zero_initialize_residual_head(self) -> None:
        layer = self._last_linear(self.res_head)
        if layer is not None:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _initialize_eta_bias(self, eta_init: float) -> None:
        layer = self._last_linear(self.eta_head)
        if layer is None or layer.bias is None:
            return
        lo, hi = self.cfg.eta_min, self.cfg.eta_max
        ratio = (eta_init - lo) / max(hi - lo, 1e-8)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        bias = math.log(ratio / (1.0 - ratio))
        nn.init.zeros_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    @staticmethod
    def _four_stats(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            x[:, 0],
            x[:, -1],
        ], dim=-1)

    def _specific_encode(self, hs: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
        out = hs.new_zeros((hs.shape[0], self.cfg.specific_dim))
        for zid in zone.unique(sorted=True):
            mask = zone == zid
            out[mask] = self.spec_encs[int(zid.item())](hs[mask])
        return out

    def _prediction_state(
        self,
        base_pred: torch.Tensor,
        fast_pred: torch.Tensor,
        log_var: torch.Tensor,
        x_fast: torch.Tensor,
        zone: torch.Tensor,
    ) -> torch.Tensor:
        last_load = x_fast[:, -1, 0]
        base_dev = torch.abs(base_pred - last_load.unsqueeze(-1))
        disagreement = torch.abs(base_pred - fast_pred.detach())
        if self.flags["use_uncertainty"]:
            pred_std = torch.exp(0.5 * torch.clamp(log_var, -6.0, 6.0))
        else:
            pred_std = torch.zeros_like(base_pred)
        if base_pred.shape[-1] > 1:
            horizon_var = torch.abs(base_pred[:, 1:] - base_pred[:, :-1])
            horizon_var = F.pad(horizon_var, (0, 1), mode="replicate")
        else:
            horizon_var = torch.zeros_like(base_pred)

        current = torch.cat([
            self._four_stats(base_dev),
            self._four_stats(disagreement),
            self._four_stats(pred_std),
            self._four_stats(horizon_var),
        ], dim=-1)

        if not self.flags["use_v1_prediction_state"]:
            current = torch.zeros_like(current)

        if self.flags["use_memory"]:
            hist = self.error_memory[zone].detach()
        else:
            hist = current.new_zeros((current.shape[0], self.history_error_dim))

        state = torch.cat([current, hist], dim=-1)
        return state

    def _semantic_tokens(self, hs: torch.Tensor, zone: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokens derived from the current slow context.
        if self.flags["use_decomposition"]:
            zc = self.shared_enc(hs)
            zr = self._specific_encode(hs, zone)
            shared_token = torch.cat([zc, torch.zeros_like(zr)], dim=-1)
            specific_token = torch.cat([torch.zeros_like(zc), zr], dim=-1)
            current_tokens = torch.stack([shared_token, specific_token], dim=1)
        else:
            unified = self.unified_enc(hs)
            zc = unified[:, :self.cfg.shared_dim]
            zr = unified[:, self.cfg.shared_dim:]
            current_tokens = unified.unsqueeze(1)
        current_tokens = self.semantic_token_proj(current_tokens)

        if self.flags["use_semantic_memory"]:
            bsz = hs.shape[0]
            global_tokens = self.global_semantic_memory.unsqueeze(0).expand(bsz, -1, -1)
            zone_tokens = self.zone_semantic_memory[zone]
            tokens = torch.cat([current_tokens, global_tokens, zone_tokens], dim=1)
        else:
            tokens = current_tokens
        return tokens, zc, zr

    def _eta(self, correction_state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        fixed_eta = self.flags.get("fixed_eta")
        if fixed_eta is not None:
            return torch.full_like(beta, float(fixed_eta))
        if not self.flags["adaptive_eta"]:
            return torch.full_like(beta, float(self.cfg.semantic_residual_scale))
        raw = self.eta_head(correction_state)
        return self.cfg.eta_min + (self.cfg.eta_max - self.cfg.eta_min) * torch.sigmoid(raw)

    def forward(self, x_fast, x_slow, zone, y_true=None):
        _, hf = self.fast(x_fast)
        hs = self.slow(x_slow)
        e = self.zone_emb(zone)

        base_state = torch.cat([hf, hs, e], dim=-1)
        base_pred = self.base_head(base_state)
        fast_pred = self.fast_head(torch.cat([hf, e], dim=-1))
        log_var = self.unc_head(base_state)

        raw_state = self._prediction_state(base_pred, fast_pred, log_var, x_fast, zone)
        state_emb = self.state_proj(self.state_norm(raw_state))
        env = self.env_proj(x_slow)

        tokens, zc, zr = self._semantic_tokens(hs, zone)
        if self.flags["use_retrieval"]:
            q = self.query_proj(torch.cat([hf, hs, env, state_emb, e], dim=-1)).unsqueeze(1)
            ctx, attn = self.attn(q, tokens, tokens, need_weights=True)
            ctx = ctx.squeeze(1)
        else:
            ctx = tokens.mean(dim=1)
            attn = None

        correction_state = torch.cat([ctx, hf, hs, state_emb, e], dim=-1)
        fused = self.fuse(correction_state)

        if self.flags["use_residual"]:
            residual = self.res_head(fused)
            beta = torch.sigmoid(self.gate_head(correction_state))
            eta = self._eta(correction_state, beta)
            gated_residual = eta * beta * residual
            pred = base_pred + gated_residual
        else:
            residual = torch.zeros_like(base_pred)
            beta = torch.zeros_like(base_pred)
            eta = torch.zeros_like(base_pred)
            gated_residual = torch.zeros_like(base_pred)
            pred = base_pred

        return {
            "pred": pred,
            "base_pred": base_pred,
            "fast_pred": fast_pred,
            "unc_logit": log_var,
            "prediction_state": raw_state,
            "zc": zc,
            "zr": zr,
            "attn": attn,
            "residual": residual,
            "beta": beta,
            "eta": eta,
            "gated_residual": gated_residual,
        }

    def update_memory(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor):
        if not self.flags["use_memory"]:
            return
        with torch.no_grad():
            base = out.get("base_pred")
            if base is None:
                return
            stats = self._four_stats(torch.abs(base - y))
            for zid in zone.unique(sorted=True):
                mask = zone == zid
                if mask.any():
                    avg = stats[mask].mean(dim=0)
                    zid_int = int(zid.item())
                    self.error_memory[zid_int] = (
                        self.cfg.error_memory_momentum * self.error_memory[zid_int]
                        + (1.0 - self.cfg.error_memory_momentum) * avg
                    )

    @staticmethod
    def _align_loss(zc: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
        centroids = []
        for zid in zone.unique(sorted=True):
            mask = zone == zid
            if mask.sum() > 1:
                centroids.append(zc[mask].mean(dim=0))
        if len(centroids) <= 1:
            return zc.new_tensor(0.0)
        c = torch.stack(centroids, dim=0)
        return ((c - c.mean(dim=0, keepdim=True)) ** 2).mean()

    @staticmethod
    def _sep_loss(zc: torch.Tensor, zr: torch.Tensor) -> torch.Tensor:
        zc0 = zc - zc.mean(dim=0, keepdim=True)
        zr0 = zr - zr.mean(dim=0, keepdim=True)
        cov = zc0.T @ zr0 / max(1, zc0.shape[0] - 1)
        return (cov ** 2).mean()

    def _ramp(self, epoch: int) -> float:
        if epoch <= self.cfg.rampup_warmup:
            return 0.0
        if epoch >= self.cfg.rampup_warmup + self.cfg.rampup_len:
            return 1.0
        return float(epoch - self.cfg.rampup_warmup) / float(self.cfg.rampup_len)

    def compute_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, zone: torch.Tensor, epoch: int = 0):
        pred_loss = F.mse_loss(out["pred"], y)
        base_loss = F.mse_loss(out["base_pred"], y)
        fast_loss = F.mse_loss(out["fast_pred"], y)

        if self.flags["use_residual"]:
            residual_target = (y - out["base_pred"]).detach()
            residual_loss = F.mse_loss(out["gated_residual"], residual_target)
            correction_l2 = out["gated_residual"].pow(2).mean()
        else:
            residual_loss = pred_loss.new_tensor(0.0)
            correction_l2 = pred_loss.new_tensor(0.0)

        if self.flags["use_uncertainty"]:
            log_var = torch.clamp(out["unc_logit"], -6.0, 6.0)
            squared_error = (out["base_pred"].detach() - y).pow(2)
            unc_loss = (torch.exp(-log_var) * squared_error + log_var).mean()
        else:
            unc_loss = pred_loss.new_tensor(0.0)

        align_loss = (
            self._align_loss(out["zc"], zone)
            if self.flags["use_align"] and self.flags["use_decomposition"]
            else pred_loss.new_tensor(0.0)
        )
        sep_loss = (
            self._sep_loss(out["zc"], out["zr"])
            if self.flags["use_sep"] and self.flags["use_decomposition"]
            else pred_loss.new_tensor(0.0)
        )
        ramp = self._ramp(epoch)

        loss = (
            pred_loss
            + self.cfg.base_loss_weight * base_loss
            + 0.05 * fast_loss
            + self.cfg.residual_loss_weight * residual_loss
            + self.cfg.hetero_unc_loss_weight * unc_loss
            + self.cfg.correction_l2_weight * correction_l2
            + ramp * self.cfg.v2_align_loss_weight * align_loss
            + ramp * self.cfg.v2_sep_loss_weight * sep_loss
        )
        return loss, {
            "pred_loss": float(pred_loss.detach().cpu()),
            "base_loss": float(base_loss.detach().cpu()),
            "fast_loss": float(fast_loss.detach().cpu()),
            "residual_loss": float(residual_loss.detach().cpu()),
            "unc_loss": float(unc_loss.detach().cpu()),
            "align_loss": float(align_loss.detach().cpu()),
            "sep_loss": float(sep_loss.detach().cpu()),
            "correction_l2": float(correction_l2.detach().cpu()),
            "beta_mean": float(out["beta"].detach().mean().cpu()),
            "eta_mean": float(out["eta"].detach().mean().cpu()),
            "ramp": ramp,
        }

def build_model(method: str, cfg: Config) -> BaseForecastModel:
    method = method.lower()
    if method == "lstm": return BaselineRNN(cfg, "lstm")
    if method == "gru": return BaselineRNN(cfg, "gru")
    if method == "tcn": return BaselineTCN(cfg)
    if method == "transformer": return BaselineTransformer(cfg)
    if method == "patchtst": return BaselinePatchTST(cfg)
    if method == "informer": return BaselineInformer(cfg)
    if method == "itransformer": return BaselineITransformer(cfg)
    # Dual-scale counterparts: use x_fast + x_slow + zone embedding.
    if method in ["dual_lstm", "lstm_dual"]: return DualScaleRNN(cfg, "lstm")
    if method in ["dual_gru", "gru_dual"]: return DualScaleRNN(cfg, "gru")
    if method in ["dual_tcn", "tcn_dual"]: return DualScaleTCN(cfg)
    if method in ["dual_transformer", "transformer_dual"]: return DualScaleTransformer(cfg)
    if method in ["dual_patchtst", "patchtst_dual"]: return DualScalePatchTST(cfg)
    if method in ["dual_informer", "informer_dual"]: return DualScaleInformer(cfg)
    if method in ["dual_itransformer", "itransformer_dual"]: return DualScaleITransformer(cfg)
    if method == "timemixer": return BaselineTimeMixer(cfg)
    if method == "timexer": return BaselineTimeXer(cfg)
    if method == "direct_fusion": return DirectFusion(cfg)
    if method == "gated_fusion": return GatedFusion(cfg)
    if method == "cross_attention_fusion": return CrossAttentionFusion(cfg)
    if method == "residual_fusion": return ResidualFusion(cfg)
    if method == "shared_bottom_mtl": return SharedBottomMTL(cfg)
    if method in ["pg_dtsf_v1", "pg_v1", "mt_eidtf_final"]:
        return PGDTSFV1(cfg)
    if (
        method in ["fused_pg_dtsf", "pg_dtsf_fused", "pg_dtsf_final", "proposed", "hybrid_pg_dtsf"]
        or method.startswith("fused_pg_")
    ):
        return FusedPGDTSF(cfg, method)
    if method.startswith("pg_v2_") or method == "pg_dtsf_v2":
        return PGDTSFV2(cfg, method)
    raise ValueError(f"Unknown method: {method}")


# =========================================================
# 7. Training and evaluation
# =========================================================

@torch.no_grad()
def evaluate_model(model: BaseForecastModel,
                   loader: DataLoader,
                   cfg: Config,
                   return_predictions: bool = False) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """Memory-safe evaluation with optional PG-DTSF-v2 diagnostics."""
    model.eval()
    meter = OnlineRegressionMetrics(eps=1.0)
    base_meter = OnlineRegressionMetrics(eps=1.0)
    has_base = False
    diag_sum = {"beta": 0.0, "eta": 0.0, "correction_abs": 0.0}
    diag_count = 0
    pred_all, true_all = [], []
    pred_scaled_all, true_scaled_all = [], []
    zone_all, t_all = [], []

    for batch in loader:
        x_fast = batch["x_fast"].to(cfg.device, non_blocking=True).float()
        x_slow = batch["x_slow"].to(cfg.device, non_blocking=True).float()
        y = batch["y"].to(cfg.device, non_blocking=True).float()
        zone = batch["zone"].to(cfg.device, non_blocking=True).long()
        mean = batch["target_mean"].to(cfg.device, non_blocking=True).float()
        scale = batch["target_scale"].to(cfg.device, non_blocking=True).float()

        out = model(x_fast, x_slow, zone)
        pred_s = out["pred"]
        pred = inverse_target(pred_s, mean, scale)
        true = inverse_target(y, mean, scale)

        pred_np = pred.detach().cpu().numpy()
        true_np = true.detach().cpu().numpy()
        meter.update(pred_np, true_np)

        base_s = out.get("base_pred")
        if base_s is not None:
            has_base = True
            base = inverse_target(base_s, mean, scale)
            base_meter.update(base.detach().cpu().numpy(), true_np)

        beta = out.get("beta")
        eta = out.get("eta")
        correction = out.get("gated_residual")
        if beta is not None and eta is not None and correction is not None:
            n_elem = int(correction.numel())
            diag_sum["beta"] += float(beta.detach().sum().cpu())
            diag_sum["eta"] += float(eta.detach().sum().cpu())
            correction_orig = correction * scale.unsqueeze(-1)
            diag_sum["correction_abs"] += float(correction_orig.detach().abs().sum().cpu())
            # Scalar gates/eta have fewer elements; normalize beta/eta separately below.
            diag_count += n_elem

        if return_predictions:
            pred_all.append(pred_np.astype(np.float32, copy=False))
            true_all.append(true_np.astype(np.float32, copy=False))
            pred_scaled_all.append(pred_s.detach().cpu().numpy().astype(np.float32, copy=False))
            true_scaled_all.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
            zone_all.append(zone.detach().cpu().numpy())
            t_all.extend(batch["timestamp"])

    metrics = meter.compute()
    if has_base:
        base_metrics = base_meter.compute()
        metrics.update({f"Base_{k}": v for k, v in base_metrics.items()})
    if diag_count > 0:
        # correction_abs is always horizon-wise. beta/eta are broadcast-compatible;
        # their direct tensor means are also accumulated during training histories.
        metrics["CorrectionAbsMean"] = diag_sum["correction_abs"] / diag_count

    pack = None
    if return_predictions:
        pack = {
            "pred": np.concatenate(pred_all, axis=0),
            "true": np.concatenate(true_all, axis=0),
            "pred_scaled": np.concatenate(pred_scaled_all, axis=0),
            "true_scaled": np.concatenate(true_scaled_all, axis=0),
            "zone": np.concatenate(zone_all, axis=0),
            "timestamp": np.array(t_all, dtype=object),
        }
    return metrics, pack


def train_one_model(method: str, seed: int, cfg: Config,
                    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                    out_dir: str) -> Dict[str, Any]:
    set_seed(seed)
    ensure_dir(out_dir)
    result_path = os.path.join(out_dir, f"result_{method}_seed{seed}.json")
    if cfg.resume and os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(f"[RESUME] Skip completed run: {method}, seed={seed}")
        return result

    model = build_model(method, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_enabled = bool(cfg.use_amp and str(cfg.device).startswith("cuda") and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        item_sums: Dict[str, float] = {}
        item_count = 0
        epoch_t0 = time.time()
        for batch in train_loader:
            x_fast = batch["x_fast"].to(cfg.device, non_blocking=True).float()
            x_slow = batch["x_slow"].to(cfg.device, non_blocking=True).float()
            y = batch["y"].to(cfg.device, non_blocking=True).float()
            zone = batch["zone"].to(cfg.device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(x_fast, x_slow, zone, y_true=y)
                loss, loss_items = model.compute_loss(out, y, zone, epoch)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            model.update_memory(out, y, zone)

            losses.append(float(loss.detach().cpu()))
            for key, value in loss_items.items():
                item_sums[key] = item_sums.get(key, 0.0) + float(value)
            item_count += 1

        train_loss = float(np.mean(losses)) if losses else np.nan
        val_metrics, _ = evaluate_model(model, val_loader, cfg, return_predictions=False)
        val_rmse = val_metrics["RMSE"]
        epoch_time = time.time() - epoch_t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_RMSE": val_rmse,
            "val_MAE": val_metrics["MAE"],
            "val_MAPE": val_metrics["MAPE"],
            "epoch_time_sec": epoch_time,
        }
        if "Base_RMSE" in val_metrics:
            row["val_Base_RMSE"] = val_metrics["Base_RMSE"]
        for key, total in item_sums.items():
            row[f"train_{key}"] = total / max(1, item_count)
        history.append(row)

        base_text = f" base_RMSE={val_metrics['Base_RMSE']:.4f}" if "Base_RMSE" in val_metrics else ""
        print(
            f"[{method}][seed={seed}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6e} val_RMSE={val_rmse:.4f} "
            f"val_MAE={val_metrics['MAE']:.4f}{base_text} time={epoch_time:.1f}s"
        )

        if val_rmse < best_val - 1e-8:
            best_val = val_rmse
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "method": method,
                "seed": seed,
                "cfg": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "best_val_RMSE": best_val,
                "best_epoch": best_epoch,
            }, os.path.join(out_dir, f"best_{method}_seed{seed}.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[{method}][seed={seed}] Early stopping at epoch {epoch}.")
                break

    training_time = time.time() - start_time
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, f"history_{method}_seed{seed}.csv"), index=False)

    ckpt_path = os.path.join(out_dir, f"best_{method}_seed{seed}.pt")
    try:
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    test_metrics, preds = evaluate_model(model, test_loader, cfg, return_predictions=cfg.save_predictions)
    if cfg.save_predictions and preds is not None:
        pred_path = os.path.join(out_dir, f"predictions_{method}_seed{seed}.npz")
        np.savez_compressed(pred_path, **preds)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val,
        "training_time_sec": training_time,
        "train_time_per_epoch_sec": training_time / max(1, len(history)),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_million": total_params / 1e6,
        "trainable_params_million": trainable_params / 1e6,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{method}][seed={seed}] Final test: {test_metrics}")
    return result


def summarize_results(results: List[Dict[str, Any]], out_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "all_seed_results_detailed.csv"), index=False)
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method, "num_seeds": len(g)}
        for metric in [
            "test_MAE", "test_RMSE", "test_MAPE", "test_R2",
            "test_Base_MAE", "test_Base_RMSE", "test_Base_MAPE", "test_Base_R2",
            "test_CorrectionAbsMean",
            "train_time_per_epoch_sec", "total_params_million"
        ]:
            if metric in g.columns:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("test_RMSE_mean" if "test_RMSE_mean" in rows[0] else "method")
    summary.to_csv(os.path.join(out_dir, "all_seed_results_mean_std.csv"), index=False)

    # Paper-ready compact table
    paper_rows = []
    for _, r in summary.iterrows():
        paper_rows.append({
            "Method": r["method"],
            "MAE": f"{r['test_MAE_mean']:.2f} $\\pm$ {r['test_MAE_std']:.2f}",
            "RMSE": f"{r['test_RMSE_mean']:.2f} $\\pm$ {r['test_RMSE_std']:.2f}",
            "MAPE": f"{r['test_MAPE_mean']:.2f} $\\pm$ {r['test_MAPE_std']:.2f}",
            "R2": f"{r['test_R2_mean']:.4f} $\\pm$ {r['test_R2_std']:.4f}",
        })
    pd.DataFrame(paper_rows).to_csv(os.path.join(out_dir, "paper_overall_table_mean_std.csv"), index=False)
    return summary


def run_experiments(cfg: Config, methods: List[str]) -> None:
    ensure_dir(cfg.run_dir)
    with open(os.path.join(cfg.run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "resolved_methods": methods}, f, ensure_ascii=False, indent=2)

    train_ds = ZoneWindowDataset(cfg, "train", max_samples=cfg.max_train_samples)
    val_ds = ZoneWindowDataset(cfg, "val", max_samples=cfg.max_eval_samples)
    test_ds = ZoneWindowDataset(cfg, "test", max_samples=cfg.max_eval_samples)

    pers = persistence_metrics(test_ds)
    pd.DataFrame([{"method": "previous_day_persistence", **pers}]).to_csv(
        os.path.join(cfg.run_dir, "persistence_test_metrics.csv"), index=False
    )
    print(f"[Persistence] {pers}")

    results: List[Dict[str, Any]] = []
    for method in methods:
        method_dir = os.path.join(cfg.run_dir, method)
        ensure_dir(method_dir)
        for seed in cfg.seeds:
            train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
            val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
            test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
            try:
                result = train_one_model(
                    method, seed, cfg, train_loader, val_loader, test_loader, method_dir
                )
                results.append(result)
                pd.DataFrame(results).to_csv(
                    os.path.join(cfg.run_dir, "running_results.csv"), index=False
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        f"[OOM] {method} seed={seed}. Reduce batch_size or hidden_dim. Error: {exc}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

    if not results:
        raise RuntimeError("No experiment completed successfully.")
    summary = summarize_results(results, cfg.run_dir)
    print("\n=== Mean ± std summary ===")
    print(summary.to_string(index=False))


# =========================================================
# 8. CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PG-DTSF-v2 multi-group diagnostic and ablation experiments"
    )
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip_experiments", action="store_true")

    parser.add_argument(
        "--experiment_group",
        type=str,
        default="diagnostic",
        choices=sorted(EXPERIMENT_GROUPS.keys()),
        help="Predefined experiment group. Ignored when --methods is non-empty.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated explicit method list; overrides --experiment_group.",
    )
    parser.add_argument("--experiment_tag", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="2026")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--sample_mode", choices=["first", "uniform", "random"], default="uniform")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # V2 objective controls for sensitivity studies.
    parser.add_argument("--base_loss_weight", type=float, default=0.05)
    parser.add_argument("--residual_loss_weight", type=float, default=0.10)
    parser.add_argument("--hetero_unc_loss_weight", type=float, default=0.01)
    parser.add_argument("--correction_l2_weight", type=float, default=1e-4)
    parser.add_argument("--v2_align_loss_weight", type=float, default=1e-5)
    parser.add_argument("--v2_sep_loss_weight", type=float, default=1e-5)
    parser.add_argument("--eta_min", type=float, default=0.05)
    parser.add_argument("--eta_max", type=float, default=1.0)
    parser.add_argument("--eta_init", type=float, default=0.30)

    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save full prediction arrays. Avoid for full million-sample test sets."
    )
    parser.add_argument("--no_save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.save_root is not None:
        cfg.save_root = args.save_root
    if args.device is not None:
        cfg.device = args.device

    cfg.run_experiments = not args.skip_experiments
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.patience = args.patience
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.hidden_dim = args.hidden_dim
    cfg.fusion_dim = args.hidden_dim
    cfg.num_workers = args.num_workers
    cfg.max_train_samples = args.max_train_samples
    cfg.max_eval_samples = args.max_eval_samples
    cfg.sample_mode = args.sample_mode
    cfg.resume = bool(args.resume)
    cfg.use_amp = bool(args.amp)
    cfg.save_predictions = bool(args.save_predictions) and not bool(args.no_save_predictions)
    cfg.seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    cfg.base_loss_weight = args.base_loss_weight
    cfg.residual_loss_weight = args.residual_loss_weight
    cfg.hetero_unc_loss_weight = args.hetero_unc_loss_weight
    cfg.correction_l2_weight = args.correction_l2_weight
    cfg.v2_align_loss_weight = args.v2_align_loss_weight
    cfg.v2_sep_loss_weight = args.v2_sep_loss_weight
    cfg.eta_min = args.eta_min
    cfg.eta_max = args.eta_max
    cfg.eta_init = args.eta_init

    methods = resolve_experiment_methods(args.experiment_group, args.methods)
    if args.experiment_tag is not None and args.experiment_tag.strip():
        cfg.experiment_tag = args.experiment_tag.strip()
    else:
        cfg.experiment_tag = f"pg_dtsf_v2_{args.experiment_group}"

    print("=" * 110)
    print("PG-DTSF-v2 multi-group experiments")
    print(
        f"Split: train [{cfg.train_start}, {cfg.train_end_exclusive}), "
        f"val [{cfg.val_start}, {cfg.val_end_exclusive}), "
        f"test [{cfg.test_start}, {cfg.test_end_exclusive})"
    )
    print(f"Save dir : {cfg.save_dir}")
    print(f"Run dir  : {cfg.run_dir}")
    print(f"Device   : {cfg.device}")
    print(f"Seeds    : {cfg.seeds}")
    print(f"Group    : {args.experiment_group}")
    print(f"Methods  : {methods}")
    print(f"Resume   : {cfg.resume}; AMP: {cfg.use_amp}; sample_mode: {cfg.sample_mode}")
    print("=" * 110)

    if cfg.run_experiments:
        run_experiments(cfg, methods)


if __name__ == "__main__":
    main()
