#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strong baseline experiments for NYISO multi-zone H288->P288 forecasting.

This script is the dedicated STRONG_CORE version for complementing SGR-PG-DTSF experiments.
It runs the core strong baseline panel under the same synchronized multi-zone data interface:

    processed/two_half_year_201601_201807_per_zone_scalers_H288_P288/
        train_by_zone/train_Xf_<zone>.npy
        train_by_zone/train_Xs_<zone>.npy
        train_by_zone/train_Y_<zone>.npy
        val_by_zone/...
        test_by_zone/...
        scalers_by_zone/<zone>/target_mean.npy
        scalers_by_zone/<zone>/target_scale.npy

It imports the original experiment script only for:
    - Config
    - VALID_ZONES
    - MultiZoneSynchronizedDataset
    - inverse_scale_batch
    - check_processed_data

The proposed SGR-PG-DTSF model is NOT imported here. This file is only for
external/comparative baselines.

Baseline groups
---------------
1) Non-parametric:
    repeat_last_value
    repeat_last_day

2) Linear / MLP:
    linear_direct
    nlinear
    dlinear
    mlp_exog
    tide_mlp
    tsmixer_exog

3) Sequence / Transformer:
    tcn_exog
    gru_exog
    lstm_exog
    transformer_exog
    patchtst_ci_exog
    itransformer_exog
    timesnet_lite_exog

4) Spatial-temporal / graph:
    stid_mlp
    graph_wavenet_lite
    mtgnn_lite
    agcrn_lite

Typical command
---------------
python two_half_year_strong_baseline_experiments.py ^
  --base_script "two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py" ^
  --save_root ".\\processed" ^
  --experiment_tag strong_baselines_5seeds ^
  --experiment_group strong_all ^
  --device cuda ^
  --seeds "2026,2027,2028,2029,2030" ^
  --epochs 30 ^
  --patience 6 ^
  --batch_size 64 ^
  --lr 0.0005 ^
  --weight_decay 0.00001 ^
  --hidden_dim 128 ^
  --amp ^
  --resume

Notes for paper writing
-----------------------
These are compact, same-interface implementations for fair benchmarking under the
current dataset. For final paper submission, describe them as "implemented
baselines under the same H288->P288 multi-zone setting". If you want exact
official reproductions of PatchTST/iTransformer/TimesNet/Graph WaveNet/MTGNN,
you can additionally run official repositories; however, this file makes it
possible to obtain a broad and reproducible baseline panel in one framework.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# Import base data interface
# =============================================================================

def resolve_base_script(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p.resolve()
    candidates = [
        Path.cwd() / path_like,
        Path.cwd() / "two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py",
        Path.cwd() / "two_half_year_sgr_pg_dtsf_experiments_true_no_slow.py",
        Path.cwd() / "two_half_year_sgr_pg_dtsf_experiments.py",
        Path.cwd() / "two_half_year_sgr_pg_dtsf_experiments(1).py",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Cannot find base experiment script. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


def import_base_module(base_script: str):
    script_path = resolve_base_script(base_script)
    module_name = "sgr_data_interface_module"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"[IMPORT] Loaded base data interface from: {script_path}")
    return module


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_seeds(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_methods(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_metrics_flat(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs(err[mask] / yt[mask])) * 100.0) if np.any(mask) else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def checkpoint_path(cfg, method: str, seed: int) -> str:
    return os.path.join(cfg.run_dir, f"best_{method}_seed{seed}.pt")


def result_exists(cfg, method: str, seed: int) -> bool:
    path = os.path.join(cfg.run_dir, "running_results.csv")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return bool(((df["method"] == method) & (df["seed"] == seed)).any())
    except Exception:
        return False


def append_result(cfg, row: Dict[str, Any]) -> None:
    ensure_dir(cfg.run_dir)
    path = os.path.join(cfg.run_dir, "running_results.csv")
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)


# =============================================================================
# Data loaders
# =============================================================================

def make_loaders(base_mod, cfg):
    train_ds = base_mod.MultiZoneSynchronizedDataset(cfg, "train", cfg.max_train_samples)
    val_ds = base_mod.MultiZoneSynchronizedDataset(cfg, "val", cfg.max_eval_samples)
    test_ds = base_mod.MultiZoneSynchronizedDataset(cfg, "test", cfg.max_eval_samples)
    pin = str(cfg.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Model blocks
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = (self.kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,Z,L]
        if self.kernel_size <= 1:
            return x
        B, Z, L = x.shape
        # Replication padding at both ends.
        front = x[:, :, 0:1].repeat(1, 1, self.padding)
        end = x[:, :, -1:].repeat(1, 1, self.padding)
        xx = torch.cat([front, x, end], dim=-1)
        out = self.avg(xx.reshape(B * Z, 1, -1)).reshape(B, Z, -1)
        # Length may be off by one if kernel even; crop/pad to original.
        if out.shape[-1] > L:
            out = out[..., :L]
        elif out.shape[-1] < L:
            out = F.pad(out, (0, L - out.shape[-1]), mode="replicate")
        return out


class GraphPropagation(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, graph_dim: int = 32, dropout: float = 0.1, mix_hop: int = 2):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.mix_hop = int(mix_hop)
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, graph_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, graph_dim) * 0.1)
        self.lin = nn.Linear(hidden_dim * (self.mix_hop + 1), hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def adjacency(self) -> torch.Tensor:
        A = torch.relu(self.node_emb1 @ self.node_emb2.t())
        A = torch.softmax(A, dim=-1)
        return A

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h [B,Z,D]
        A = self.adjacency()
        outs = [h]
        x = h
        for _ in range(self.mix_hop):
            x = torch.einsum("ij,bjd->bid", A, x)
            outs.append(x)
        out = torch.cat(outs, dim=-1)
        out = self.lin(out)
        return self.dropout(out)


# =============================================================================
# Baseline models
# =============================================================================

class BaseForecastModel(nn.Module):
    def forward(self, x_fast: torch.Tensor, x_slow: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class RepeatLastValue(BaseForecastModel):
    def __init__(self, cfg):
        super().__init__()
        self.pred_len = int(cfg.pred_len)

    def forward(self, x_fast, x_slow):
        last = x_fast[:, :, -1, 0]
        pred = last.unsqueeze(-1).repeat(1, 1, self.pred_len)
        return {"pred": pred}


class RepeatLastDay(BaseForecastModel):
    def __init__(self, cfg):
        super().__init__()
        self.pred_len = int(cfg.pred_len)

    def forward(self, x_fast, x_slow):
        load = x_fast[:, :, :, 0]
        if load.shape[-1] == self.pred_len:
            pred = load
        elif load.shape[-1] > self.pred_len:
            pred = load[:, :, -self.pred_len:]
        else:
            reps = math.ceil(self.pred_len / load.shape[-1])
            pred = load.repeat(1, 1, reps)[:, :, :self.pred_len]
        return {"pred": pred}


class LinearDirect(BaseForecastModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.hist_len * cfg.fast_dim + cfg.slow_dim + cfg.zone_emb_dim
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.proj = nn.Linear(in_dim, cfg.pred_len)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        inp = torch.cat([x_fast.reshape(B, Z, L * Fdim), x_slow, ez], dim=-1)
        pred = self.proj(inp)
        return {"pred": pred}


class NLinear(BaseForecastModel):
    """Normalized Linear baseline: linear prediction over load history after subtracting last value."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.hist_len, cfg.pred_len)

    def forward(self, x_fast, x_slow):
        load = x_fast[:, :, :, 0]
        last = load[:, :, -1:].detach()
        pred = self.linear(load - last) + last
        return {"pred": pred}


class DLinear(BaseForecastModel):
    """DLinear-style decomposition baseline."""
    def __init__(self, cfg, kernel_size: int = 25):
        super().__init__()
        self.cfg = cfg
        self.moving_avg = MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(cfg.hist_len, cfg.pred_len)
        self.linear_trend = nn.Linear(cfg.hist_len, cfg.pred_len)

    def forward(self, x_fast, x_slow):
        x = x_fast[:, :, :, 0]
        trend = self.moving_avg(x)
        seasonal = x - trend
        pred = self.linear_seasonal(seasonal) + self.linear_trend(trend)
        return {"pred": pred}


class MLPExog(BaseForecastModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        in_dim = cfg.hist_len * cfg.fast_dim + cfg.slow_dim + cfg.zone_emb_dim
        self.mlp = MLP(in_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=3)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        inp = torch.cat([x_fast.reshape(B, Z, L * Fdim), x_slow, ez], dim=-1)
        pred = self.mlp(inp)
        return {"pred": pred}


class TiDEMLP(BaseForecastModel):
    """Compact TiDE-like encoder/decoder MLP with residual connection to last value."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        in_dim = cfg.hist_len * cfg.fast_dim + cfg.slow_dim + cfg.zone_emb_dim
        self.encoder = MLP(in_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout, n_layers=2)
        self.decoder = MLP(cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        enc_in = torch.cat([x_fast.reshape(B, Z, L * Fdim), x_slow, ez], dim=-1)
        h = self.encoder(enc_in)
        dec_in = torch.cat([h, x_slow, ez], dim=-1)
        delta = self.decoder(dec_in)
        last = x_fast[:, :, -1, 0].unsqueeze(-1)
        pred = last + self.res_scale * delta
        return {"pred": pred}


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm_t = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
        )
        self.norm_c = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        # x [B*Z,L,D]
        y = self.norm_t(x)
        y = y.transpose(1, 2)
        y = self.time_mlp(y).transpose(1, 2)
        x = x + y
        y = self.norm_c(x)
        y = self.channel_mlp(y)
        return x + y


class TSMixerExog(BaseForecastModel):
    def __init__(self, cfg, n_layers: int = 4):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([TSMixerBlock(cfg.hist_len, cfg.hidden_dim, cfg.dropout) for _ in range(n_layers)])
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hist_len * cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = self.in_proj(x_fast.reshape(B * Z, L, Fdim))
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(B, Z, L * self.cfg.hidden_dim)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([x, x_slow, ez], dim=-1))
        return {"pred": pred}


class TCNExog(BaseForecastModel):
    def __init__(self, cfg, levels: int = 5):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_ch = cfg.fast_dim
        hidden = cfg.hidden_dim
        for i in range(levels):
            dilation = 2 ** i
            pad = dilation * 2
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size=3, padding=pad, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(hidden + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = x_fast.reshape(B * Z, L, Fdim).transpose(1, 2)
        h = self.tcn(x)
        h = h[:, :, :L]  # crop if padded longer
        last = h[:, :, -1].reshape(B, Z, -1)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([last, x_slow, ez], dim=-1))
        return {"pred": pred}


class RNNExog(BaseForecastModel):
    def __init__(self, cfg, rnn_type: str = "gru"):
        super().__init__()
        self.cfg = cfg
        cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn_type = rnn_type
        self.rnn = cls(cfg.fast_dim, cfg.hidden_dim, num_layers=2, batch_first=True, dropout=cfg.dropout)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        out = self.rnn(x_fast.reshape(B * Z, L, Fdim))
        if self.rnn_type == "gru":
            h = out[1][-1]
        else:
            h = out[1][0][-1]
        h = h.reshape(B, Z, -1)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([h, x_slow, ez], dim=-1))
        return {"pred": pred}


class TransformerExog(BaseForecastModel):
    def __init__(self, cfg, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.hist_len, cfg.hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=n_heads, dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = self.in_proj(x_fast.reshape(B * Z, L, Fdim)) + self.pos_emb[:, :L]
        h = self.encoder(x)[:, -1].reshape(B, Z, -1)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([h, x_slow, ez], dim=-1))
        return {"pred": pred}


class PatchTSTCIExog(BaseForecastModel):
    """Channel-independent PatchTST-style per-zone patched Transformer."""
    def __init__(self, cfg, patch_len: int = 16, stride: int = 8, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.cfg = cfg
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.num_patches = 1 + max(0, (cfg.hist_len - self.patch_len) // self.stride)
        patch_dim = self.patch_len * cfg.fast_dim
        self.patch_proj = nn.Linear(patch_dim, cfg.hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, cfg.hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=n_heads, dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(self.num_patches * cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = x_fast.reshape(B * Z, L, Fdim)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B*Z,N,F,patch]
        patches = patches.permute(0, 1, 3, 2).reshape(B * Z, self.num_patches, self.patch_len * Fdim)
        tokens = self.patch_proj(patches) + self.pos_emb
        h = self.encoder(tokens).reshape(B, Z, self.num_patches * self.cfg.hidden_dim)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([h, x_slow, ez], dim=-1))
        return {"pred": pred}


class ITransformerExog(BaseForecastModel):
    """iTransformer-style model: zones are variate tokens."""
    def __init__(self, cfg, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.cfg = cfg
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        token_dim = cfg.hist_len * cfg.fast_dim + cfg.slow_dim + cfg.zone_emb_dim
        self.token_proj = nn.Linear(token_dim, cfg.hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=n_heads, dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        token = torch.cat([x_fast.reshape(B, Z, L * Fdim), x_slow, ez], dim=-1)
        h = self.encoder(self.token_proj(token))
        pred = self.head(h)
        return {"pred": pred}


class TimesNetLiteExog(BaseForecastModel):
    """Lightweight multi-kernel temporal convolution inspired by multi-period temporal variation modeling."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kernels = [3, 5, 9, 17, 33]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cfg.fast_dim, cfg.hidden_dim, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=k, padding=k // 2),
                nn.GELU(),
            ) for k in kernels
        ])
        self.fuse = nn.Linear(cfg.hidden_dim * len(kernels), cfg.hidden_dim)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = x_fast.reshape(B * Z, L, Fdim).transpose(1, 2)
        feats = []
        for br in self.branches:
            h = br(x)
            h = h[:, :, :L].mean(dim=-1)
            feats.append(h)
        h = self.fuse(torch.cat(feats, dim=-1)).reshape(B, Z, -1)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([h, x_slow, ez], dim=-1))
        return {"pred": pred}


class STIDMLP(BaseForecastModel):
    """Spatial-Temporal Identity style MLP using zone id, time encodings, slow semantics, and load summaries."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.hidden_dim)
        # Summary: last, mean, std, min, max, ramp + hour_sin/cos last + slow
        in_dim = cfg.hidden_dim + 8 + cfg.slow_dim
        self.mlp = MLP(in_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=3)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        load = x_fast[:, :, :, 0]
        ramp = torch.abs(load[:, :, 1:] - load[:, :, :-1]).mean(dim=-1)
        stats = torch.stack([
            load[:, :, -1],
            load.mean(dim=-1),
            load.std(dim=-1, unbiased=False),
            load.min(dim=-1).values,
            load.max(dim=-1).values,
            ramp,
            x_fast[:, :, -1, 1],
            x_fast[:, :, -1, 2],
        ], dim=-1)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.mlp(torch.cat([ez, stats, x_slow], dim=-1))
        return {"pred": pred}


class GraphWaveNetLite(BaseForecastModel):
    """Adaptive graph + dilated temporal convolution baseline."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temporal = TCNExog(cfg, levels=5)
        # Replace temporal head with hidden extraction modules.
        self.temporal_head_removed = True
        self.in_conv = nn.Conv1d(cfg.fast_dim, cfg.hidden_dim, kernel_size=1)
        self.dilated = nn.ModuleList([
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=2, dilation=2 ** i, padding=2 ** i)
            for i in range(5)
        ])
        self.graph = GraphPropagation(cfg.num_zones, cfg.hidden_dim, graph_dim=32, dropout=cfg.dropout, mix_hop=2)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = x_fast.reshape(B * Z, L, Fdim).transpose(1, 2)
        h = self.in_conv(x)
        for conv in self.dilated:
            res = h
            y = torch.tanh(conv(h)[:, :, :h.shape[-1]])
            h = y + res
        h = h[:, :, -1].reshape(B, Z, -1)
        h = self.graph(h)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([h, x_slow, ez], dim=-1))
        return {"pred": pred}


class MTGNNLite(BaseForecastModel):
    """MTGNN-style adaptive graph temporal baseline with mix-hop graph propagation."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temporal_branches = nn.ModuleList([
            nn.Conv1d(cfg.fast_dim, cfg.hidden_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])
        self.graph1 = GraphPropagation(cfg.num_zones, cfg.hidden_dim, graph_dim=32, dropout=cfg.dropout, mix_hop=2)
        self.graph2 = GraphPropagation(cfg.num_zones, cfg.hidden_dim, graph_dim=32, dropout=cfg.dropout, mix_hop=2)
        self.fuse = nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hidden_dim + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = x_fast.reshape(B * Z, L, Fdim).transpose(1, 2)
        outs = []
        for conv in self.temporal_branches:
            y = torch.relu(conv(x))[:, :, :L].mean(dim=-1)
            outs.append(y)
        h = self.fuse(torch.cat(outs, dim=-1)).reshape(B, Z, -1)
        h = 0.5 * self.graph1(h) + 0.5 * self.graph2(h)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([h, x_slow, ez], dim=-1))
        return {"pred": pred}


class AGCRNLite(BaseForecastModel):
    """Adaptive graph convolutional recurrent baseline."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.fast_dim, cfg.hidden_dim)
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, num_layers=1, batch_first=True)
        self.graph = GraphPropagation(cfg.num_zones, cfg.hidden_dim, graph_dim=32, dropout=cfg.dropout, mix_hop=2)
        self.node_adapt = nn.Embedding(cfg.num_zones, cfg.hidden_dim)
        self.zone_emb = nn.Embedding(cfg.num_zones, cfg.zone_emb_dim)
        self.head = MLP(cfg.hidden_dim * 2 + cfg.slow_dim + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout, n_layers=2)

    def forward(self, x_fast, x_slow):
        B, Z, L, Fdim = x_fast.shape
        x = self.in_proj(x_fast.reshape(B * Z, L, Fdim))
        _, h = self.gru(x)
        h = h[-1].reshape(B, Z, -1)
        hg = self.graph(h)
        zids = torch.arange(Z, device=x_fast.device).unsqueeze(0).expand(B, Z)
        na = self.node_adapt(zids)
        ez = self.zone_emb(zids)
        pred = self.head(torch.cat([hg, na, x_slow, ez], dim=-1))
        return {"pred": pred}


# =============================================================================
# Model factory
# =============================================================================

NON_TRAINABLE_METHODS = {"repeat_last_value", "repeat_last_day"}

STRONG_CORE_METHODS = [
    "repeat_last_day",
    "linear_direct",
    "nlinear",
    "dlinear",
    "mlp_exog",
    "tide_mlp",
    "tsmixer_exog",
    "patchtst_ci_exog",
    "itransformer_exog",
    "timesnet_lite_exog",
    "stid_mlp",
    "graph_wavenet_lite",
    "mtgnn_lite",
    "agcrn_lite",
]

ALL_METHODS = [
    "repeat_last_value",
    "repeat_last_day",
    "linear_direct",
    "nlinear",
    "dlinear",
    "mlp_exog",
    "tide_mlp",
    "tsmixer_exog",
    "tcn_exog",
    "gru_exog",
    "lstm_exog",
    "transformer_exog",
    "patchtst_ci_exog",
    "itransformer_exog",
    "timesnet_lite_exog",
    "stid_mlp",
    "graph_wavenet_lite",
    "mtgnn_lite",
    "agcrn_lite",
]

def experiment_methods(group: str) -> List[str]:
    g = group.lower().strip()
    if g == "quick":
        return ["repeat_last_day", "dlinear", "nlinear", "patchtst_ci_exog", "itransformer_exog", "graph_wavenet_lite"]
    if g == "strong_core":
        return list(STRONG_CORE_METHODS)
    if g == "linear_mlp":
        return ["repeat_last_value", "repeat_last_day", "linear_direct", "nlinear", "dlinear", "mlp_exog", "tide_mlp", "tsmixer_exog"]
    if g == "sequence_transformer":
        return ["tcn_exog", "gru_exog", "lstm_exog", "transformer_exog", "patchtst_ci_exog", "itransformer_exog", "timesnet_lite_exog"]
    if g == "graph":
        return ["stid_mlp", "graph_wavenet_lite", "mtgnn_lite", "agcrn_lite"]
    if g == "strong_all":
        return list(ALL_METHODS)
    raise ValueError(f"Unknown experiment_group={group}. Valid: quick, strong_core, linear_mlp, sequence_transformer, graph, strong_all")


def build_model(method: str, cfg) -> BaseForecastModel:
    m = method.lower()
    if m == "repeat_last_value":
        return RepeatLastValue(cfg)
    if m == "repeat_last_day":
        return RepeatLastDay(cfg)
    if m == "linear_direct":
        return LinearDirect(cfg)
    if m == "nlinear":
        return NLinear(cfg)
    if m == "dlinear":
        return DLinear(cfg)
    if m == "mlp_exog":
        return MLPExog(cfg)
    if m == "tide_mlp":
        return TiDEMLP(cfg)
    if m == "tsmixer_exog":
        return TSMixerExog(cfg)
    if m == "tcn_exog":
        return TCNExog(cfg)
    if m == "gru_exog":
        return RNNExog(cfg, "gru")
    if m == "lstm_exog":
        return RNNExog(cfg, "lstm")
    if m == "transformer_exog":
        return TransformerExog(cfg)
    if m == "patchtst_ci_exog":
        return PatchTSTCIExog(cfg)
    if m == "itransformer_exog":
        return ITransformerExog(cfg)
    if m == "timesnet_lite_exog":
        return TimesNetLiteExog(cfg)
    if m == "stid_mlp":
        return STIDMLP(cfg)
    if m == "graph_wavenet_lite":
        return GraphWaveNetLite(cfg)
    if m == "mtgnn_lite":
        return MTGNNLite(cfg)
    if m == "agcrn_lite":
        return AGCRNLite(cfg)
    raise ValueError(f"Unknown method={method}. Valid methods: {ALL_METHODS}")


# =============================================================================
# Evaluation and training
# =============================================================================

@torch.no_grad()
def evaluate_model(model: BaseForecastModel, loader: DataLoader, cfg, base_mod, return_arrays: bool = False):
    model.eval()
    device = torch.device(cfg.device)

    total_abs = 0.0
    total_sq = 0.0
    total_abs_pct = 0.0
    total_count = 0
    total_mape_count = 0
    sum_y = 0.0
    sum_y2 = 0.0
    sum_res2 = 0.0

    y_true_list = []
    y_pred_list = []

    for batch in loader:
        x_fast = batch["x_fast"].to(device, non_blocking=True)
        x_slow = batch["x_slow"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mean = batch["target_mean"].to(device, non_blocking=True)
        scale = batch["target_scale"].to(device, non_blocking=True)

        out = model(x_fast, x_slow)
        pred_u = base_mod.inverse_scale_batch(out["pred"], mean, scale)
        y_u = base_mod.inverse_scale_batch(y, mean, scale)
        err = pred_u - y_u

        total_abs += float(err.abs().sum().detach().cpu())
        total_sq += float(err.pow(2).sum().detach().cpu())
        mask = y_u.abs() > 1e-8
        total_abs_pct += float(((err.abs() / y_u.abs().clamp_min(1e-8))[mask]).sum().detach().cpu())
        total_mape_count += int(mask.sum().detach().cpu())

        n = int(y_u.numel())
        total_count += n
        y_flat = y_u.reshape(-1)
        err_flat = err.reshape(-1)
        sum_y += float(y_flat.sum().detach().cpu())
        sum_y2 += float((y_flat ** 2).sum().detach().cpu())
        sum_res2 += float((err_flat ** 2).sum().detach().cpu())

        if return_arrays:
            y_true_list.append(y_u.detach().cpu().numpy())
            y_pred_list.append(pred_u.detach().cpu().numpy())

    mae = total_abs / max(total_count, 1)
    rmse = math.sqrt(total_sq / max(total_count, 1))
    mape = 100.0 * total_abs_pct / max(total_mape_count, 1)
    y_mean = sum_y / max(total_count, 1)
    ss_tot = sum_y2 - total_count * (y_mean ** 2)
    r2 = 1.0 - sum_res2 / max(ss_tot, 1e-12)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    if return_arrays:
        return metrics, np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)
    return metrics, None, None


def train_one_model(method: str, seed: int, cfg, base_mod, loaders=None) -> Dict[str, Any]:
    ensure_dir(cfg.run_dir)

    if cfg.resume and result_exists(cfg, method, seed):
        print(f"[SKIP] result exists for method={method}, seed={seed}")
        rr = pd.read_csv(os.path.join(cfg.run_dir, "running_results.csv"))
        row = rr[(rr["method"] == method) & (rr["seed"] == seed)].iloc[-1].to_dict()
        return row

    set_seed(seed)
    device = torch.device(cfg.device)
    if loaders is None:
        train_loader, val_loader, test_loader = make_loaders(base_mod, cfg)
    else:
        train_loader, val_loader, test_loader = loaders

    model = build_model(method, cfg).to(device)
    n_params = count_parameters(model)
    start = time.time()

    if method in NON_TRAINABLE_METHODS:
        val_metrics, _, _ = evaluate_model(model, val_loader, cfg, base_mod)
        test_metrics, _, _ = evaluate_model(model, test_loader, cfg, base_mod)
        row = {
            "method": method,
            "seed": seed,
            "best_epoch": 0,
            "best_val_RMSE": val_metrics["RMSE"],
            "train_time_sec": 0.0,
            "train_time_per_epoch_sec": 0.0,
            "total_params_million": n_params / 1e6,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        append_result(cfg, row)
        print(f"[TEST][NONTRAIN] {method} seed={seed}: {test_metrics}")
        return row

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    best_rmse = float("inf")
    best_epoch = -1
    bad = 0
    best_path = checkpoint_path(cfg, method, seed)
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x_fast = batch["x_fast"].to(device, non_blocking=True)
            x_slow = batch["x_slow"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
                out = model(x_fast, x_slow)
                loss = F.mse_loss(out["pred"], y)

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            ep_loss += float(loss.detach().cpu())
            ep_batches += 1

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        val_metrics, _, _ = evaluate_model(model, val_loader, cfg, base_mod)
        mean_loss = ep_loss / max(ep_batches, 1)
        print(
            f"[{method}][seed={seed}] epoch={epoch:03d} "
            f"loss={mean_loss:.6f} val_RMSE={val_metrics['RMSE']:.4f} "
            f"val_MAE={val_metrics['MAE']:.4f} time={epoch_time:.1f}s"
        )

        if val_metrics["RMSE"] < best_rmse:
            best_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "cfg": cfg.__dict__ if not hasattr(cfg, "__dataclass_fields__") else asdict(cfg),
                "val_metrics": val_metrics,
                "total_params": n_params,
            }, best_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[EARLY STOP] {method} seed={seed} at epoch={epoch}, best_epoch={best_epoch}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, _, _ = evaluate_model(model, test_loader, cfg, base_mod)

    elapsed = time.time() - start
    row = {
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse,
        "train_time_sec": elapsed,
        "train_time_per_epoch_sec": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "total_params_million": n_params / 1e6,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    append_result(cfg, row)
    print(f"[TEST] {method} seed={seed}: {test_metrics}, params={n_params/1e6:.3f}M, elapsed={elapsed/60:.1f} min")
    return row


def summarize_results(cfg) -> None:
    path = os.path.join(cfg.run_dir, "running_results.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    df.to_csv(os.path.join(cfg.run_dir, "all_seed_results_detailed.csv"), index=False)

    metric_cols = [
        "test_MAE", "test_RMSE", "test_MAPE", "test_R2",
        "train_time_per_epoch_sec", "total_params_million"
    ]
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method, "num_seeds": len(g)}
        for c in metric_cols:
            if c in g.columns:
                row[c + "_mean"] = float(g[c].mean())
                row[c + "_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if "test_RMSE_mean" in out.columns:
        out = out.sort_values("test_RMSE_mean")
    out.to_csv(os.path.join(cfg.run_dir, "all_seed_results_mean_std.csv"), index=False)

    paper_rows = []
    for _, r in out.iterrows():
        paper_rows.append({
            "Method": r["method"],
            "MAE": f"{r.get('test_MAE_mean', np.nan):.2f} ± {r.get('test_MAE_std', 0):.2f}",
            "RMSE": f"{r.get('test_RMSE_mean', np.nan):.2f} ± {r.get('test_RMSE_std', 0):.2f}",
            "MAPE(%)": f"{r.get('test_MAPE_mean', np.nan):.2f} ± {r.get('test_MAPE_std', 0):.2f}",
            "R2": f"{r.get('test_R2_mean', np.nan):.4f} ± {r.get('test_R2_std', 0):.4f}",
            "Time/epoch(s)": f"{r.get('train_time_per_epoch_sec_mean', np.nan):.2f}",
            "Params(M)": f"{r.get('total_params_million_mean', np.nan):.3f}",
        })
    pd.DataFrame(paper_rows).to_csv(os.path.join(cfg.run_dir, "paper_strong_baseline_table_mean_std.csv"), index=False)


def run_experiments(base_mod, cfg, methods: List[str]) -> None:
    ensure_dir(cfg.run_dir)
    with open(os.path.join(cfg.run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        cfg_dict = cfg.__dict__ if not hasattr(cfg, "__dataclass_fields__") else asdict(cfg)
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.run_dir, "baseline_method_list.json"), "w", encoding="utf-8") as f:
        json.dump({"methods": methods, "all_available_methods": ALL_METHODS}, f, indent=2, ensure_ascii=False)

    if hasattr(base_mod, "check_processed_data"):
        base_mod.check_processed_data(cfg)

    loaders = make_loaders(base_mod, cfg)

    for method in methods:
        for seed in cfg.seeds:
            train_one_model(method, seed, cfg, base_mod, loaders=loaders)
            summarize_results(cfg)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Strong baselines for H288-P288 multi-zone load forecasting.")
    parser.add_argument("--base_script", type=str, default="two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py")
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--experiment_tag", type=str, default="strong_baselines")
    parser.add_argument("--experiment_group", type=str, default="strong_core", help="Default is strong_core. Other groups are retained for compatibility.")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="2026,2027,2028,2029,2030")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--zone_emb_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--sample_mode", type=str, default="uniform", choices=["first", "uniform", "random"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    base_mod = import_base_module(args.base_script)
    cfg = base_mod.Config()
    if args.save_root is not None:
        cfg.save_root = args.save_root
    cfg.experiment_tag = args.experiment_tag
    cfg.seeds = parse_seeds(args.seeds)
    cfg.epochs = int(args.epochs)
    cfg.patience = int(args.patience)
    cfg.batch_size = int(args.batch_size)
    cfg.lr = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.hidden_dim = int(args.hidden_dim)
    cfg.zone_emb_dim = int(args.zone_emb_dim)
    cfg.dropout = float(args.dropout)
    cfg.grad_clip = float(args.grad_clip)
    cfg.num_workers = int(args.num_workers)
    cfg.max_train_samples = int(args.max_train_samples)
    cfg.max_eval_samples = int(args.max_eval_samples)
    cfg.sample_mode = args.sample_mode
    cfg.resume = bool(args.resume)
    cfg.use_amp = bool(args.amp)

    if args.device is not None:
        cfg.device = args.device if args.device.startswith("cuda") or args.device == "cpu" else f"cuda:{args.device}"

    if args.methods.strip():
        methods = parse_methods(args.methods)
    else:
        methods = experiment_methods(args.experiment_group)

    print("=" * 90)
    print("Strong baseline experiment")
    print(f"save_dir={cfg.save_dir}")
    print(f"run_dir ={cfg.run_dir}")
    print(f"methods={methods}")
    print(f"seeds={cfg.seeds}")
    print(f"device={cfg.device}, amp={cfg.use_amp}")
    print(f"hist_len={cfg.hist_len}, pred_len={cfg.pred_len}, zones={cfg.num_zones}")
    print("=" * 90)

    run_experiments(base_mod, cfg, methods)


if __name__ == "__main__":
    main()
