# -*- coding: utf-8 -*-
"""
SGR-PG-DTSF: Semantic Graph Residual Prediction-Guided Dual-Timescale Forecasting
================================================================================

This script implements a synchronized multi-zone experiment on top of the already
built two-and-a-half-year per-zone samples:

    processed/two_half_year_201601_201807_per_zone_scalers_H288_P288/
        train_by_zone/train_Xf_<zone>.npy, train_Xs_<zone>.npy, train_Y_<zone>.npy
        val_by_zone/...
        test_by_zone/...
        scalers_by_zone/<zone>/target_mean.npy, target_scale.npy

The script does NOT need to rebuild raw features.  It only changes the data loader
from independent zone windows [B, L, F] to synchronized multi-zone windows
[B, Z, L, F].  The key idea is to model the future forecasting error as a semantic
graph residual field across NYISO zones rather than independent zone-wise residuals.

Main methods with unambiguous names
-------------------------------------
    sgr_pg_dtsf_full
        Full SGR-PG-DTSF with base dual-timescale prediction, dynamic residual graph,
        learned graph prior, slow variables, fast load history, and zone identity.

    base_dual_timescale_only
        Uses only the base dual-timescale predictor. Residual correction and graph
        propagation are disabled. This is not a naive baseline; it still uses fast,
        slow, and zone inputs.

    graph_local_residual_only
        Keeps local residual correction but disables cross-zone graph propagation.

    graph_static_learned_prior
        Replaces the dynamic graph with a learned static graph prior.

    input_no_slow_variables
        True slow-input ablation. The slow-variable tensor is zeroed before the slow
        encoder, so both base prediction and residual graph conditioning cannot use
        real slow variables.

    input_no_fast_load_history
        True fast-load-history ablation. The historical load channel of x_fast is
        zeroed before the fast encoder; hour_sin/hour_cos are retained. Residual
        graph fast-state conditioning is also disabled.

    input_no_zone_identity
        True explicit-zone-identity ablation. Zone embeddings are zeroed in both the
        base predictor and residual graph conditioning, and the learned zone-pair
        graph prior is disabled.

    cond_no_slow_state
        Residual-graph conditioning ablation only. Slow variables are still used in
        the base predictor, but slow-effect statistics and slow representation are
        removed from the residual graph condition.

    cond_no_fast_state
        Residual-graph conditioning ablation only. Fast load history is still used
        by the base predictor, but fast-state statistics and fast representation are
        removed from residual graph conditioning.

    cond_no_zone_state
        Residual-graph conditioning ablation only. Zone embeddings are still used
        by the base predictor and learned graph prior, but not by the residual graph
        condition vector.

    cond_no_base_forecast_morphology
        Removes base-forecast morphology statistics only from the prediction state.

Legacy ambiguous names such as sgr_no_slow_state, sgr_no_fast_state,
sgr_no_dynamic_graph, and sgr_no_graph_propagation are intentionally rejected.

Typical use
-----------
Smoke test:
    python two_half_year_sgr_pg_dtsf_experiments.py --save_root ".\\processed" \
        --methods "sgr_pg_dtsf" --experiment_tag sgr_smoke --device cuda --seeds "2026" \
        --epochs 3 --patience 2 --max_train_samples 5000 --max_eval_samples 2000 --sample_mode uniform

Full ablation:
    python two_half_year_sgr_pg_dtsf_experiments.py --save_root ".\\processed" \
        --experiment_group all_clear_ablation --experiment_tag clear_ablation_5seeds \
        --device cuda --seeds "2026,2027,2028,2029,2030" --epochs 30 --patience 6 --batch_size 64 \
        --lr 0.0005 --weight_decay 0.00001 --hidden_dim 128 --sgr_graph_layers 2 --amp --resume --no_save_predictions

Author note
-----------
This is a self-contained model and experiment entrypoint. It consumes preprocessed inputs and does not implement raw-data preparation.
"""

import os
import json
import time
import math
import random
import argparse
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

VALID_ZONES = [
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL",
    "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"
]

FAST_FEATURES = ["load", "hour_sin", "hour_cos"]
SLOW_FEATURES = [
    "season_id", "dayofweek", "is_weekend", "is_us_holiday",
    "max_temp", "min_temp", "max_wet_bulb", "min_wet_bulb",
    "mean_temp", "mean_wet_bulb", "temp_range", "wet_bulb_range", "mean_temp_wb_gap"
]


@dataclass
class Config:
    # Existing processed data root.  The script uses the per-zone files inside save_dir.
    save_root: str = r"./data/processed"

    hist_len: int = 288
    pred_len: int = 288
    fast_dim: int = 3
    slow_dim: int = 13
    num_zones: int = len(VALID_ZONES)

    # training
    seeds: Tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False

    # model
    hidden_dim: int = 128
    zone_emb_dim: int = 16
    dropout: float = 0.1
    base_loss_weight: float = 0.05
    residual_loss_weight: float = 0.10
    correction_l2_weight: float = 1e-4

    # SGR graph residual field
    sgr_graph_layers: int = 2
    sgr_graph_dim: int = 64
    sgr_graph_alpha: float = 0.50
    sgr_temperature: float = 1.00
    sgr_self_loop_bias: float = 1.00
    sgr_graph_dropout: float = 0.05
    sgr_residual_scale: float = 1.00
    sgr_gate_init_bias: float = -1.50
    sgr_graph_sparsity_weight: float = 0.0
    sgr_graph_entropy_weight: float = 0.0

    # run control
    run_experiments: bool = True
    save_predictions: bool = False
    resume: bool = False
    experiment_tag: str = "sgr_pg_dtsf"
    max_train_samples: int = 0
    max_eval_samples: int = 0
    sample_mode: str = "uniform"  # first | uniform | random

    # analysis
    run_sgr_analysis: bool = False
    sgr_analysis_method: str = "sgr_pg_dtsf_full"
    sgr_analysis_seed: int = 2026
    sgr_analysis_max_samples: int = 30000

    @property
    def save_dir(self) -> str:
        return os.path.join(
            self.save_root,
            "two_half_year_201601_201807_per_zone_scalers_H288_P288"
        )

    @property
    def run_dir(self) -> str:
        return os.path.join(self.save_dir, f"H288_P288_{self.experiment_tag}")


# =========================================================
# 2. Utilities
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


def parse_seeds(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def load_target_scalers(save_dir: str) -> Dict[str, Tuple[float, float]]:
    scaler_dir = os.path.join(save_dir, "scalers_by_zone")
    out: Dict[str, Tuple[float, float]] = {}
    for zone in VALID_ZONES:
        zsafe = zone_safe_name(zone)
        zdir = os.path.join(scaler_dir, zsafe)
        mean_path = os.path.join(zdir, "target_mean.npy")
        scale_path = os.path.join(zdir, "target_scale.npy")
        if not (os.path.exists(mean_path) and os.path.exists(scale_path)):
            raise FileNotFoundError(
                f"Missing target scaler for zone={zone}. Expected {mean_path} and {scale_path}"
            )
        out[zone] = (float(np.load(mean_path)[0]), float(np.load(scale_path)[0]))
    return out


def check_processed_data(cfg: Config) -> None:
    missing = []
    for split in ["train", "val", "test"]:
        base_dir = os.path.join(cfg.save_dir, f"{split}_by_zone")
        for zone in VALID_ZONES:
            zsafe = zone_safe_name(zone)
            for kind in ["Xf", "Xs", "Y"]:
                p = os.path.join(base_dir, f"{split}_{kind}_{zsafe}.npy")
                if not os.path.exists(p):
                    missing.append(p)
    if missing:
        msg = "\n".join(missing[:10])
        raise FileNotFoundError(
            "Per-zone processed data not found. This script does not rebuild raw CSV features.\n"
            "Run one of the previous two-half-year data-building scripts first, then pass.\n"
            f"First missing files:\n{msg}"
        )


def compute_metrics_flat(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0) if np.any(mask) else float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# =========================================================
# 3. Multi-zone synchronized dataset
# =========================================================

class MultiZoneSynchronizedDataset(Dataset):
    """Lazy synchronized multi-zone dataset.

    Output shapes:
        x_fast: [Z, L, F]
        x_slow: [Z, S]
        y     : [Z, P]
        target_mean / target_scale: [Z]

    It assumes each zone's per-zone samples were generated from the same global
    timestamp grid and are aligned by local index.  If lengths differ slightly,
    the minimum length across all zones is used.
    """

    def __init__(self, cfg: Config, split: str, max_samples: int = 0):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg = cfg
        self.split = split
        self.base_dir = os.path.join(cfg.save_dir, f"{split}_by_zone")
        self.scalers = load_target_scalers(cfg.save_dir)

        self.xf_arrays: List[np.ndarray] = []
        self.xs_arrays: List[np.ndarray] = []
        self.y_arrays: List[np.ndarray] = []
        self.t_arrays: List[Optional[np.ndarray]] = []
        self.zone_means: List[float] = []
        self.zone_scales: List[float] = []
        self.lengths: List[int] = []

        for zone in VALID_ZONES:
            zsafe = zone_safe_name(zone)
            xfp = os.path.join(self.base_dir, f"{split}_Xf_{zsafe}.npy")
            xsp = os.path.join(self.base_dir, f"{split}_Xs_{zsafe}.npy")
            yp = os.path.join(self.base_dir, f"{split}_Y_{zsafe}.npy")
            tp = os.path.join(self.base_dir, f"{split}_T_{zsafe}.npy")
            if not (os.path.exists(xfp) and os.path.exists(xsp) and os.path.exists(yp)):
                raise FileNotFoundError(f"Missing synchronized zone files for split={split}, zone={zone}")
            Xf = np.load(xfp, mmap_mode="r")
            Xs = np.load(xsp, mmap_mode="r")
            Y = np.load(yp, mmap_mode="r")
            if Xf.shape[0] != Xs.shape[0] or Xf.shape[0] != Y.shape[0]:
                raise ValueError(f"Length mismatch for split={split}, zone={zone}: {Xf.shape}, {Xs.shape}, {Y.shape}")
            self.xf_arrays.append(Xf)
            self.xs_arrays.append(Xs)
            self.y_arrays.append(Y)
            self.lengths.append(int(Xf.shape[0]))
            T = None
            if os.path.exists(tp):
                try:
                    T = np.load(tp, allow_pickle=True)
                except Exception:
                    T = None
            self.t_arrays.append(T)
            mean, scale = self.scalers[zone]
            self.zone_means.append(float(mean))
            self.zone_scales.append(float(scale))

        self.min_len = int(min(self.lengths))
        if len(set(self.lengths)) != 1:
            print(f"[WARN] zone lengths differ in {split}: {dict(zip(VALID_ZONES, self.lengths))}. Using min_len={self.min_len}")

        if max_samples and max_samples > 0 and self.min_len > max_samples:
            if cfg.sample_mode == "uniform":
                self.indices = np.linspace(0, self.min_len - 1, num=max_samples, dtype=np.int64)
            elif cfg.sample_mode == "random":
                rng = np.random.default_rng(2026)
                self.indices = np.sort(rng.choice(self.min_len, size=max_samples, replace=False)).astype(np.int64)
            else:
                self.indices = np.arange(max_samples, dtype=np.int64)
        else:
            self.indices = np.arange(self.min_len, dtype=np.int64)

        self.zone_means_np = np.array(self.zone_means, dtype=np.float32)
        self.zone_scales_np = np.array(self.zone_scales, dtype=np.float32)
        print(
            f"[MULTIZONE DATASET] {split}: samples={len(self.indices)}, "
            f"Z={len(VALID_ZONES)}, Xf=[Z,{cfg.hist_len},{cfg.fast_dim}], Xs=[Z,{cfg.slow_dim}], Y=[Z,{cfg.pred_len}] [lazy mmap]"
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        local = int(self.indices[idx])
        xf_list = []
        xs_list = []
        y_list = []
        for z in range(len(VALID_ZONES)):
            xf_list.append(np.array(self.xf_arrays[z][local], dtype=np.float32, copy=True))
            xs_list.append(np.array(self.xs_arrays[z][local], dtype=np.float32, copy=True))
            y_list.append(np.array(self.y_arrays[z][local], dtype=np.float32, copy=True))
        T = self.t_arrays[0]
        if T is not None:
            try:
                timestamp = str(T[local])
            except Exception:
                timestamp = f"{self.split}_{local}"
        else:
            timestamp = f"{self.split}_{local}"
        return {
            "x_fast": torch.from_numpy(np.stack(xf_list, axis=0)),
            "x_slow": torch.from_numpy(np.stack(xs_list, axis=0)),
            "y": torch.from_numpy(np.stack(y_list, axis=0)),
            "target_mean": torch.from_numpy(self.zone_means_np.copy()),
            "target_scale": torch.from_numpy(self.zone_scales_np.copy()),
            "timestamp": timestamp,
            "local_index": torch.tensor(local, dtype=torch.long),
        }


# =========================================================
# 4. Neural modules
# =========================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastGRUEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq, h = self.gru(x)
        last = self.norm(h[-1])
        return seq, last


def last_linear(module: nn.Module) -> Optional[nn.Linear]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


# =========================================================
# 5. SGR-PG-DTSF model
# =========================================================

SGR_BASE_FLAGS = {
    # Residual graph architecture switches
    "use_residual": True,
    "use_graph_propagation": True,
    "use_dynamic_graph": True,
    "use_learned_graph_prior": True,

    # True input switches. These control whether the corresponding information
    # is available to the base predictor and all downstream modules.
    "use_slow_input_in_base": True,
    "use_fast_load_input": True,
    "use_zone_embedding": True,

    # Residual-graph conditioning switches. These do NOT remove the input from
    # the base predictor; they only remove the state from residual graph modules.
    "condition_slow_state": True,
    "condition_fast_state": True,
    "condition_zone_state": True,
    "condition_base_morphology": True,
}


METHOD_DESCRIPTIONS = {
    "sgr_pg_dtsf_full": (
        "Full model: base dual-timescale predictor + local residual + dynamic residual graph + "
        "learned graph prior + slow variables + fast load history + zone identity."
    ),
    "base_dual_timescale_only": (
        "Base predictor only. Residual correction and graph propagation are disabled; "
        "fast, slow, and zone inputs are still used by the base predictor."
    ),
    "graph_local_residual_only": (
        "Local residual correction only. Cross-zone graph propagation is disabled."
    ),
    "graph_static_learned_prior": (
        "Static graph residual model. The dynamic graph query-key term is disabled, "
        "and residual propagation uses a learned static graph prior."
    ),
    "input_no_slow_variables": (
        "True slow-input ablation. The slow-variable tensor is zeroed before the slow encoder; "
        "neither the base predictor nor the residual graph sees real slow variables."
    ),
    "input_no_fast_load_history": (
        "True fast-load-history ablation. The historical load channel is zeroed before the fast encoder; "
        "hour_sin/hour_cos are retained, and fast-state residual conditioning is disabled."
    ),
    "input_no_zone_identity": (
        "True explicit-zone-identity ablation. Zone embeddings are zeroed in the base predictor and "
        "residual graph condition; the learned zone-pair graph prior is also disabled."
    ),
    "cond_no_slow_state": (
        "Residual-graph conditioning ablation. Slow variables remain in the base predictor, "
        "but slow-effect statistics and slow representation are removed from residual graph conditioning."
    ),
    "cond_no_fast_state": (
        "Residual-graph conditioning ablation. Fast load history remains in the base predictor, "
        "but fast-shape statistics and fast representation are removed from residual graph conditioning."
    ),
    "cond_no_zone_state": (
        "Residual-graph conditioning ablation. Zone identity remains in the base predictor and graph prior, "
        "but zone embedding is removed from residual graph conditioning."
    ),
    "cond_no_base_forecast_morphology": (
        "Prediction-state ablation. Base forecast morphology statistics are removed from the residual state; "
        "the base forecast itself is still used."
    ),
}


LEGACY_AMBIGUOUS_NAMES = {
    "sgr_no_slow_state": "Use input_no_slow_variables for true slow-input ablation, or cond_no_slow_state for residual-graph conditioning ablation.",
    "sgr_no_fast_state": "Use input_no_fast_load_history for true fast-load-history ablation, or cond_no_fast_state for residual-graph conditioning ablation.",
    "sgr_no_zone_state": "Use input_no_zone_identity for true explicit-zone-identity ablation, or cond_no_zone_state for residual-graph conditioning ablation.",
    "sgr_no_base_morphology": "Use cond_no_base_forecast_morphology.",
    "sgr_base_only": "Use base_dual_timescale_only.",
    "base_only": "Use base_dual_timescale_only.",
    "sgr_static_graph": "Use graph_static_learned_prior.",
    "sgr_no_dynamic_graph": "Use graph_static_learned_prior. The old names were duplicate code paths.",
    "static_graph_residual": "Use graph_static_learned_prior.",
    "sgr_local_psr": "Use graph_local_residual_only.",
    "sgr_no_graph_propagation": "Use graph_local_residual_only. The old names were duplicate code paths.",
    "sgr_no_graph_prop": "Use graph_local_residual_only.",
    "local_psr": "Use graph_local_residual_only.",
}


def method_description(method: str) -> str:
    name = method.lower()
    if name == "sgr_pg_dtsf":
        name = "sgr_pg_dtsf_full"
    return METHOD_DESCRIPTIONS.get(name, "")


def sgr_flags_from_name(method: str) -> Dict[str, bool]:
    name = method.lower()
    if name == "sgr_pg_dtsf":
        name = "sgr_pg_dtsf_full"

    if name in LEGACY_AMBIGUOUS_NAMES:
        raise ValueError(
            f"Ambiguous legacy method name: {method}. {LEGACY_AMBIGUOUS_NAMES[name]}"
        )

    flags = dict(SGR_BASE_FLAGS)

    if name in ["sgr_pg_dtsf_full", "sgr", "proposed_sgr"]:
        return flags

    if name == "base_dual_timescale_only":
        flags["use_residual"] = False
        flags["use_graph_propagation"] = False

    elif name == "graph_local_residual_only":
        flags["use_graph_propagation"] = False

    elif name == "graph_static_learned_prior":
        flags["use_dynamic_graph"] = False

    elif name == "input_no_slow_variables":
        flags["use_slow_input_in_base"] = False
        flags["condition_slow_state"] = False

    elif name == "input_no_fast_load_history":
        flags["use_fast_load_input"] = False
        flags["condition_fast_state"] = False

    elif name == "input_no_zone_identity":
        flags["use_zone_embedding"] = False
        flags["condition_zone_state"] = False
        flags["use_learned_graph_prior"] = False

    elif name == "cond_no_slow_state":
        flags["condition_slow_state"] = False

    elif name == "cond_no_fast_state":
        flags["condition_fast_state"] = False

    elif name == "cond_no_zone_state":
        flags["condition_zone_state"] = False

    elif name == "cond_no_base_forecast_morphology":
        flags["condition_base_morphology"] = False

    else:
        valid = sorted(METHOD_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown SGR variant: {method}. Valid unambiguous names are: {valid}")
    return flags


class SGRPGDTSF(nn.Module):
    """Semantic Graph Residual PG-DTSF.

    SGR models multi-zone forecasting errors as a graph residual field:
        1) A zone-wise dual-timescale base predictor gives y_base[z].
        2) A local prediction-state residual gives r_z^(0).
        3) A dynamic semantic graph A(s_i, s_j) propagates residual information
           across zones, producing a multi-zone residual field r_z^(L).
        4) y_hat[z] = y_base[z] + r_z^(L).
    """

    def __init__(self, cfg: Config, method: str = "sgr_pg_dtsf_full"):
        super().__init__()
        self.cfg = cfg
        self.method = method
        self.flags = sgr_flags_from_name(method)
        self.Z = int(cfg.num_zones)
        self.P = int(cfg.pred_len)

        self.fast = FastGRUEncoder(cfg.fast_dim, cfg.hidden_dim, cfg.dropout)
        self.slow = MLP(cfg.slow_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.zone_emb = nn.Embedding(self.Z, cfg.zone_emb_dim)
        self.base_head = MLP(cfg.hidden_dim * 2 + cfg.zone_emb_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

        # state = base shape stats + slow intervention effect stats + recent fast shape stats
        self.curve_stats_dim = 8
        self.state_raw_dim = self.curve_stats_dim * 3
        self.state_norm = nn.LayerNorm(self.state_raw_dim)
        self.state_proj = MLP(self.state_raw_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)

        self.cond_dim = cfg.hidden_dim * 3 + cfg.zone_emb_dim  # state_emb, hf, hs, zone
        self.local_residual = MLP(self.cond_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)
        self.local_gate = MLP(self.cond_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout)

        graph_in_dim = self.cond_dim
        self.graph_q = nn.Linear(graph_in_dim, cfg.sgr_graph_dim)
        self.graph_k = nn.Linear(graph_in_dim, cfg.sgr_graph_dim)
        prior = torch.zeros(self.Z, self.Z)
        prior.fill_(0.0)
        prior += torch.eye(self.Z) * float(cfg.sgr_self_loop_bias)
        self.graph_prior = nn.Parameter(prior)

        self.graph_update_heads = nn.ModuleList([
            MLP(self.cond_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout) for _ in range(cfg.sgr_graph_layers)
        ])
        self.graph_gate_heads = nn.ModuleList([
            MLP(self.cond_dim, cfg.hidden_dim, cfg.pred_len, cfg.dropout) for _ in range(cfg.sgr_graph_layers)
        ])
        self.graph_dropout = nn.Dropout(cfg.sgr_graph_dropout)

        self._init_residual_and_gates()

    def _init_residual_and_gates(self) -> None:
        # Start close to the base forecast/local residual with conservative graph correction.
        for mod in [self.local_residual] + list(self.graph_update_heads):
            layer = last_linear(mod)
            if layer is not None:
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for mod in [self.local_gate] + list(self.graph_gate_heads):
            layer = last_linear(mod)
            if layer is not None and layer.bias is not None:
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, float(self.cfg.sgr_gate_init_bias))

    @staticmethod
    def curve_stats(x: torch.Tensor) -> torch.Tensor:
        # x: [B, Z, H]
        if x.shape[-1] > 1:
            ramp = torch.abs(x[..., 1:] - x[..., :-1]).mean(dim=-1)
        else:
            ramp = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        peak_pos = torch.argmax(x, dim=-1).to(dtype=x.dtype)
        if x.shape[-1] > 1:
            peak_pos = peak_pos / float(x.shape[-1] - 1)
        return torch.stack([
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            x[..., 0],
            x[..., -1],
            x.max(dim=-1).values,
            x.min(dim=-1).values,
            peak_pos,
            ramp,
        ], dim=-1)

    def _encode(self, x_fast: torch.Tensor, x_slow: torch.Tensor):
        # x_fast [B,Z,L,F], x_slow [B,Z,S]
        # True input ablations are applied before the encoders, so method names
        # correspond exactly to the information available to the base predictor.
        if not self.flags["use_fast_load_input"]:
            # Only the standardized load channel is removed. Calendar time encodings
            # hour_sin/hour_cos are retained; hence the method name is
            # input_no_fast_load_history rather than no_fast_input.
            x_fast_model = x_fast.clone()
            x_fast_model[..., 0] = 0.0
        else:
            x_fast_model = x_fast

        if not self.flags["use_slow_input_in_base"]:
            x_slow_model = torch.zeros_like(x_slow)
        else:
            x_slow_model = x_slow

        B, Z, L, Fdim = x_fast_model.shape
        xf = x_fast_model.reshape(B * Z, L, Fdim)
        xs = x_slow_model.reshape(B * Z, x_slow_model.shape[-1])
        _, hf = self.fast(xf)
        hs = self.slow(xs)

        zids = torch.arange(Z, device=x_fast.device, dtype=torch.long).unsqueeze(0).expand(B, Z).reshape(B * Z)
        ez_raw = self.zone_emb(zids)
        if not self.flags["use_zone_embedding"]:
            ez_raw = torch.zeros_like(ez_raw)

        hf = hf.reshape(B, Z, -1)
        hs = hs.reshape(B, Z, -1)
        ez = ez_raw.reshape(B, Z, -1)
        return hf, hs, ez, x_fast_model, x_slow_model

    def _base_forecast(self, hf: torch.Tensor, hs: torch.Tensor, ez: torch.Tensor) -> torch.Tensor:
        B, Z, _ = hf.shape
        out = self.base_head(torch.cat([hf, hs, ez], dim=-1).reshape(B * Z, -1))
        return out.reshape(B, Z, self.P)

    def _build_prediction_state(
        self,
        base_pred: torch.Tensor,
        slow_effect: torch.Tensor,
        x_fast_model: torch.Tensor,
    ) -> torch.Tensor:
        # base_shape and fast_shape are relative to the last observed standardized
        # load used by the model. If input_no_fast_load_history is active, this is
        # already the zeroed load channel.
        last_load = x_fast_model[:, :, -1, 0]
        base_shape = base_pred - last_load.unsqueeze(-1)
        fast_shape = x_fast_model[:, :, :, 0] - last_load.unsqueeze(-1)

        if not self.flags["condition_base_morphology"]:
            base_shape = torch.zeros_like(base_shape)
        if not self.flags["condition_slow_state"]:
            slow_effect = torch.zeros_like(slow_effect)
        if not self.flags["condition_fast_state"]:
            fast_shape = torch.zeros_like(fast_shape)

        raw = torch.cat([
            self.curve_stats(base_shape),
            self.curve_stats(slow_effect),
            self.curve_stats(fast_shape),
        ], dim=-1)
        return raw

    def _condition(self, raw_state, hf, hs, ez):
        B, Z, _ = hf.shape
        state_emb = self.state_proj(self.state_norm(raw_state).reshape(B * Z, -1)).reshape(B, Z, -1)
        hf_use = hf if self.flags["condition_fast_state"] else torch.zeros_like(hf)
        hs_use = hs if self.flags["condition_slow_state"] else torch.zeros_like(hs)
        ez_use = ez if self.flags["condition_zone_state"] else torch.zeros_like(ez)
        cond = torch.cat([state_emb, hf_use, hs_use, ez_use], dim=-1)
        return cond, state_emb

    def _dynamic_graph(self, cond: torch.Tensor) -> torch.Tensor:
        # cond [B,Z,C]
        B, Z, _ = cond.shape
        if self.flags["use_learned_graph_prior"]:
            prior = self.graph_prior
        else:
            prior = torch.zeros_like(self.graph_prior)

        if self.flags["use_dynamic_graph"]:
            q = self.graph_q(cond)
            k = self.graph_k(cond)
            scores = torch.einsum("bzd,bjd->bzj", q, k) / math.sqrt(float(q.shape[-1]))
            scores = scores / max(float(self.cfg.sgr_temperature), 1e-6)
            scores = scores + prior.unsqueeze(0)
        else:
            scores = prior.unsqueeze(0).expand(B, Z, Z)
        A = F.softmax(scores, dim=-1)
        return A

    def _local_residual(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Z, _ = cond.shape
        flat = cond.reshape(B * Z, -1)
        res = self.local_residual(flat).reshape(B, Z, self.P)
        gate = torch.sigmoid(self.local_gate(flat)).reshape(B, Z, self.P)
        return float(self.cfg.sgr_residual_scale) * gate * res, gate

    def _graph_propagate(self, r0: torch.Tensor, cond: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.flags["use_graph_propagation"]:
            return r0, []
        B, Z, _ = cond.shape
        r = r0
        gates = []
        for upd_head, gate_head in zip(self.graph_update_heads, self.graph_gate_heads):
            neigh = torch.einsum("bij,bjp->bip", A, r)
            flat = cond.reshape(B * Z, -1)
            local_update = upd_head(flat).reshape(B, Z, self.P)
            gate = torch.sigmoid(gate_head(flat)).reshape(B, Z, self.P)
            gates.append(gate)
            delta = (neigh - r) + 0.10 * local_update
            delta = self.graph_dropout(delta)
            r = r + float(self.cfg.sgr_graph_alpha) * gate * delta
        return r, gates

    def forward(self, x_fast: torch.Tensor, x_slow: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hf, hs, ez, x_fast_model, x_slow_model = self._encode(x_fast, x_slow)
        base_pred = self._base_forecast(hf, hs, ez)

        # Counterfactual-style slow-context intervention for prediction state only.
        # If input_no_slow_variables is active, x_slow_model is already all zeros,
        # so cf_pred equals the base slow-input setting and slow_effect is zero.
        hs_zero = self.slow(torch.zeros_like(x_slow_model.reshape(-1, x_slow_model.shape[-1]))).reshape_as(hs)
        cf_pred = self._base_forecast(hf, hs_zero, ez)
        slow_effect = base_pred - cf_pred

        raw_state = self._build_prediction_state(base_pred, slow_effect, x_fast_model)
        cond, state_emb = self._condition(raw_state, hf, hs, ez)

        if self.flags["use_residual"]:
            r0, local_gate = self._local_residual(cond)
            A = self._dynamic_graph(cond)
            residual, graph_gates = self._graph_propagate(r0, cond, A)
            pred = base_pred + residual
        else:
            r0 = torch.zeros_like(base_pred)
            residual = torch.zeros_like(base_pred)
            local_gate = torch.zeros_like(base_pred)
            A = torch.eye(self.Z, device=x_fast.device, dtype=x_fast.dtype).unsqueeze(0).expand(x_fast.shape[0], self.Z, self.Z)
            graph_gates = []
            pred = base_pred

        return {
            "pred": pred,
            "base_pred": base_pred,
            "cf_pred": cf_pred,
            "slow_effect": slow_effect,
            "prediction_state": raw_state,
            "state_emb": state_emb,
            "local_residual": r0,
            "residual": residual,
            "local_gate": local_gate,
            "graph_adj": A,
            "graph_gates": graph_gates,
        }

    def compute_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_loss = F.mse_loss(out["pred"], y)
        base_loss = F.mse_loss(out["base_pred"], y)
        if self.flags["use_residual"]:
            residual_target = (y - out["base_pred"]).detach()
            residual_loss = F.mse_loss(out["residual"], residual_target)
            correction_l2 = out["residual"].pow(2).mean()
        else:
            residual_loss = pred_loss.new_tensor(0.0)
            correction_l2 = pred_loss.new_tensor(0.0)

        A = out["graph_adj"]
        # Optional graph regularizers.  By default disabled; use after performance is verified.
        graph_l1 = A.abs().mean()
        entropy = -(A * torch.log(A.clamp_min(1e-8))).sum(dim=-1).mean()

        loss = (
            pred_loss
            + float(self.cfg.base_loss_weight) * base_loss
            + float(self.cfg.residual_loss_weight) * residual_loss
            + float(self.cfg.correction_l2_weight) * correction_l2
            + float(self.cfg.sgr_graph_sparsity_weight) * graph_l1
            - float(self.cfg.sgr_graph_entropy_weight) * entropy
        )
        logs = {
            "pred_loss": float(pred_loss.detach().cpu()),
            "base_loss": float(base_loss.detach().cpu()),
            "residual_loss": float(residual_loss.detach().cpu()),
            "correction_l2": float(correction_l2.detach().cpu()),
            "graph_l1": float(graph_l1.detach().cpu()),
            "graph_entropy": float(entropy.detach().cpu()),
            "local_gate_mean": float(out["local_gate"].detach().mean().cpu()),
            "graph_self_weight": float(torch.diagonal(A, dim1=-2, dim2=-1).mean().detach().cpu()),
        }
        return loss, logs


# =========================================================
# 6. Data loaders and evaluation
# =========================================================

def make_loaders(cfg: Config):
    train_ds = MultiZoneSynchronizedDataset(cfg, "train", cfg.max_train_samples)
    val_ds = MultiZoneSynchronizedDataset(cfg, "val", cfg.max_eval_samples)
    test_ds = MultiZoneSynchronizedDataset(cfg, "test", cfg.max_eval_samples)
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


def inverse_scale_batch(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x [B,Z,P], mean/scale [B,Z]
    return x * scale.unsqueeze(-1) + mean.unsqueeze(-1)


@torch.no_grad()
def evaluate_model(model: SGRPGDTSF, loader: DataLoader, cfg: Config, return_extra: bool = False):
    model.eval()
    device = torch.device(cfg.device)
    total_abs = 0.0
    total_sq = 0.0
    total_abs_pct = 0.0
    total_count = 0
    total_mape_count = 0
    # R2 sufficient statistics
    sum_y = 0.0
    sum_y2 = 0.0
    sum_res2 = 0.0
    # Base metrics for horizon reduction
    base_abs_h = torch.zeros(cfg.pred_len, dtype=torch.float64)
    pred_abs_h = torch.zeros(cfg.pred_len, dtype=torch.float64)
    h_count = 0

    extras = {
        "graph_adj": [],
        "y_true": [],
        "y_pred": [],
        "base_pred": [],
        "x_slow": [],
        "target_mean": [],
        "target_scale": [],
    } if return_extra else None

    for batch in loader:
        x_fast = batch["x_fast"].to(device, non_blocking=True)
        x_slow = batch["x_slow"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mean = batch["target_mean"].to(device, non_blocking=True)
        scale = batch["target_scale"].to(device, non_blocking=True)
        out = model(x_fast, x_slow)
        pred_u = inverse_scale_batch(out["pred"], mean, scale)
        base_u = inverse_scale_batch(out["base_pred"], mean, scale)
        y_u = inverse_scale_batch(y, mean, scale)
        err = pred_u - y_u
        base_err = base_u - y_u
        total_abs += float(err.abs().sum().detach().cpu())
        total_sq += float(err.pow(2).sum().detach().cpu())
        mask = y_u.abs() > 1e-8
        total_abs_pct += float(((err.abs() / y_u.abs().clamp_min(1e-8))[mask]).sum().detach().cpu())
        total_mape_count += int(mask.sum().detach().cpu())
        n = int(y_u.numel())
        total_count += n
        y_flat = y_u.reshape(-1)
        sum_y += float(y_flat.sum().detach().cpu())
        sum_y2 += float((y_flat ** 2).sum().detach().cpu())
        sum_res2 += float((err.reshape(-1) ** 2).sum().detach().cpu())
        pred_abs_h += err.abs().mean(dim=(0, 1)).detach().cpu().double()
        base_abs_h += base_err.abs().mean(dim=(0, 1)).detach().cpu().double()
        h_count += 1

        if return_extra:
            assert extras is not None
            extras["graph_adj"].append(out["graph_adj"].detach().cpu())
            extras["y_true"].append(y_u.detach().cpu())
            extras["y_pred"].append(pred_u.detach().cpu())
            extras["base_pred"].append(base_u.detach().cpu())
            extras["x_slow"].append(x_slow.detach().cpu())
            extras["target_mean"].append(mean.detach().cpu())
            extras["target_scale"].append(scale.detach().cpu())

    mae = total_abs / max(total_count, 1)
    rmse = math.sqrt(total_sq / max(total_count, 1))
    mape = 100.0 * total_abs_pct / max(total_mape_count, 1)
    y_mean = sum_y / max(total_count, 1)
    ss_tot = sum_y2 - total_count * (y_mean ** 2)
    r2 = 1.0 - sum_res2 / max(ss_tot, 1e-12)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    if h_count > 0:
        metrics["Base_Horizon_MAE_Mean"] = float((base_abs_h / h_count).mean().item())
        metrics["Pred_Horizon_MAE_Mean"] = float((pred_abs_h / h_count).mean().item())

    if return_extra:
        assert extras is not None
        for k in list(extras.keys()):
            extras[k] = torch.cat(extras[k], dim=0) if extras[k] else None
        return metrics, extras
    return metrics, None


# =========================================================
# 7. Training loop
# =========================================================

def build_model(method: str, cfg: Config) -> SGRPGDTSF:
    return SGRPGDTSF(cfg, method)


def checkpoint_path(cfg: Config, method: str, seed: int) -> str:
    return os.path.join(cfg.run_dir, f"best_{method}_seed{seed}.pt")


def result_exists(cfg: Config, method: str, seed: int) -> bool:
    path = os.path.join(cfg.run_dir, "running_results.csv")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return bool(((df["method"] == method) & (df["seed"] == seed)).any())
    except Exception:
        return False


def append_result(cfg: Config, row: Dict[str, Any]) -> None:
    ensure_dir(cfg.run_dir)
    path = os.path.join(cfg.run_dir, "running_results.csv")
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)


def train_one_model(method: str, seed: int, cfg: Config, loaders=None) -> Dict[str, Any]:
    ensure_dir(cfg.run_dir)
    if cfg.resume and result_exists(cfg, method, seed):
        print(f"[SKIP] result exists for method={method}, seed={seed}")
        rr = pd.read_csv(os.path.join(cfg.run_dir, "running_results.csv"))
        row = rr[(rr["method"] == method) & (rr["seed"] == seed)].iloc[-1].to_dict()
        return row

    set_seed(seed)
    device = torch.device(cfg.device)
    if loaders is None:
        train_loader, val_loader, test_loader = make_loaders(cfg)
    else:
        train_loader, val_loader, test_loader = loaders

    model = build_model(method, cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    best_rmse = float("inf")
    best_epoch = -1
    bad = 0
    best_path = checkpoint_path(cfg, method, seed)
    start = time.time()

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
                out = model(x_fast, x_slow, y_true=y)
                loss, logs = model.compute_loss(out, y, epoch)
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            ep_loss += float(loss.detach().cpu())
            ep_batches += 1

        val_metrics, _ = evaluate_model(model, val_loader, cfg, return_extra=False)
        epoch_time = time.time() - t0
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
                "cfg": asdict(cfg),
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, best_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[EARLY STOP] {method} seed={seed} at epoch={epoch}, best_epoch={best_epoch}")
                break

    # Test best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, _ = evaluate_model(model, test_loader, cfg, return_extra=False)
    elapsed = time.time() - start
    row = {
        "method": method,
        "method_description": method_description(method),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse,
        "train_time_sec": elapsed,
        **{f"Test_{k}": v for k, v in test_metrics.items()},
    }
    append_result(cfg, row)
    print(f"[TEST] {method} seed={seed}: {test_metrics}, elapsed={elapsed/60:.1f} min")
    return row


def summarize_results(cfg: Config) -> None:
    path = os.path.join(cfg.run_dir, "running_results.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df.to_csv(os.path.join(cfg.run_dir, "all_seed_results_detailed.csv"), index=False)
    metric_cols = [c for c in df.columns if c.startswith("Test_")]
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method, "n_seeds": len(g)}
        for c in metric_cols:
            row[c + "_mean"] = float(g[c].mean())
            row[c + "_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    # Sort by RMSE if available.
    if "Test_RMSE_mean" in out.columns:
        out = out.sort_values("Test_RMSE_mean")
    out.to_csv(os.path.join(cfg.run_dir, "all_seed_results_mean_std.csv"), index=False)

    # Paper-friendly table.
    paper_rows = []
    for _, r in out.iterrows():
        m = r["method"]
        paper_rows.append({
            "Method": m,
            "MAE": f"{r.get('Test_MAE_mean', np.nan):.2f} ± {r.get('Test_MAE_std', 0):.2f}",
            "RMSE": f"{r.get('Test_RMSE_mean', np.nan):.2f} ± {r.get('Test_RMSE_std', 0):.2f}",
            "MAPE(%)": f"{r.get('Test_MAPE_mean', np.nan):.2f} ± {r.get('Test_MAPE_std', 0):.2f}",
            "R2": f"{r.get('Test_R2_mean', np.nan):.4f} ± {r.get('Test_R2_std', 0):.4f}",
        })
    pd.DataFrame(paper_rows).to_csv(os.path.join(cfg.run_dir, "paper_overall_table_mean_std.csv"), index=False)


def experiment_methods(group: str) -> List[str]:
    g = group.lower()
    if g in ["all_clear_ablation", "sgr_clear_ablation"]:
        return [
            "sgr_pg_dtsf_full",
            "base_dual_timescale_only",
            "graph_local_residual_only",
            "graph_static_learned_prior",
            "input_no_slow_variables",
            "input_no_fast_load_history",
            "input_no_zone_identity",
            "cond_no_slow_state",
            "cond_no_fast_state",
            "cond_no_zone_state",
            "cond_no_base_forecast_morphology",
        ]
    if g == "input_ablation":
        return [
            "sgr_pg_dtsf_full",
            "input_no_slow_variables",
            "input_no_fast_load_history",
            "input_no_zone_identity",
        ]
    if g == "residual_conditioning_ablation":
        return [
            "sgr_pg_dtsf_full",
            "cond_no_slow_state",
            "cond_no_fast_state",
            "cond_no_zone_state",
            "cond_no_base_forecast_morphology",
        ]
    if g == "graph_structure_ablation":
        return [
            "sgr_pg_dtsf_full",
            "graph_local_residual_only",
            "graph_static_learned_prior",
            "base_dual_timescale_only",
        ]
    if g == "paper_core_ablation":
        return [
            "sgr_pg_dtsf_full",
            "base_dual_timescale_only",
            "graph_local_residual_only",
            "graph_static_learned_prior",
            "input_no_slow_variables",
            "input_no_fast_load_history",
            "input_no_zone_identity",
        ]
    if g in ["sgr_ablation", "sgr_diagnostic"]:
        raise ValueError(
            f"Legacy experiment_group={group!r} is disabled to avoid ambiguous method names. "
            "Use all_clear_ablation, paper_core_ablation, input_ablation, "
            "residual_conditioning_ablation, or graph_structure_ablation."
        )
    raise ValueError(f"Unknown experiment_group={group}")


def run_experiments(cfg: Config, methods: List[str]) -> None:
    ensure_dir(cfg.run_dir)
    with open(os.path.join(cfg.run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.run_dir, "method_definitions.json"), "w", encoding="utf-8") as f:
        json.dump({m: {"description": method_description(m), "flags": sgr_flags_from_name(m)} for m in methods}, f, indent=2, ensure_ascii=False)
    check_processed_data(cfg)
    loaders = make_loaders(cfg)
    for method in methods:
        for seed in cfg.seeds:
            train_one_model(method, seed, cfg, loaders=loaders)
            summarize_results(cfg)


# =========================================================
# 8. SGR interpretability analysis
# =========================================================

def run_sgr_analysis(cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = cfg.sgr_analysis_method
    seed = int(cfg.sgr_analysis_seed)
    device = torch.device(cfg.device)
    analysis_dir = os.path.join(cfg.run_dir, f"sgr_interpretability_{method}_seed{seed}")
    ensure_dir(analysis_dir)

    # Use a bounded subset for analysis.
    old_max = cfg.max_eval_samples
    cfg.max_eval_samples = int(cfg.sgr_analysis_max_samples)
    _, _, test_loader = make_loaders(cfg)
    cfg.max_eval_samples = old_max

    model = build_model(method, cfg).to(device)
    ckpt_path = checkpoint_path(cfg, method, seed)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for analysis: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics, extra = evaluate_model(model, test_loader, cfg, return_extra=True)
    with open(os.path.join(analysis_dir, "analysis_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    assert extra is not None
    A = extra["graph_adj"].numpy()  # [N,Z,Z]
    y_true = extra["y_true"].numpy()
    y_pred = extra["y_pred"].numpy()
    y_base = extra["base_pred"].numpy()
    x_slow = extra["x_slow"].numpy()

    # Overall graph
    A_mean = A.mean(axis=0)
    pd.DataFrame(A_mean, index=VALID_ZONES, columns=VALID_ZONES).to_csv(os.path.join(analysis_dir, "sgr_mean_dynamic_graph.csv"))
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(A_mean, aspect="auto")
    plt.xticks(range(len(VALID_ZONES)), VALID_ZONES, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(VALID_ZONES)), VALID_ZONES, fontsize=8)
    plt.colorbar(label="Mean graph weight")
    plt.title("Mean Dynamic Semantic Graph")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "fig_mean_dynamic_graph.png"), dpi=220)
    plt.close()

    # Graph hub scores
    hub = pd.DataFrame({
        "zone": VALID_ZONES,
        "outgoing_mean": A_mean.mean(axis=1),
        "incoming_mean": A_mean.mean(axis=0),
        "self_weight": np.diag(A_mean),
    })
    hub.to_csv(os.path.join(analysis_dir, "sgr_zone_graph_hub_scores.csv"), index=False)

    # Context masks.  x_slow is scaled; binary features are >0 after standardization in typical cases.
    max_temp_idx = SLOW_FEATURES.index("max_temp")
    weekend_idx = SLOW_FEATURES.index("is_weekend")
    holiday_idx = SLOW_FEATURES.index("is_us_holiday")
    sample_temp = x_slow[:, :, max_temp_idx].mean(axis=1)
    high_temp = sample_temp >= np.quantile(sample_temp, 0.80)
    weekend = x_slow[:, :, weekend_idx].mean(axis=1) > 0.0
    holiday = x_slow[:, :, holiday_idx].mean(axis=1) > 0.0
    peak_score = y_true.max(axis=(1, 2))
    peak_day = peak_score >= np.quantile(peak_score, 0.80)
    contexts = {
        "all": np.ones(A.shape[0], dtype=bool),
        "high_temp": high_temp,
        "weekend": weekend,
        "holiday": holiday,
        "peak_day": peak_day,
    }

    ctx_rows = []
    for name, mask in contexts.items():
        if int(mask.sum()) == 0:
            continue
        A_ctx = A[mask].mean(axis=0)
        pd.DataFrame(A_ctx, index=VALID_ZONES, columns=VALID_ZONES).to_csv(os.path.join(analysis_dir, f"sgr_graph_{name}.csv"))
        ctx_rows.append({
            "context": name,
            "n_samples": int(mask.sum()),
            "mean_self_weight": float(np.trace(A_ctx) / len(VALID_ZONES)),
            "mean_offdiag_weight": float((A_ctx.sum() - np.trace(A_ctx)) / (len(VALID_ZONES) * (len(VALID_ZONES) - 1))),
            "max_incoming_zone": VALID_ZONES[int(A_ctx.mean(axis=0).argmax())],
            "max_outgoing_zone": VALID_ZONES[int(A_ctx.mean(axis=1).argmax())],
        })
    pd.DataFrame(ctx_rows).to_csv(os.path.join(analysis_dir, "sgr_context_graph_summary.csv"), index=False)

    # Horizon-wise error reduction vs base
    base_abs_h = np.abs(y_base - y_true).mean(axis=(0, 1))
    pred_abs_h = np.abs(y_pred - y_true).mean(axis=(0, 1))
    red_pct = 100.0 * (base_abs_h - pred_abs_h) / np.maximum(base_abs_h, 1e-8)
    hdf = pd.DataFrame({
        "horizon_step": np.arange(1, cfg.pred_len + 1),
        "base_mae": base_abs_h,
        "sgr_mae": pred_abs_h,
        "mae_reduction_pct": red_pct,
    })
    hdf.to_csv(os.path.join(analysis_dir, "sgr_horizon_error_reduction.csv"), index=False)
    plt.figure(figsize=(8.5, 3.8))
    plt.plot(hdf["horizon_step"], hdf["base_mae"], label="Base")
    plt.plot(hdf["horizon_step"], hdf["sgr_mae"], label="SGR")
    plt.xlabel("Forecast horizon step")
    plt.ylabel("MAE")
    plt.title("Horizon-wise Error Reduction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "fig_horizon_mae_reduction.png"), dpi=220)
    plt.close()

    # Peak / valley / ramp step improvements.  Define steps by average true trajectory over samples and zones.
    avg_curve = y_true.mean(axis=(0, 1))
    peak_thr = np.quantile(avg_curve, 0.80)
    valley_thr = np.quantile(avg_curve, 0.20)
    ramps = np.abs(np.diff(avg_curve, prepend=avg_curve[0]))
    ramp_thr = np.quantile(ramps, 0.80)
    masks = {
        "all": np.ones(cfg.pred_len, dtype=bool),
        "peak_steps": avg_curve >= peak_thr,
        "valley_steps": avg_curve <= valley_thr,
        "ramp_steps": ramps >= ramp_thr,
    }
    rows = []
    abs_base = np.abs(y_base - y_true)
    abs_pred = np.abs(y_pred - y_true)
    for name, hm in masks.items():
        rows.append({
            "segment": name,
            "n_steps": int(hm.sum()),
            "base_mae": float(abs_base[:, :, hm].mean()),
            "sgr_mae": float(abs_pred[:, :, hm].mean()),
            "reduction_pct": float(100.0 * (abs_base[:, :, hm].mean() - abs_pred[:, :, hm].mean()) / max(abs_base[:, :, hm].mean(), 1e-8)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(analysis_dir, "sgr_peak_valley_ramp_error_reduction.csv"), index=False)

    # Zone-wise error reduction
    zrows = []
    for z, zone in enumerate(VALID_ZONES):
        b = float(abs_base[:, z, :].mean())
        p = float(abs_pred[:, z, :].mean())
        zrows.append({"zone": zone, "base_mae": b, "sgr_mae": p, "reduction_pct": 100.0 * (b - p) / max(b, 1e-8)})
    pd.DataFrame(zrows).to_csv(os.path.join(analysis_dir, "sgr_zone_error_reduction.csv"), index=False)

    print(f"[SGR ANALYSIS] Saved to {analysis_dir}")


# =========================================================
# 9. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="SGR-PG-DTSF multi-zone graph residual experiments")
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--experiment_group", type=str, default="")
    parser.add_argument("--experiment_tag", type=str, default="sgr_pg_dtsf")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="2026,2027,2028,2029,2030")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--sample_mode", type=str, default="uniform", choices=["first", "uniform", "random"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_save_predictions", action="store_true")
    parser.add_argument("--skip_experiments", action="store_true")

    # SGR parameters
    parser.add_argument("--sgr_graph_layers", type=int, default=None)
    parser.add_argument("--sgr_graph_dim", type=int, default=None)
    parser.add_argument("--sgr_graph_alpha", type=float, default=None)
    parser.add_argument("--sgr_temperature", type=float, default=None)
    parser.add_argument("--sgr_self_loop_bias", type=float, default=None)
    parser.add_argument("--sgr_residual_scale", type=float, default=None)
    parser.add_argument("--sgr_gate_init_bias", type=float, default=None)
    parser.add_argument("--sgr_graph_sparsity_weight", type=float, default=None)
    parser.add_argument("--sgr_graph_entropy_weight", type=float, default=None)
    parser.add_argument("--base_loss_weight", type=float, default=None)
    parser.add_argument("--residual_loss_weight", type=float, default=None)
    parser.add_argument("--correction_l2_weight", type=float, default=None)

    # analysis
    parser.add_argument("--run_sgr_analysis", action="store_true")
    parser.add_argument("--sgr_analysis_method", type=str, default="sgr_pg_dtsf_full")
    parser.add_argument("--sgr_analysis_seed", type=int, default=2026)
    parser.add_argument("--sgr_analysis_max_samples", type=int, default=30000)

    args = parser.parse_args()
    cfg = Config()
    if args.save_root is not None:
        cfg.save_root = args.save_root
    cfg.run_experiments = not bool(args.skip_experiments)
    cfg.experiment_tag = args.experiment_tag
    cfg.seeds = parse_seeds(args.seeds)
    cfg.max_train_samples = int(args.max_train_samples)
    cfg.max_eval_samples = int(args.max_eval_samples)
    cfg.sample_mode = args.sample_mode
    cfg.resume = bool(args.resume)
    cfg.use_amp = bool(args.amp)
    cfg.save_predictions = not bool(args.no_save_predictions)
    if args.device is not None:
        cfg.device = args.device if args.device.startswith("cuda") or args.device == "cpu" else f"cuda:{args.device}"
    for name in [
        "epochs", "patience", "batch_size", "lr", "weight_decay", "hidden_dim", "num_workers",
        "sgr_graph_layers", "sgr_graph_dim", "sgr_graph_alpha", "sgr_temperature", "sgr_self_loop_bias",
        "sgr_residual_scale", "sgr_gate_init_bias", "sgr_graph_sparsity_weight", "sgr_graph_entropy_weight",
        "base_loss_weight", "residual_loss_weight", "correction_l2_weight"
    ]:
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    cfg.run_sgr_analysis = bool(args.run_sgr_analysis)
    cfg.sgr_analysis_method = args.sgr_analysis_method
    cfg.sgr_analysis_seed = int(args.sgr_analysis_seed)
    cfg.sgr_analysis_max_samples = int(args.sgr_analysis_max_samples)

    check_processed_data(cfg)
    ensure_dir(cfg.run_dir)

    if args.methods.strip():
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    elif args.experiment_group.strip():
        methods = experiment_methods(args.experiment_group)
    else:
        methods = ["sgr_pg_dtsf_full"]

    print("=" * 80)
    print("SGR-PG-DTSF experiment")
    print(f"save_dir={cfg.save_dir}")
    print(f"run_dir={cfg.run_dir}")
    print(f"methods={methods}")
    print(f"seeds={cfg.seeds}")
    print(f"device={cfg.device}, amp={cfg.use_amp}")
    print("=" * 80)

    if cfg.run_experiments:
        run_experiments(cfg, methods)
    if cfg.run_sgr_analysis:
        run_sgr_analysis(cfg)


if __name__ == "__main__":
    main()
