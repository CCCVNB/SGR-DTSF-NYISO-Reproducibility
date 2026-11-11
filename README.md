# SGR-DTSF Reproducibility — NYISO Release

**Release:** v0.2.0  
**Scope:** NYISO H288→P288 only  
**Status:** Selective source-code release candidate

This repository provides the NYISO implementation of **Semantic Graph Residual Dual-Timescale Forecasting (SGR-DTSF)** and the author-implemented baseline models used in the NYISO experiments reported in the manuscript.

> **Scope boundary:** This is an NYISO-only code release. It does not contain GEFCom2012 source code, raw datasets, data preprocessing, feature engineering, scaler fitting, chronological splitting, or dataset-generation programs. It also does not include processed arrays, fitted scalers, trained checkpoints, optimizer states, prediction dumps, or internal experiment-audit utilities.

## 1. Included NYISO methods

The NYISO H288→P288 comparison includes the following methods:

| Group | Method |
|---|---|
| Proposed method | SGR-DTSF |
| Persistence / linear | Repeat-last-day, NLinear, DLinear |
| Fusion | Direct Fusion, Residual Fusion, Cross-attention Fusion |
| MLP / Transformer | TiDE-MLP, STID-MLP, PatchTST-Exog, iTransformer-Exog |
| Temporal / graph | TimesNet-Lite-Exog, Graph WaveNet-Lite, MTGNN-Lite |

The baseline implementations are **author-implemented adaptations for the NYISO multi-zone forecasting protocol**. They are not represented as verbatim copies of the original authors' official repositories. The `-Lite`, `-Exog`, and `-MLP` suffixes describe the compact or input-adapted variants used in this study.

The fusion runner contains additional historical experimental variants. Only `direct_fusion`, `residual_fusion`, and `cross_attention_fusion` correspond to the three fusion methods listed above. Other method keys in that source file should not be interpreted as additional manuscript comparison results.

## 2. Repository layout

```text
SGR-DTSF-NYISO-Reproducibility/
├── src/
│   └── nyiso/
│       ├── two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py
│       ├── two_half_year_strong_core_baseline_experiments.py
│       └── two_half_year_fused_pg_dtsf_experiments.py
├── docs/
├── requirements.txt
├── CITATION.cff
├── SOURCE_PROVENANCE_SHA256.csv
├── RELEASE_MANIFEST.json
└── SHA256SUMS.txt
```

## 3. Environment

The scripts use Python, NumPy, pandas, Matplotlib, and PyTorch. `requirements.txt` is a minimal dependency list, not a lockfile and not a claim that it exactly recreates every historical environment. Install a PyTorch build compatible with the local CPU/GPU and CUDA runtime.

```bash
python -m pip install -r requirements.txt
```

## 4. NYISO input data contract

Raw NYISO load and weather data are not included. No raw-data preprocessing, feature construction, scaler fitting, chronological splitting, or dataset-generation code is provided.

Supply already-prepared per-zone arrays under:

```text
<save_root>/two_half_year_201601_201807_per_zone_scalers_H288_P288/
├── train_by_zone/
│   ├── train_Xf_<zone>.npy
│   ├── train_Xs_<zone>.npy
│   └── train_Y_<zone>.npy
├── val_by_zone/
│   ├── val_Xf_<zone>.npy
│   ├── val_Xs_<zone>.npy
│   └── val_Y_<zone>.npy
├── test_by_zone/
│   ├── test_Xf_<zone>.npy
│   ├── test_Xs_<zone>.npy
│   └── test_Y_<zone>.npy
└── scalers_by_zone/
    └── <zone>/
        ├── target_mean.npy
        └── target_scale.npy
```

The 11 NYISO zones are:

`CAPITL`, `CENTRL`, `DUNWOD`, `GENESE`, `HUD VL`, `LONGIL`, `MHK VL`, `MILLWD`, `N.Y.C.`, `NORTH`, and `WEST`.

The model protocol uses five-minute sampling and H288→P288. Per-zone input arrays must be aligned on the same sample/time index across all 11 zones. The model-side dataset adapters load these preprocessed arrays and target-scaler statistics; they do not create or transform the dataset.

Expected per-zone array layouts are:

- `Xf`: `[N, 288, 3]`
- `Xs`: `[N, 13]`
- `Y`: `[N, 288]`

Optional timestamp arrays may be supplied using the matching `*_T_<zone>.npy` naming convention if available. The `target_mean.npy` and `target_scale.npy` files are required for inverse scaling.

See [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md) for the concise input contract.

## 5. NYISO experiment commands

The following commands are execution examples. They do not imply that training was performed during assembly of this public release. Check each script's `--help` output and verify that the required preprocessed arrays are present before launching a run.

### 5.1 SGR-DTSF and proposed-model ablations

```bash
python src/nyiso/two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py \
  --save_root ./data/processed \
  --experiment_group paper_core_ablation \
  --experiment_tag sgr_dtsf_nyiso \
  --device cuda \
  --seeds 2026,2027,2028,2029,2030 \
  --epochs 30 --patience 6 --batch_size 64 \
  --lr 0.0005 --weight_decay 0.00001 --hidden_dim 128 \
  --sgr_graph_layers 2 --amp --no_save_predictions
```

The proposed-model entrypoint also exposes the ablation and diagnostic options shown by:

```bash
python src/nyiso/two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py --help
```

### 5.2 NYISO persistence, linear, MLP, Transformer, temporal, and graph baselines

```bash
python src/nyiso/two_half_year_strong_core_baseline_experiments.py \
  --base_script src/nyiso/two_half_year_sgr_pg_dtsf_experiments_clear_ablation.py \
  --save_root ./data/processed \
  --methods repeat_last_day,nlinear,dlinear,tide_mlp,stid_mlp,patchtst_ci_exog,itransformer_exog,timesnet_lite_exog,graph_wavenet_lite,mtgnn_lite \
  --experiment_tag nyiso_paper_strong_baselines \
  --device cuda \
  --seeds 2026,2027,2028,2029,2030 \
  --epochs 30 --patience 6 --batch_size 64 \
  --lr 0.0005 --weight_decay 0.00001 --hidden_dim 128 --amp
```

### 5.3 NYISO fusion baselines

```bash
python src/nyiso/two_half_year_fused_pg_dtsf_experiments.py \
  --save_root ./data/processed \
  --methods direct_fusion,residual_fusion,cross_attention_fusion \
  --experiment_tag nyiso_paper_fusion_baselines \
  --device cuda \
  --seeds 2026,2027,2028,2029,2030 \
  --epochs 30 --batch_size 64 --patience 6 \
  --lr 0.0005 --weight_decay 0.00001 --hidden_dim 128 --amp \
  --no_save_predictions
```

Training may consume substantial GPU time. Use the documented seed set and experimental settings when comparing with the manuscript, and retain the run configuration and logs for provenance.

## 6. Baseline implementation mapping

See [`docs/BASELINE_IMPLEMENTATION_FIDELITY.md`](docs/BASELINE_IMPLEMENTATION_FIDELITY.md) for the NYISO-only mapping between manuscript methods, source files, class names/method keys, and implementation summaries.

## 7. Excluded materials

This selective NYISO release intentionally excludes:

- All GEFCom2012 source code and experiments.
- Raw NYISO load/weather data and any third-party data files.
- Processed `.npy` arrays and fitted scaler statistics.
- Raw-data preprocessing, feature engineering, scaler-fitting, time-splitting, and dataset-generation programs.
- Trained model checkpoints, optimizer states, prediction dumps, training logs, and large intermediate outputs.
- Server-specific launch scripts, credentials, local machine paths, and internal audit utilities.

## 8. Reproducibility scope and limitations

This repository makes the NYISO SGR-DTSF implementation and the NYISO baseline implementations available for inspection and fresh experiments when the required preprocessed inputs are supplied. It is not a self-contained dataset release and is not a replayable snapshot of every historical experiment. It does not claim that the adapted baselines are exact official implementations of the cited methods.

The released source files were adapted from the corresponding project source files. Changes made specifically for public release, including removal of raw-data preprocessing logic and replacement of local machine paths, are recorded in `SOURCE_PROVENANCE_SHA256.csv`.

## 9. License status

No open-source license has been selected in this candidate package. A public GitHub repository can be made viewable without selecting a license, but reuse, modification, and redistribution permissions remain unspecified until the authors select and add a license. Before public release, verify that all included source files are owned by the authors or are otherwise permitted for redistribution.
