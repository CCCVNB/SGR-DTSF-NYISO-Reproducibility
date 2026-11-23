# SGR-DTSF Reproducibility — NYISO Release

**Release:** v0.2.0  
**Scope:** NYISO H288→P288 only  

This repository provides the NYISO implementation of **Semantic Graph Residual Dual-Timescale Forecasting (SGR-DTSF)** and the author-implemented baseline models used in the NYISO experiments reported in the manuscript.


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

