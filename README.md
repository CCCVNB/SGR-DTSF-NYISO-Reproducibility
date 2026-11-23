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


## 2. Environment

The scripts use Python, NumPy, pandas, Matplotlib, and PyTorch.

```bash
python -m pip install -r requirements.txt
```

