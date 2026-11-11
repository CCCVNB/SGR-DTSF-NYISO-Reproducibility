# NYISO baseline implementation fidelity and manuscript mapping

This document maps the baseline rows in the NYISO H288→P288 experiments to the author implementations included in this selective source release. These implementations are task-adapted for the synchronized 11-zone forecasting protocol. They are not claimed to be verbatim copies of the cited methods' official repositories.

| Manuscript method | Source file | Implementation class / method key | Implementation summary |
|---|---|---|---|
| Repeat-last-day | `src/nyiso/two_half_year_strong_core_baseline_experiments.py` | `RepeatLastDay` / `repeat_last_day` | Repeats the last daily load trajectory from the historical input window. |
| NLinear | same | `NLinear` / `nlinear` | Linear projection over load history after subtracting the last observation, then adds the last observation back. |
| DLinear | same | `DLinear` / `dlinear` | Moving-average trend/seasonal decomposition with separate linear projections; the load-history path is used. |
| Direct Fusion | `src/nyiso/two_half_year_fused_pg_dtsf_experiments.py` | `DirectFusion` / `direct_fusion` | Fast GRU and slow MLP embeddings are concatenated with zone identity and passed to a prediction head. |
| Residual Fusion | same | `ResidualFusion` / `residual_fusion` | Fast-GRU base forecast plus a gated residual correction using fast, slow, and zone features. |
| Cross-attention Fusion | same | `CrossAttentionFusion` / `cross_attention_fusion` | The fast GRU summary queries a projected slow-context token through multi-head attention; this is a compact one-token adaptation. |
| TiDE-MLP | `src/nyiso/two_half_year_strong_core_baseline_experiments.py` | `TiDEMLP` / `tide_mlp` | TiDE-style MLP encoder/decoder adaptation with slow context and zone identity. |
| STID-MLP | same | `STIDMLP` / `stid_mlp` | Identity-aware MLP using zone identity, slow context, and handcrafted summaries of load/time history. |
| PatchTST-Exog | same | `PatchTSTCIExog` / `patchtst_ci_exog` | Channel-independent patch encoder with slow context and zone identity in the prediction head; a compact adaptation. |
| iTransformer-Exog | same | `ITransformerExog` / `itransformer_exog` | Inverted Transformer-style zone representation with slow context and zone identity. |
| TimesNet-Lite-Exog | same | `TimesNetLiteExog` / `timesnet_lite_exog` | Lightweight multi-kernel temporal convolution branches with slow context and zone identity. |
| Graph WaveNet-Lite | same | `GraphWaveNetLite` / `graph_wavenet_lite` | Compact dilated temporal convolutions and adaptive graph propagation with slow context and zone identity. |
| MTGNN-Lite | same | `MTGNNLite` / `mtgnn_lite` | Compact temporal-kernel and adaptive graph modules with slow context and zone identity. |

The historical fusion runner contains additional experimental variants, including gated and shared-bottom variants, that are not rows in the final NYISO main-comparison table. Their presence in the source file must not be interpreted as evidence that they were all reported in the manuscript.

## Proposed-model ablations

The NYISO SGR-DTSF script includes the proposed-model ablation variants used in the study, including:

- `sgr_pg_dtsf_full`
- `base_dual_timescale_only`
- `graph_local_residual_only`
- `graph_static_learned_prior`
- Input ablations involving slow variables, fast load history, and zone identity.
- Residual-conditioning ablations involving slow state, fast state, zone state, and base-forecast morphology.

These are proposed-model ablations, not external baseline architectures.

## Fidelity statement

The repository provides the author implementations used for the NYISO comparisons, with the adaptations described above. It does not claim exact reproduction of official third-party repositories or every implementation/training detail in the cited papers. The `Lite`, `Exog`, and `MLP` suffixes denote the compact/task-adapted implementations used in this work.
