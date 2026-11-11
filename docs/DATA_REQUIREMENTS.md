# NYISO model input data contract

This repository intentionally contains no raw-data preprocessing, feature engineering, scaler-fitting, chronological splitting, or dataset-generation code. The training and evaluation entrypoints include only the minimal dataset-loading and scaler-reading interfaces needed to consume user-supplied preprocessed inputs.

## Dataset root

Place already-prepared inputs under:

```text
<save_root>/two_half_year_201601_201807_per_zone_scalers_H288_P288/
├── train_by_zone/
├── val_by_zone/
├── test_by_zone/
└── scalers_by_zone/
```

For each of the 11 zones and each split, the dataset adapter expects files using the following naming convention:

- `<split>_Xf_<zone>.npy`
- `<split>_Xs_<zone>.npy`
- `<split>_Y_<zone>.npy`
- optionally `<split>_T_<zone>.npy`

The scaler directory contains:

- `scalers_by_zone/<zone>/target_mean.npy`
- `scalers_by_zone/<zone>/target_scale.npy`

The zone-safe filenames replace spaces and periods with underscores, consistent with the loader's `zone_safe_name` function.

## Zones and expected array layouts

The 11 zones are:

`CAPITL`, `CENTRL`, `DUNWOD`, `GENESE`, `HUD VL`, `LONGIL`, `MHK VL`, `MILLWD`, `N.Y.C.`, `NORTH`, and `WEST`.

For the H288→P288 protocol, the expected per-zone arrays are:

- `Xf`: `[N, 288, 3]`
- `Xs`: `[N, 13]`
- `Y`: `[N, 288]`

The 11 zones must be synchronized and aligned on the same sample/time index. Target means and scales must be supplied for inverse scaling. The model-side loaders do not derive, fit, or alter these statistics.

This repository does not provide the data pipeline used to generate these arrays. Users must supply arrays that match the feature dimensions, feature order, zone mapping, temporal alignment, and normalization protocol documented in the manuscript and expected by the source loaders.
