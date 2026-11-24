# Raw-native XGBoost learning-curve source

This directory contains only the small artifacts required to generate the
raw-native XGBoost learning curves and validate their final scores:

- `training_logs.csv` for train/validation Micro-F1 and Macro-F1 over boosting steps;
- `metrics.json` for seed, modality, input length, preprocessing and final-score validation.

The files were selectively imported from remote branch
`results/xgboost-raw-native-2026-11-20` at commit
`a595ef08a4e98f29c3bd0165b6ffcfd0cff87d8f`.

The original model pickles, prediction pickles, per-label checkpoints, caches,
generated splits, console logs and recovery metadata are intentionally excluded
to keep these sources isolated from the existing raw benchmark results.

The imported run covers all five modalities and seeds 41--45. NMR and MS/MS use
10,000 input points; IR uses 1,800 input points. The spectra use the raw/native
representation without smoothing, nonlinear transformation or normalization.

These external-GPU logs are the source for learning-curve shape only. Their
recorded fit and evaluation times must not be mixed with the Sanjose timing run.
Hardware-comparable XGBoost times are stored separately under
`raw_native_xgboost_timing_sanjose` when Slurm job `172550` completes.
