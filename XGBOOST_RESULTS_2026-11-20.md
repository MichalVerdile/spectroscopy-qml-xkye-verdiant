# XGBoost raw-native results, 20 November 2026

All 25 runs completed successfully. F1 values use a scale from 0 to 1.

| Modality | Seed | Train micro | Train macro | Val micro | Val macro | Test micro | Test macro | Fit h | Wall h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ¹³C-NMR | 41 | 0.9182 | 0.6885 | 0.8488 | 0.5108 | 0.8494 | 0.5117 | 2.86 | 0.28 |
| ¹³C-NMR | 42 | 0.9219 | 0.6758 | 0.8499 | 0.5088 | 0.8500 | 0.5062 | 2.87 | 3.23 |
| ¹³C-NMR | 43 | 0.9204 | 0.6838 | 0.8505 | 0.5168 | 0.8501 | 0.5033 | 2.84 | 3.38 |
| ¹³C-NMR | 44 | 0.9221 | 0.6922 | 0.8513 | 0.5176 | 0.8497 | 0.5195 | 3.00 | 3.37 |
| ¹³C-NMR | 45 | 0.9205 | 0.6795 | 0.8490 | 0.5058 | 0.8507 | 0.5139 | 2.95 | 3.29 |
| ¹H-NMR | 41 | 0.9787 | 0.7725 | 0.8901 | 0.5597 | 0.8893 | 0.5612 | 5.34 | 6.66 |
| ¹H-NMR | 42 | 0.9766 | 0.7638 | 0.8893 | 0.5591 | 0.8882 | 0.5538 | 5.30 | 6.30 |
| ¹H-NMR | 43 | 0.9767 | 0.7589 | 0.8890 | 0.5614 | 0.8890 | 0.5500 | 5.21 | 6.19 |
| ¹H-NMR | 44 | 0.9781 | 0.7775 | 0.8894 | 0.5579 | 0.8888 | 0.5605 | 5.43 | 6.44 |
| ¹H-NMR | 45 | 0.9764 | 0.7633 | 0.8890 | 0.5572 | 0.8895 | 0.5606 | 5.23 | 6.20 |
| IR | 41 | 0.9636 | 0.8383 | 0.8639 | 0.5586 | 0.8640 | 0.5641 | 0.71 | 0.99 |
| IR | 42 | 0.9665 | 0.8196 | 0.8632 | 0.5586 | 0.8637 | 0.5570 | 0.72 | 0.79 |
| IR | 43 | 0.9636 | 0.8061 | 0.8635 | 0.5603 | 0.8635 | 0.5602 | 0.73 | 0.79 |
| IR | 44 | 0.9598 | 0.8081 | 0.8634 | 0.5540 | 0.8629 | 0.5594 | 0.72 | 0.77 |
| IR | 45 | 0.9662 | 0.8079 | 0.8639 | 0.5562 | 0.8640 | 0.5586 | 0.73 | 0.80 |
| MS/MS+ | 41 | 0.8447 | 0.4911 | 0.7689 | 0.3326 | 0.7687 | 0.3291 | 2.80 | 3.42 |
| MS/MS+ | 42 | 0.8467 | 0.4807 | 0.7694 | 0.3280 | 0.7694 | 0.3314 | 2.89 | 3.37 |
| MS/MS+ | 43 | 0.8508 | 0.4902 | 0.7703 | 0.3384 | 0.7703 | 0.3306 | 2.97 | 3.44 |
| MS/MS+ | 44 | 0.8392 | 0.4728 | 0.7674 | 0.3297 | 0.7677 | 0.3351 | 2.67 | 3.06 |
| MS/MS+ | 45 | 0.8438 | 0.4737 | 0.7681 | 0.3295 | 0.7686 | 0.3345 | 2.66 | 3.07 |
| MS/MS− | 41 | 0.8343 | 0.4893 | 0.7891 | 0.3905 | 0.7887 | 0.3886 | 2.22 | 2.66 |
| MS/MS− | 42 | 0.8380 | 0.4831 | 0.7901 | 0.3870 | 0.7892 | 0.3823 | 2.27 | 2.54 |
| MS/MS− | 43 | 0.8364 | 0.4908 | 0.7888 | 0.3881 | 0.7885 | 0.3959 | 2.27 | 2.54 |
| MS/MS− | 44 | 0.8459 | 0.4913 | 0.7917 | 0.3863 | 0.7923 | 0.3861 | 2.54 | 2.92 |
| MS/MS− | 45 | 0.8424 | 0.4805 | 0.7908 | 0.3891 | 0.7914 | 0.3883 | 2.40 | 2.77 |

The archived CNMR seed 41 wall time covers recovery, while training_seconds records the retained model fits. These timings cover different scopes.

## Five-seed test summaries

| Modality | Test micro mean ± sample SD | Test macro mean ± sample SD |
|---|---:|---:|
| ¹³C-NMR | 0.8500 ± 0.0005 | 0.5109 ± 0.0064 |
| ¹H-NMR | 0.8890 ± 0.0005 | 0.5572 ± 0.0050 |
| IR | 0.8636 ± 0.0005 | 0.5599 ± 0.0027 |
| MS/MS+ | 0.7690 ± 0.0010 | 0.3321 ± 0.0026 |
| MS/MS− | 0.7900 ± 0.0018 | 0.3882 ± 0.0050 |

## Files and storage

The runpod directory contains the source used for the experiments. The six tar.gz archives contain all 1,233 exported experiment, configuration, diagnostic and log files, with their repository-relative paths preserved. Extract each archive at the repository root using tar -xzf ARCHIVE.tar.gz. The CSV retains the full original summary columns, including split hashes and per-label best iterations.

Models, UBJ checkpoints and binary prediction results are retained on the RunPod global volume under /workspace/experiments/xgboost_raw_native. Their paths and sizes are listed in persistent_artifacts.json inside the support archive. The pod was stopped after the exports were secured.

The complete RunPod artifact archive was added on 24 November 2026. Models, checkpoints, predictions, generated splits and caches are available under [results/xgboost_raw_native_2026-11-20](results/xgboost_raw_native_2026-11-20). See the [LFS archive instructions](results/xgboost_raw_native_2026-11-20/LFS_ARCHIVE_README.md) for downloads and reconstruction of the two large cache files.
