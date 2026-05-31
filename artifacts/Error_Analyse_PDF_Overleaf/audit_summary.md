# Error-Analysis Audit

The figures in `artifacts/Error_Analyse` were regenerated from the existing CSV and JSON metrics files.
This corrects mislabeled figure titles and recreates matching PDF exports for Overleaf.

| Dataset | Split | F1 Macro | Mean Error | Median Error |
| --- | --- | ---: | ---: | ---: |
| CNN_IR | test | 0.6844 | 35.2952% | 26.7481% |
| CNN_IR | validation | 0.6778 | 24.5181% | 15.5660% |
| CNN_H-NMR | test | 0.0435 | 59.7268% | 60.3163% |
| CNN_H-NMR | validation | 0.0436 | 60.1577% | 61.9565% |
| CNN_C-NMR | test | 0.1069 | 81.3032% | 99.2548% |
| CNN_C-NMR | validation | 0.1070 | 78.5919% | 99.2076% |
| CNN_MSMS+ | test | 0.1142 | 88.8344% | 99.2090% |
| CNN_MSMS+ | validation | 0.1136 | 88.8653% | 98.9590% |
| CNN_MSMS- | test | 0.1227 | 88.2394% | 99.2500% |
| CNN_MSMS- | validation | 0.1228 | 83.5708% | 96.4299% |
| MPS_IR | test | 0.6712 | 36.1261% | 29.4264% |
| MPS_IR | validation | 0.6621 | 34.2184% | 28.9940% |
| MPS_H-NMR | test | 0.2563 | 60.3206% | 71.8750% |
| MPS_H-NMR | validation | 0.2576 | 56.7710% | 61.7410% |
| MPS_C-NMR | test | 0.3306 | 58.0859% | 63.6913% |
| MPS_C-NMR | validation | 0.3324 | 55.6577% | 55.9752% |
| MPS_MSMS+ | test | 0.1745 | 44.8070% | 43.8538% |
| MPS_MSMS+ | validation | 0.1746 | 43.3371% | 42.3715% |
| MPS_MSMS- | test | 0.1712 | 43.2151% | 38.6525% |
| MPS_MSMS- | validation | 0.1722 | 41.7265% | 36.2590% |
| TTN_IR | test | 0.6070 | 35.3186% | 32.3966% |
| TTN_IR | validation | 0.5874 | 32.7094% | 29.8180% |
| TTN_H-NMR | test | 0.3143 | 65.1864% | 70.2090% |
| TTN_H-NMR | validation | 0.3172 | 64.8856% | 69.5417% |
| TTN_C-NMR | test | 0.3286 | 63.3078% | 72.7273% |
| TTN_C-NMR | validation | 0.3290 | 62.7674% | 72.3077% |
| TTN_MSMS+ | test | 0.1579 | 85.5936% | 96.3025% |
| TTN_MSMS+ | validation | 0.1576 | 85.6605% | 97.1143% |
| TTN_MSMS- | test | 0.1518 | 86.1154% | 98.0392% |
| TTN_MSMS- | validation | 0.1511 | 86.1722% | 97.9409% |
| XGB_IR | test | 0.5460 | 53.6676% | 55.2209% |
| XGB_IR | validation | 0.5458 | 53.5654% | 55.6585% |
| XGB_H_NMR | test | 0.3677 | 68.7854% | 80.8429% |
| XGB_H_NMR | validation | 0.3791 | 70.9277% | 81.1784% |
| XGB_C_NMR | test | 0.4450 | 66.2805% | 74.2574% |
| XGB_C_NMR | validation | 0.4687 | 63.9473% | 74.4828% |
| XGB_MSMS+ | test | 0.1049 | 87.3980% | 99.9136% |
| XGB_MSMS+ | validation | 0.1055 | 90.0731% | 99.8917% |
| XGB_MSMS- | test | 0.1120 | 89.3550% | 99.9474% |
| XGB_MSMS- | validation | 0.1122 | 89.3472% | 99.9619% |
