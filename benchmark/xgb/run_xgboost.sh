# H NMR
python ./benchmark/xgb/scripts/run_xgb_baseline.py \
    --analytical_data ./data/raw/ \
    --out_path ./benchmark/xgb/models/h_nmr \
    --column h_nmr_spectra

# C NMR
python ./benchmark/xgb/scripts/run_xgb_baseline.py \
    --analytical_data ./data/raw/ \
    --out_path ./benchmark/xgb/models/c_nmr \
    --column c_nmr_spectra

# IR
python ./benchmark/xgb/scripts/run_xgb_baseline.py \
    --analytical_data ./data/raw/ \
    --out_path ./benchmark/xgb/models/ir \
    --column ir_spectra

# Pos MSMS
python ./benchmark/xgb/scripts/run_xgb_baseline.py \
    --analytical_data ./data/raw/ \
    --out_path ./benchmark/xgb/models/pos_msms \
    --column pos_msms

# Neg MSMS
python ./benchmark/xgb/scripts/run_xgb_baseline.py \
    --analytical_data ./data/raw/ \
    --out_path ./benchmark/xgb/models/neg_msms \
    --column neg_msms

# Evaluate and visualize results
echo "Evaluating XGBoost models..."
python ./benchmark/xgb/scripts/evaluate_results.py \
--models_dir ./benchmark/xgb/models \
--output_dir ./benchmark/xgb/results