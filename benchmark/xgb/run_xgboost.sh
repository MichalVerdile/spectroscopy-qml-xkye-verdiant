# H NMR
#python ./benchmark/xgb/scripts/run_xgb_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/xgb/models_600/h_nmr \
#    --columns h_nmr_spectra --no_kfold

# C NMR
#python ./benchmark/xgb/scripts/run_xgb_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/xgb/models_600/c_nmr \
#    --columns c_nmr_spectra --no_kfold

# IR
#python ./benchmark/xgb/scripts/run_xgb_baseline.py \
#    --analytical_data ./data/raw/ \
#    --out_path ./benchmark/xgb/models_600/ir \
#    --column ir_spectra

# Pos MSMS
#python ./benchmark/xgb/scripts/run_xgb_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/xgb/models_600/pos_msms \
#    --columns pos_msms --no_kfold

# Neg MSMS
#python ./benchmark/xgb/scripts/run_xgb_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/xgb/models_600/neg_msms \
#    --columns neg_msms --no_kfold

# Evaluate and visualize results
echo "Evaluating XGBoost models..."
python ./benchmark/xgb/scripts/evaluate_results.py \
--models_dir ./benchmark/xgb/models_600 \
--output_dir ./benchmark/xgb/results_600

#python ./benchmark/xgb/scripts/analyse_xgb.py --results-pickle benchmark/xgb/models_600/ir/original/results.pickle --output-dir benchmark/cnn/results/evaluation/ir
#python ./benchmark/xgb/scripts/analyse_xgb.py --results-pickle benchmark/xgb/models_600/cnmr/original/results.pickle --output-dir benchmark/cnn/results/evaluation/cnmr
#python ./benchmark/xgb/scripts/analyse_xgb.py --results-pickle benchmark/xgb/models_600/hnmr/original/results.pickle --output-dir benchmark/cnn/results/evaluation/hnmr
#python ./benchmark/xgb/scripts/analyse_xgb.py --results-pickle benchmark/xgb/models_600/neg_msms/original/results.pickle --output-dir benchmark/cnn/results/evaluation/neg_msms
#python ./benchmark/xgb/scripts/analyse_xgb.py --results-pickle benchmark/xgb/models_600/pos_msms/original/results.pickle --output-dir benchmark/cnn/results/evaluation/pos_msms



#python ./benchmark/xgb/scripts/evaluate_error.py --model_path ./benchmark/xgb/models_600/ir/ir_xgboost_model.pickle --results_pickle_path ./benchmark/xgb/models_600/ir/results.pickle --column ir_spectra --base_out_path ./benchmark/xgb/results_600/ir_evaluation
#python ./benchmark/xgb/scripts/evaluate_error.py --model_path ./benchmark/xgb/models_600/h_nmr/hnmr/original/hnmr_xgboost_model.pickle --results_pickle_path ./benchmark/xgb/models_600/h_nmr/hnmr/original/results.pickle --column h_nmr_spectra --base_out_path ./benchmark/xgb/results_600/hnmr_evaluation
#python ./benchmark/xgb/scripts/evaluate_error.py --model_path ./benchmark/xgb/models_600/c_nmr/cnmr/original/cnmr_xgboost_model.pickle --results_pickle_path ./benchmark/xgb/models_600/c_nmr/cnmr/original/results.pickle --column c_nmr_spectra --base_out_path ./benchmark/xgb/results_600/cnmr_evaluation
#python ./benchmark/xgb/scripts/evaluate_error.py --model_path ./benchmark/xgb/models_600/neg_msms/neg_msms/original/neg_msms_xgboost_model.pickle --results_pickle_path ./benchmark/xgb/models_600/neg_msms/neg_msms/original/results.pickle --column neg_msms --base_out_path ./benchmark/xgb/results_600/neg_msms_evaluation
#python ./benchmark/xgb/scripts/evaluate_error.py --model_path ./benchmark/xgb/models_600/pos_msms/pos_msms/original/pos_msms_xgboost_model.pickle --results_pickle_path ./benchmark/xgb/models_600/pos_msms/pos_msms/original/results.pickle --column pos_msms --base_out_path ./benchmark/xgb/results_600/pos_msms_evaluation