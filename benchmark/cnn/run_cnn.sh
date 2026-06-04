# export HF_DATASETS_CACHE= SET IT HERE
# export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64

# Common settings for fair comparison
SEED=42
COLUMNS="h_nmr_spectra"

# Train ORIGINAL models (without K-Fold) - uses same 80/20 test split
#echo "Training ORIGINAL models (no K-Fold)..."
#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models_600_LR \
#    --columns h_nmr_spectra \
#    --seed $SEED \
#    --no_kfold
#
#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models_600_LR \
#    --columns pos_msms \
#    --seed $SEED \
#    --no_kfold

#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models_600_LR \
#    --columns neg_msms \
#    --seed $SEED \
#    --no_kfold
#
#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models_600_LR \
#    --columns ir_spectra \
#    --seed $SEED \
#    --no_kfold

#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models_600_LR \
#    --columns c_nmr_spectra \
#    --seed $SEED \
#    --no_kfold

#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models \
#    --columns pos_msms \
#    --seed $SEED \
#    --no_kfold

#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models \
#    --columns neg_msms \
#    --seed $SEED \
#    --no_kfold

#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --out_path ./benchmark/cnn/models \
#    --column $COLUMNS \
#    --seed $SEED

## Train K-FOLD models - uses same 80/20 test split
#echo "Training K-FOLD models..."
#python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
#    --analytical_data ./data/raw/ \
#    --base_out_path ./benchmark/cnn/models \
#    --columns $COLUMNS \
#    --seed $SEED \
#    --use_kfold \
#    --n_folds 5

# Evaluate and visualize results
echo "Evaluating ORIGINAL CNN models..."
python ./benchmark/cnn/scripts/evaluate_results.py \
    --models_dir ./benchmark/cnn/models_600_LR \
    --output_dir ./benchmark/cnn/results_600_LR \
    --model_type original

#echo "Evaluating K-FOLD CNN models..."
#python ./benchmark/cnn/scripts/evaluate_results.py \
#    --models_dir ./benchmark/cnn/models \
#    --output_dir ./benchmark/cnn/results \
#    --model_type k_fold


#python ./benchmark/cnn/scripts/evaluate_error.py --model_path ./benchmark/cnn/models/ir/original/ir_model.keras --results_pickle_path ./benchmark/cnn/models/ir/original/results.pickle --column ir_spectra --base_out_path ./benchmark/cnn/results_600/ir_evaluation
#python ./benchmark/cnn/scripts/evaluate_error.py --model_path ./benchmark/cnn/models/hnmr/original/hnmr_model.keras --results_pickle_path ./benchmark/cnn/models/hnmr/original/results.pickle --column h_nmr_spectra --base_out_path ./benchmark/cnn/results_600/hnmr_evaluation
#python ./benchmark/cnn/scripts/evaluate_error.py --model_path ./benchmark/cnn/models/cnmr/original/cnmr_model.keras --results_pickle_path ./benchmark/cnn/models/cnmr/original/results.pickle --column c_nmr_spectra --base_out_path ./benchmark/cnn/results_600/cnmr_evaluation
#python ./benchmark/cnn/scripts/evaluate_error.py --model_path ./benchmark/cnn/models/neg_msms/original/neg_msms_model.keras --results_pickle_path ./benchmark/cnn/models/neg_msms/original/results.pickle --column neg_msms --base_out_path ./benchmark/cnn/results_600/neg_msms_evaluation
#python ./benchmark/cnn/scripts/evaluate_error.py --model_path ./benchmark/cnn/models/pos_msms/original/pos_msms_model.keras --results_pickle_path ./benchmark/cnn/models/pos_msms/original/results.pickle --column pos_msms --base_out_path ./benchmark/cnn/results_600/pos_msms_evaluation

#python ./benchmark/cnn/scripts/analyse_cnn.py --results-pickle benchmark/cnn/models/cnmr/original/results.pickle --output-dir benchmark/cnn/results/evaluation/cnmr
#python ./benchmark/cnn/scripts/analyse_cnn.py --results-pickle benchmark/cnn/models/hnmr/original/results.pickle --output-dir benchmark/cnn/results/evaluation/hnmr
#python ./benchmark/cnn/scripts/analyse_cnn.py --results-pickle benchmark/cnn/models/neg_msms/original/results.pickle --output-dir benchmark/cnn/results/evaluation/neg_msms
#python ./benchmark/cnn/scripts/analyse_cnn.py --results-pickle benchmark/cnn/models/pos_msms/original/results.pickle --output-dir benchmark/cnn/results/evaluation/pos_msms
