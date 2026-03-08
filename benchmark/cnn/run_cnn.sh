# export HF_DATASETS_CACHE= SET IT HERE
# export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64

# Common settings for fair comparison
SEED=42
COLUMNS="h_nmr_spectra,c_nmr_spectra,ir_spectra,pos_msms,neg_msms"

# Train ORIGINAL models (without K-Fold) - uses same 80/20 test split
echo "Training ORIGINAL models (no K-Fold)..."
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
    --analytical_data ./data/raw/ \
    --base_out_path ./benchmark/cnn/models \
    --columns $COLUMNS \
    --seed $SEED \
    --no_kfold

# Train K-FOLD models - uses same 80/20 test split
echo "Training K-FOLD models..."
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
    --analytical_data ./data/raw/ \
    --base_out_path ./benchmark/cnn/models \
    --columns $COLUMNS \
    --seed $SEED \
    --use_kfold \
    --n_folds 5

# Evaluate and visualize results
echo "Evaluating CNN models..."
python ./benchmark/cnn/scripts/evaluate_results.py \
    --models_dir ./benchmark/cnn/models \
    --output_dir ./benchmark/cnn/results
