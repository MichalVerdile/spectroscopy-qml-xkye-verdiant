# export HF_DATASETS_CACHE= SET IT HERE
# export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64


# HNMR
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
--analytical_data ./data/raw/ \
--out_path ./benchmark/cnn/models/hnmr \
--column h_nmr_spectra

# CNMR
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
--analytical_data ./data/raw/ \
--out_path ./benchmark/cnn/models/cnmr \
--column c_nmr_spectra

# IR
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
--analytical_data ./data/raw/ \
--out_path ./benchmark/cnn/models/ir \
--column ir_spectra

# Pos MSMS
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
--analytical_data ./data/raw/ \
--out_path ./benchmark/cnn/models/pos_msms \
--column pos_msms

# Neg MSMS
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
--analytical_data ./data/raw/ \
--out_path ./benchmark/cnn/models/neg_msms \
--column neg_msms

# Evaluate and visualize results
echo "Evaluating CNN models..."
python ./benchmark/cnn/scripts/evaluate_results.py \
--models_dir ./benchmark/cnn/models \
--output_dir ./benchmark/cnn/results
