#!/bin/bash
# export HF_DATASETS_CACHE= SET IT HERE
# export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64

# Create logs directory
mkdir -p ./benchmark/cnn/logs

# Train all CNN models in a single run (data loaded only once)
echo "Starting CNN training... (output saved to ./benchmark/cnn/logs/training.log)"
python ./benchmark/cnn/scripts/run_cnn_jung_baseline.py \
--analytical_data ./data/raw/ \
--base_out_path ./benchmark/cnn/models \
--columns h_nmr_spectra,c_nmr_spectra,ir_spectra,pos_msms,neg_msms \
2>&1 | tee ./benchmark/cnn/logs/training.log

# Check if training succeeded (use PIPESTATUS to get python exit code, not tee)
TRAIN_EXIT=${PIPESTATUS[0]}
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $TRAIN_EXIT"
    echo "  Check ./benchmark/cnn/logs/training.log for details"
    exit 1
fi

# Evaluate and visualize results
echo "Evaluating CNN models..."
python ./benchmark/cnn/scripts/evaluate_results.py \
--models_dir ./benchmark/cnn/models \
--output_dir ./benchmark/cnn/results \
2>&1 | tee ./benchmark/cnn/logs/evaluation.log

EVAL_EXIT=${PIPESTATUS[0]}
if [ $EVAL_EXIT -eq 0 ]; then
    echo "Evaluation completed successfully"
else
    echo "Evaluation failed with exit code $EVAL_EXIT"
    echo "  Check ./benchmark/cnn/logs/evaluation.log for details"
    exit 1
fi

echo "All done! Check ./benchmark/cnn/logs/ for detailed logs"
