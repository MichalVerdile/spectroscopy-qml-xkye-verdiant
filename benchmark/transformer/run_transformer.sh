#!/bin/bash

export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64

run_dir=./runs

# Run replication for structure prediction from spectra

# H NMR
python ./benchmark/transformer/generate_input.py \
        --analytical_data ./data/raw/ \
        --out_path ./benchmark/transformer/models/h_nmr \
        --formula \
        --h_nmr 

python ./benchmark/transformer/start_training.py \
        --out_path ./benchmark/transformer/models/h_nmr 
 

# C NMR
python ./benchmark/transformer/generate_input.py \
        --analytical_data  ./data/raw/ \
        --out_path ./benchmark/transformer/models/c_nmr \
        --formula \
        --c_nmr

python ./benchmark/transformer/start_training.py \
        --out_path ./benchmark/transformer/models/c_nmr 


# C NMR + H NMR
python ./benchmark/transformer/generate_input.py \
        --analytical_data  ./data/raw/ \
        --out_path ./benchmark/transformer/models/c_h_nmr \
        --formula \
        --h_nmr \
        --c_nmr

python ./benchmark/transformer/start_training.py \
        --out_path ./benchmark/transformer/models/c_h_nmr 

