#!/bin/bash

module load python

mamba activate olmo2

folder=/n/holyscratch01/sham_lab/summer_2024/checkpoints/test-attention_layer_norm-false-39333978/step3000-unsharded

python ../scripts/convert_olmo_to_hf_new.py --input_dir="$folder" --output_dir="${folder}-hf" --tokenizer_json_path=../olmo_data/tokenizers/8-token-tokenizer.json