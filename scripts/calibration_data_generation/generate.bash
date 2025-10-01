#!/bin/bash
enviroment="ARMOR_main"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment
#get the model name, and the 
model=$1
dataset=$2
n_samples=$3

#if we get a 4th argument, use it as the seqlen
if [ -z "$4" ]; then
    seqlen=-1
    echo "Using default seqlen of maximum context length."
else
    seqlen=$4
fi
echo "Generating calibration data for model: $model, dataset: $dataset, blockwise: $blockwise, n_samples: $n_samples, seqlen: $seqlen"

#if we are doing blockwise 

echo "Using standard generation script."
python scripts/calibration_data_generation/generate.py \
    --model $model \
    --dataset config/dataset/${dataset}.yaml \
    --seqlen $seqlen \
    --n_samples $n_samples \
    --seed 0 \
    --forward_batch_size 1 \
    --log_object hessian_diag \
    --save_path "../LLM_data" \
    --save_weights \
    --save_calibration_data

