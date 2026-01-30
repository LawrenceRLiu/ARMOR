#!/bin/bash

enviroment="ARMOR_main"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment


model_name=$1 #example: "meta-llama/Llama-2-7b-hf"

gpus=$2 #specify which gpus to use change this for your machine
        
log_dir="./logs/"
datetime=$(date +"%Y%m%d_%H%M%S")
block_size=128
n_iters=20000
seqlen=4096
resume=false

echo "Compressing model: $model_name"
echo "Generating data"
generate_cmd="CUDA_VISIBLE_DEVICES=${gpus} scripts/calibration_data_generation/generate.bash ${model_name} SlimPajama-627B 128 ${seqlen}"
echo "$generate_cmd"
eval "$generate_cmd"
if [ $? -ne 0 ]; then
    echo "Data generation failed for model: $model_name"
    return 1
fi
echo "generated data"
log_dir_use="${log_dir}/${model_name}/"
mkdir -p "$log_dir_use"
echo "Running compression for model: $model_name"
results_path="models/${model_name}/compressed/BlockPrune/${block_size}_${n_iters}/${datetime}"
if [ -d "$results_path" ] && [ "$resume" = false ]; then
    echo "Results path $results_path already exists. Skipping compression."
else
    echo "compressing to $results_path"
    scripts/compress/ARMOR.bash run_name=${datetime} \
        model="${model_name}" \
        block_size=$block_size \
        n_iters=$n_iters \
        dataset_config="[{dataset_config:SlimPajama-627B,n_samples:128,ctx_len:${seqlen}}]" \
        gpus="${gpus}" > "${log_dir_use}/BlockPrune_${block_size}_${n_iters}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "Compression failed for model: $model_name"
        exit 1
    fi
    echo "Compression completed for model: $model_name"
fi


CUDA_VISIBLE_DEVICES=$gpus python scripts/evaluation/ppl.py \
    --model_name ${model_name} \
    --model_path ${results_path}/model \
    --seqlen ${seqlen} \
    --load_custom_model \
    --save --results_path ${results_path}/ppl.yaml > "${log_dir_use}/ppl_compressed.log" 2>&1
