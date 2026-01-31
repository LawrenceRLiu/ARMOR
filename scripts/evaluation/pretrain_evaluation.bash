#!/bin/bash

#defualt args
declare gpus=4,5,6,7 #change this for your machine
declare model_name="google/gemma-2-9b"
declare model_path="default"
declare generate_non_compressed_model=false
declare wandb=false
declare results_path="default"
#lm_eval related stuff
declare apply_chat_template=false
declare fewshot_as_multiturn=false
declare additional_model_args=""
declare parallelize=false
declare num_processes=-1
declare max_length=-1
declare lm_eval_model_type="hf" #choices are hf, or vllm
declare lm_eval_batch_size="auto"
# declare task_groups="reasoning,math,coding"


#loop through the args
for arg in "$@"
do
    # Check if the argument is in key=value format
    if [[ "$arg" == *"="* ]]; then
        # Split into key and value
        key="${arg%%=*}"
        value="${arg#*=}"

        # 3. Use 'declare' again to update the variable
        # This will only affect variables already declared above.
        # For safety, you can check if the variable exists first.
        if declare -p "$key" &>/dev/null; then
            declare "$key=$value"
            echo "Updated argument '$key' to '$value'"
        else
            echo "WARN: Unknown argument '$key' ignored." >&2
        fi
    fi
done

if [ "$model_path" = "default" ]; then
    model_path=$model_name
    echo "Model path is set to default: $model_path"
     results_path_="./models/${model_name}/original/eval"
else
    results_path_="${model_path}/eval"
fi
#if we need to generate a non compressed model
if [ "$generate_non_compressed_model" = true ]; then
    echo "Generating non compressed model from $model_name to $model_path"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ARMOR_main
    CUDA_VISIBLE_DEVICES="$gpus" python -u scripts/misc/generate_non_compressed_model.py "$model_name" "$model_path"
    if [ $? -ne 0 ]; then
        echo "Command failed, check the log file"
        exit 1
    fi
    results_path_="${model_path}/eval"
    model_path="${model_path}/non_compressed_model"
fi

if [ "$results_path" = "default" ]; then
    results_path="$results_path_"
    echo "Results path is set to default: $results_path"
else
    echo "Results path is set to: $results_path"
fi

#add the current datetime to the results path, so we don't overwrite the results on accident
results_path=${results_path}/${current_datetime}
mkdir -p "$results_path"

#if "inst" or "chat" in the lower case of the model name, then we are evaluating a chat model
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
if [[ "$model_name_lower" == *"inst"* ]] || [[ "$model_name_lower" == *"chat"* ]]; then
    echo "Applying chat template to the model"
    apply_chat_template=true
fi
#if "inst" or "chat" in the lower case of the model name, then we are evaluating a chat model
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
if [[ "$model_name_lower" == *"inst"* ]] || [[ "$model_name_lower" == *"chat"* ]]; then
    echo "Applying chat template to the model"
    apply_chat_template=true
fi

#task based evaluation
export HF_ALLOW_CODE_EVAL="1"
enviroment="ARMOR_eval"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment
#apply model args
model_args="pretrained=$model_path"
if [ "$additional_model_args" != "" ]; then
    model_args+=",$additional_model_args"
fi
if [ "$parallelize" = true ]; then
    model_args+=",parallelize=True"
fi  
#if we are evaluating the gemma-3 family of models, we need to set the maximum length
#if max length is not -1 
if [ "$max_length" -ne -1 ]; then
    model_args+=",max_length=$max_length"
# if gemma-3 is in the model name, set max_length to 128000
elif [[ "$model_name" == "google/gemma-3"* ]]; then
    model_args+=",max_length=128000"
fi 

#count the number of gpus availible to overide the accelerate defualt config
num_gpus=$(echo $gpus | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1)) # Add 1 for the last GPU
echo "gpus available: $gpus, number of gpus: $num_gpus"

run_task() {
    local task_name=$1
    local n_shot=$2
    if [ "$lm_eval_model_type" == "vllm" ]; then 
        cmd="CUDA_VISIBLE_DEVICES=${gpus}"
    elif [ "$num_gpus" == 1 ]; then #if we only have one gpu, we don't need to use accelerate
        cmd="CUDA_VISIBLE_DEVICES=${gpus}"
    else
        if [ "$parallelize" = true ]; then
            #if num_processes is -1
            if [ "$num_processes" -eq -1 ]; then
                cmd="CUDA_VISIBLE_DEVICES=${gpus}"
            else
                cmd="python -m accelerate.commands.launch --num_processes $num_processes --gpu_ids $gpus --multi_gpu -m"
            fi
            # cmd="CUDA_VISIBLE_DEVICES=${gpus}"
        else
            cmd="python -m accelerate.commands.launch --num_processes $num_gpus --gpu_ids $gpus -m"
        fi
    fi

    cmd+=" lm_eval \
        --model ${lm_eval_model_type} \
        --model_args $model_args \
        --tasks $task_name \
        --num_fewshot ${n_shot} \
        --batch_size ${lm_eval_batch_size} \
        --confirm_run_unsafe_code \
        --output_path ${results_path}/${task_name}_${n_shot}.json"
    if [ "$fewshot_as_multiturn" = true ]; then
        cmd+=" --fewshot_as_multiturn"
    fi
    if [ "$apply_chat_template" = true ]; then
        cmd+=" --apply_chat_template"
    fi
    echo "$cmd"
    eval "$cmd"
    if [ $? -ne 0 ]; then
        echo "Command failed for task $task_name, check the log file"
        exit 1
    fi
}



export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

run_task "mmlu" 5
run_task "gsm8k" 8
run_task "arc_challenge" 25
run_task "hellaswag" 0
run_task "bbh" 3
# run_task "minerva_math" 4
run_task "winogrande" 5
run_task "gpqa_main_n_shot" 5
# run_task "humaneval" 0


#if we generated a non compressed model, we delete it to save space
if [ "$generate_non_compressed_model" = true ]; then
    echo "Deleting model path $model_path"
    rm -rf "$model_path"
fi
echo "run complete"
