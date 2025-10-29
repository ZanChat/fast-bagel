# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# run this script at the root of the project folder
#pip install httpx==0.23.0
#pip install openai==1.87.0
#pip install datasets
#pip install megfile
export TRITON_CACHE_DIR="/tmp/triton_cache"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTHONPATH=.

MODEL_NAME=$1
MODEL_NAME_STR=$2
NUM_STEPS=$3
N_GPU=2  # Number of GPU used in for the evaluation

MODEL_PATH="/some/path/Step1X-Edit/${MODEL_NAME}"

OUTPUT_DIR="/some/path/GEdit-Step1x/"


GEN_DIR="$OUTPUT_DIR/gen_image"
LOG_DIR="$OUTPUT_DIR/logs"

AZURE_ENDPOINT="https://azure_endpoint_url_you_use"  # set up the azure openai endpoint url
AZURE_OPENAI_KEY="0"  # set up the azure openai key
N_GPT_PARALLEL=1


mkdir -p "$OUTPUT_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$LOG_DIR"

echo "Model Name: $MODEL_NAME | Num Steps: $NUM_STEPS"
# # ----------------------------
# #    Download GEdit Dataset
# # ----------------------------
#python -c "from datasets import load_dataset; dataset = load_dataset('stepfun-ai/GEdit-Bench')"
ds_path='/some/path/GEdit-Bench/'


# # ---------------------
# #    Image Gneration
# # ---------------------

#cd eval/gen/gedit

python3 -u gen_images_gedit-step1x.py --offload --input_dir $ds_path --model_path "$MODEL_PATH"  --output_dir "$OUTPUT_DIR" --seed 1234 --size_level 1024 --version v1.0 --model_name "$MODEL_NAME_STR" --task_list "${@:4}" >> out-eval-2-$MODEL_NAME_STR 2>&1

#for ((i=0; i<$N_GPU; i++)); do
#    CUDA_VISIBLE_DEVICES=$i python3 -u gen_images_gedit-step1x.py --offload --input_dir $ds_path --model_path "$MODEL_PATH"  --output_dir "$OUTPUT_DIR" --seed 1234 --size_level 1024 --version v1.0 --model_name "$MODEL_NAME_STR" --task_list "${@:4}" >> out-eval-$MODEL_NAME_STR 2>&1 #2>&1 | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
#done



#for ((i=0; i<$N_GPU; i++)); do
#    CUDA_VISIBLE_DEVICES=$i nohup python3 -u eval/gen/gedit/gen_images_i2ebench.py --model_path "$MODEL_PATH" --ds_path $ds_path  --output_dir "$OUTPUT_DIR"  --shard_id $i --total_shards "$N_GPU" --num_timesteps 50 --cfg_img_scale 2.0 --cfg_text_scale 4.0 --timestep_shift 4.0 --device cuda --model_name "$MODEL_NAME_STR" --task_list "${@:4}" >> out-eval-$MODEL_NAME_STR 2>&1 & #2>&1 | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
#done

