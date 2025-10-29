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

NUM_STEPS=20
N_GPU=4  # Number of GPU used in for the evaluation
MODEL_PATH="/some/path/FLUX.1-Kontext-dev"
OUTPUT_DIR="/some/path/gedit-bench-flux-s${NUM_STEPS}/results"
GEN_DIR="$OUTPUT_DIR/gen_image"
LOG_DIR="$OUTPUT_DIR/logs"

AZURE_ENDPOINT="https://azure_endpoint_url_you_use"  # set up the azure openai endpoint url
AZURE_OPENAI_KEY="0"  # set up the azure openai key
N_GPT_PARALLEL=1


mkdir -p "$OUTPUT_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$LOG_DIR"


# # ----------------------------
# #    Download GEdit Dataset
# # ----------------------------
#python -c "from datasets import load_dataset; dataset = load_dataset('stepfun-ai/GEdit-Bench')"
ds_path='/some/path/GEdit-Bench'
echo "Dataset Downloaded in llm dir $ds_path"


# # ---------------------
# #    Generate Images
# # ---------------------
for ((i=0; i<$N_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup python3 eval/gen/gedit/gen_images_gedit_flux.py --model_path "$MODEL_PATH" --ds_path $ds_path  --output_dir "$GEN_DIR"  --shard_id $i --total_shards "$N_GPU" --num_timesteps $NUM_STEPS  2>&1 | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
done

wait
echo "Image Generation Done"


# # ---------------------
# #    GPT Evaluation
# # ---------------------


#cd eval/gen/gedit
#CUDA_VISIBLE_DEVICES=0,1 python test_gedit_score.py --save_path "$OUTPUT_DIR" --ds_path $ds_path --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY" --backbone "qwen25vl" --max_workers "$N_GPT_PARALLEL" --groups "background_change" "color_alter" "material_alter"  "subject-replace" "text_change" &
#CUDA_VISIBLE_DEVICES=2,3 python test_gedit_score.py --save_path "$OUTPUT_DIR" --ds_path $ds_path --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY" --backbone "qwen25vl" --max_workers "$N_GPT_PARALLEL" --groups "ps_human" "style_change" "subject-add" "motion_change" "subject-remove" "tone_transfer" &


#     "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
#python test_gedit_score.py --save_path "$OUTPUT_DIR" --ds_path $ds_path --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY" --backbone "qwen25vl" --max_workers "$N_GPT_PARALLEL" --groups "background_change" "color_alter" "material_alter" "motion_change" &
#python test_gedit_score.py --save_path "$OUTPUT_DIR" --ds_path $ds_path --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY" --backbone "qwen25vl" --max_workers "$N_GPT_PARALLEL" --groups "ps_human" "style_change" "subject-add" "subject-remove"&
#python test_gedit_score.py --save_path "$OUTPUT_DIR" --ds_path $ds_path --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY" --backbone "qwen25vl" --max_workers "$N_GPT_PARALLEL" --groups "subject-replace" "text_change" "tone_transfer" &
echo "Evaluation Done"
#
#
## # --------------------
## #    Print Results
## # --------------------
#python calculate_statistics.py --save_path "$OUTPUT_DIR"  --language en --backend "qwen25vl" > "$MODEL_NAME-s$NUM_STEPS-stat.txt"

