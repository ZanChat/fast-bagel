# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
export PYTHONPATH=.
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=180000000
export NCCL_DEBUG=INFO

export TRITON_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True



# replace the variables with your own
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/all_gen.yaml \
  --llm_path /some/path/bagel-7b-pretrained \
  --tokenizer_path /some/path/BAGEL-7B-MoT/ \
  --visual_und True \
  --freeze_vit True \
  --freeze_und True \
  --freeze_llm False \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.5 \
  --start_step 0 \
  --vae_path /some/path/ae.safetensors \
  --vit_path /some/path/BAGEL-7B-MoT/ \
  --use_flex True \
  --tie_word_embeddings False \
  --finetune_from_ema False \
  --resume_model_only False \
  --wandb_name "bagel-7b-mot-pretrain-20250726-2" \
  --num_shard 2 \
  --cpu_offload True \
  --save_every 200 \
  --auto_resume True \
  --acc_step 20 \
  --log_every 1 \
  --prefetch_factor 2 \
  --num_workers 2 \
  --warmup_steps 0 \
  --total_steps 8000 \
  --max_latent_size 70 \
  --max_num_tokens_per_sample 30000 \
  --max_num_tokens 36864 \
  --global_seed 222 \
  --ema 0.995 \
  --timestep_shift 4 \
  --lr 2.5e-5 \
  --results_dir /some/path/BAGEL-OUT \
  --checkpoint_dir /some/path/BAGEL-CHECK
