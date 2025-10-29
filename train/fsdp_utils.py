# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel import Bagel
from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
        max_save_num=10,
    ):

        # <<< ADDED: Keep only 5 most recent checkpoints >>>
        if dist.get_rank() == 0:
            subdirs = [
                d for d in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()
            ]
            subdirs_sorted = sorted(subdirs, key=lambda x: int(x))
            if len(subdirs_sorted) >= max_save_num:
                num_to_remove = len(subdirs_sorted) - max_save_num + 1  # we keep 4 old + the one we're about to save
                for old_ckpt in subdirs_sorted[:num_to_remove]:
                    old_ckpt_path = os.path.join(ckpt_dir, old_ckpt)
                    logger.info(f"Removing old checkpoint: {old_ckpt_path}")
                    shutil.rmtree(old_ckpt_path)  # Remove old checkpoint directory

        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        ema_save_path = os.path.join(save_path, "ema")
        os.makedirs(ema_save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        gc.collect()
        if ema_model is not None and False:
            with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    ema_model.cpu()
                    logger.info(f"Saving ema checkpoint to {ema_save_path}.")
                    ema_model.save_pretrained(ema_save_path, state_dict=ema_state_dict, safe_serialization=True, max_shard_size="1GB", variant='ema')
                del ema_state_dict
                gc.collect()
                # ema_model.to(torch.device('cuda'))
            # if dist.get_rank() == 0:
            #     ema_model.save_pretrained(ema_save_path, safe_serialization=True, max_shard_size="1GB", variant='ema')

        with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            state_dict = model.state_dict()
            # state_dict = FSDP.state_dict(model, state_dict_type=FSDP.StateDictType.FULL_STATE_DICT)
            if dist.get_rank() == 0:
                # model.cpu()
                logger.info(f"Saving checkpoint to {save_path}.")
                model.save_pretrained(save_path, state_dict=state_dict, safe_serialization=True, max_shard_size="1GB")
                # model.to(torch.device('cuda'))
            del state_dict
            gc.collect()
        # if dist.get_rank() == 0:
        #     logger.info('NOT SAVING CHK')
        #     model.save_pretrained(save_path, safe_serialization=True, max_shard_size="1GB")

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                # torch.save(optimizer.state_dict(), optimizer_save_path)
                logger.info(f"NOT Saving optimizer checkpoint to {optimizer_save_path}.")
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                # if dist.get_rank() < fsdp_config.num_shard:
                #     torch.save(optimizer.state_dict(), optimizer_save_path)
                logger.info(f"NOT Saving optimizer checkpoint to {optimizer_save_path}.")
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, "ema")
                model_variant = "ema"
            else:
                model_state_dict_path = resume_from
                model_variant = None

            logger.info(f"resume model from loaded {model_state_dict_path}.")

            # with FSDP.state_dict_type(
            #         model,
            #         StateDictType.FULL_STATE_DICT,
            #         FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            # ):
            loaded_state_dict = Bagel.from_pretrained(
                model_state_dict_path,
                variant=model_variant,
                # return_unused_kwargs=True,
                # torch_dtype=torch.float32,  # or torch.float16 if you used it
                low_cpu_mem_usage=True,  # helps reduce memory
                device_map=None  # keep weights on CPU for FSDP
            ).state_dict()
            # Step 4: Load into current FSDP-wrapped model
            model.load_state_dict(loaded_state_dict)

            if ema_model is not None:
                if resume_from_ema or not os.path.exists(os.path.join(resume_from, "ema")):
                    logger.info(f"resume ema from loaded ema.")
                    ema_model.load_state_dict(loaded_state_dict)
                    del loaded_state_dict
                else:
                    del loaded_state_dict
                    logger.info(f"resume ema from ema.")
                    # with FSDP.state_dict_type(
                    #         ema_model,
                    #         StateDictType.FULL_STATE_DICT,
                    #         FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
                    ema_loaded_state_dict = Bagel.from_pretrained(
                        os.path.join(resume_from, "ema"),
                        variant='ema',
                        # return_unused_kwargs=True,
                        # torch_dtype=torch.float32,  # or torch.float16 if you used it
                        low_cpu_mem_usage=True,  # helps reduce memory
                        device_map=None  # keep weights on CPU for FSDP
                    ).state_dict()
                    ema_model.load_state_dict(ema_loaded_state_dict)
                    del ema_loaded_state_dict
            else:
                del loaded_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            # optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            # optimizer.load_state_dict(optimizer_state_dict)
            # del optimizer_state_dict
            print(f"NOT Loading optimizer state from {optimizer_state_dict_path}.")

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            if os.path.exists(scheduler_state_dict_path):
                scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
                scheduler.load_state_dict(scheduler_state_dict)
                del scheduler_state_dict
            else:
                print(f'NOT Loading scheduler state from {scheduler_state_dict_path}.')

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
