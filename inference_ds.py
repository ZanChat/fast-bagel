import argparse
import os
import time


from PIL import Image
import torch
import deepspeed

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
import sys
import torch.distributed.tensor as dtensor


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="fast-bagel")
parser.add_argument("--model_variant", type=str, default="")
parser.add_argument("--num_timesteps", type=int, default=20, nargs='+')
parser.add_argument("--time_shift", type=float, default=3)
parser.add_argument("--text_cfg", type=float, default=4)
parser.add_argument("--image_cfg", type=float, default=2)

parser.add_argument("--disable_heun", type=bool, default=False)

args, unknown = parser.parse_known_args()  # This ignores DeepSpeed args

use_heun = not args.disable_heun
num_timesteps = args.num_timesteps
time_shift = args.time_shift
model_variant = args.model_variant if args.model_variant else None
base_name = '-'.join(args.model_dir.split('/')[-3:])

cur_time = time.time()



tokenizer_path = args.model_dir  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
model_path = args.model_dir # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

print(f'infer on {model_path}-{model_variant} of {num_timesteps} steps use heun {use_heun}')

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

gpu_cnt = torch.cuda.device_count()

with torch.no_grad():

    model = Bagel.from_pretrained(model_path, device_map='cuda',
                                  use_heun=use_heun,
                                  torch_dtype=torch.fl, trust_remote_code=True, variant=model_variant)

    model = model.cuda().to(torch.bfloat16).eval()

    compile_mode = 'max-autotune'

    # model = torch.compile(model, mode=compile_mode)

    if gpu_cnt == 1:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["TORCH_DTENSOR_LAZY_INIT"] = "1"

    print(f'Using {gpu_cnt} GPUs')
    ds_model = deepspeed.init_inference(
            model,
            mp_size=gpu_cnt,  # number of GPUs for model parallelism
            dtype=torch.bfloat16,
            replace_with_kernel_inject=gpu_cnt > 1  # use DeepSpeed's optimized kernels
    )

    print(f'Model loaded device {model.device}')


    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(tokenizer_path, "ae.safetensors"))
    print(f'load vae model {vae_model.decoder.norm_out.weight.data.dtype} {vae_model.decoder.norm_out.weight.data.device}')

    vae_model = vae_model.cuda().eval().to(torch.bfloat16)

    # vae_model = torch.compile(vae_model, mode=compile_mode)

    ds_vae_model = deepspeed.init_inference(
        vae_model,
        mp_size=gpu_cnt,  # number of GPUs for model parallelism
        dtype=torch.bfloat16,
        replace_with_kernel_inject=True  # use DeepSpeed's optimized kernels
    )

    from inferencer import InterleaveInferencer

    inferencer = InterleaveInferencer(
        model=ds_model.module,
        vae_model=ds_vae_model.module,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )

    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prompts = [
        ('test_images/women.jpg', 'add a red hat')
    ]
    total_time = 0.0

    for step in num_timesteps:
        for idx, prompt in enumerate(prompts):
            print(prompt)
            print('-'*10)
            if type(prompt) is tuple:
                image_path, prompt = prompt
                image = Image.open(image_path)

            inference_hyper = dict(
                cfg_text_scale=args.text_cfg,
                cfg_img_scale=args.image_cfg,
                cfg_interval=[0.4, 1.0],
                timestep_shift=time_shift,
                num_timesteps=step,
                cfg_renorm_min=1.0,
                cfg_renorm_type="text_channel",
            )
            start_time = time.time()
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            output_dict = inferencer(image=image, text=prompt, **inference_hyper)
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")

            output_prefix = f'm{base_name}-u{use_heun}-s{step}-t{time_shift}-cfg{args.text_cfg}-{args.image_cfg}-{cur_time}'

            print(f'finsh inference output to heun-{output_prefix}-{idx}.webp')
            output_dict['image'].save(f'heun-{output_prefix}-{idx}.webp')
        print(f'avg time is {total_time/len(prompts)}')
