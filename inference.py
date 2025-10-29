import argparse
import gc
import os
import time
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights, dispatch_model
from transformers.utils.model_parallel_utils import get_device_map

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


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="fast-bagel")
parser.add_argument("--model_variant", type=str, default="")
parser.add_argument("--num_timesteps", type=int, default=20)
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

output_prefix=f'm{base_name}-u{use_heun}-s{num_timesteps}-t{time_shift}-cfg{args.text_cfg}-{args.image_cfg}'

tokenizer_path = args.model_dir  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
model_path = args.model_dir # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)


model = Bagel.from_pretrained(model_path, device_map='auto',
                              use_heun=use_heun,
                              torch_dtype=torch.bfloat16, trust_remote_code=True, variant=model_variant)

model.to(torch.bfloat16)


first_device = "cuda:0"
last_device = "cuda:" + str(torch.cuda.device_count() -1)

max_mem_per_gpu = {
    i: f"{int((torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) * 1.0 / (1024 ** 3))}GiB"
    for i in range(torch.cuda.device_count())
}

device_map = infer_auto_device_map(
    model,
    max_memory=max_mem_per_gpu,
    no_split_module_classes=["Qwen2MoTDecoderLayer"],
)

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    para_device_map = get_device_map(len(model.language_model.model.layers), range(0, torch.cuda.device_count()))

    print(f'para_device_map: {para_device_map}')
    for k, v in para_device_map.items():
        cuda_device = "cuda:" + str(k)
        for layer in v:
            # model.language_model.model.layers[layer] = model.language_model.model.layers[layer].to(cuda_device)
            device_map[f'language_model.model.layers.{layer}'] = cuda_device
else:
    print('Using single GPU')
    model = model.cuda().eval()

first_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'connector',
    'vit_pos_embed',
    'vit_model',
    # 'llm2vae',
]

last_device_modules = [
    'llm2vae',
]

# first_device = device_map.get(same_device_modules[0], "cuda:0")
for k in first_device_modules:
    device_map[k] = first_device
for k in last_device_modules:
    device_map[k] = last_device

for k, v in device_map.items():
    module_list = k.split('.')
    module = model
    for module_name in module_list:
        if hasattr(module, module_name):
            module = getattr(module, module_name)
        else:
            print(f'Warning: Module {module_name} not found in {k}')
            break
    if isinstance(module, torch.nn.Module):
        print(f'Setting device for {k} to {v}')
        module.to(v)

# model = dispatch_model(model, device_map=device_map, offload_dir="/tmp/offload", offload_buffers=False, force_hooks=True)

# model = model.to('cuda').eval().to(torch.bfloat16)
# model.language_model.parallelize()
print(f'Model loaded device {model.device}')


# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(tokenizer_path, "ae.safetensors"))
print(f'load vae model {vae_model.decoder.norm_out.weight.data.dtype} {vae_model.decoder.norm_out.weight.data.device}')

vae_model = vae_model.cuda().eval().to(torch.bfloat16)
# vae_model = vae_model.to(first_device)
vae_model.encoder = vae_model.encoder.to(first_device)
vae_model.decoder = vae_model.decoder.to(last_device)

# Print device of each submodule
for name, module in model.named_modules():
    # Get the first parameter (if any) to determine device
    try:
        param = next(module.parameters())
        print(f"{name or '[root]'}: {param.device}")
    except StopIteration:
        # Module has no parameters
        print(f"{name or '[root]'}: No parameters")

from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
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


inference_hyper=dict(
    cfg_text_scale=args.text_cfg,
    cfg_img_scale=args.image_cfg,
    cfg_interval=[0.4, 1.0],
    timestep_shift=time_shift,
    num_timesteps=num_timesteps,
    cfg_renorm_min=1.0,
    cfg_renorm_type="text_channel",
)


image = Image.open('test_images/women.jpg')
# prompt = 'She boards a modern subway, quietly reading a newspaper.'
prompts = [
    'The woman boards a modern subway, quietly reading a newspaper. keep the face to the same person. keep the clothes color unchanged.',
    'change the clothes color to green',
    'Convert the photo into a Pixar-style animated scene. Stylize the person into a 3D cartoon character with expressive eyes, smooth skin, and rounded features, while keeping their hair, clothing, and identity recognizable. Place them in a colorful, cinematic environment.',
    'add a green hat',
    'add a red hat',
    'change the hair to green',
        'change the hair to red',
        'change the photo to cartoon style',
        'make the background in beach'
    # 'change scene to a modern subway, the woman is reading a newspaper.'
]

for idx, prompt in enumerate(prompts):
    print(prompt)
    print('-'*10)
    start_time = time.time()
    output_dict = inferencer(image=image, text=prompt, **inference_hyper)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    print(f'finsh inference output to heun-{output_prefix}-{idx}.webp')
    output_dict['image'].save(f'heun-{output_prefix}-{idx}.webp')

