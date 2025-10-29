# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from typing import List
import base64
import os

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

IMAGE_DATA_PREFIX = '/some/path/SEED-Data-Edit-Part2-3/conv_data'

def load_pil_image(image_data, prefix=None) -> Image.Image:
    if image_data.startswith("data:image"):
        # Image data is in base64 format
        _, image_data = image_data.split(",", 1)
        image_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(image_bytes))
    else:
        # Image data is a file path
        if prefix is None:
            path = image_data
        else:
            path = os.path.join(prefix, image_data)
        try:
            pil_img = Image.open(path)
        except:
            print(f'error open path {path}')
            pil_img = Image.new(mode="RGB", size=(512, 512))
    pil_img = pil_img.convert("RGB")
    return pil_img
def load_pil_images(images: List[str], prefix=None) -> List[Image.Image]:
    """

    Support file path or base64 images.

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []
    for image_data in images:
        pil_img = load_pil_image(image_data, prefix)
        pil_images.append(pil_img)

    return pil_images


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            # pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            pil_img2rgb(load_pil_image(row["image_list"][start_idx], prefix=IMAGE_DATA_PREFIX)),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += row["instruction_list"][idx-1] + ". "
            # print(f'test yiled edit ds {instruction}')
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                # pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                pil_img2rgb(load_pil_image(row["image_list"][end_idx], prefix=IMAGE_DATA_PREFIX)),
                need_loss=True,
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = row["instruction_list"][idx-1]
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data,
                        pil_img2rgb(load_pil_image(row["image_list"][idx], prefix=IMAGE_DATA_PREFIX)),
                        # pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data,
                        # pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        pil_img2rgb(load_pil_image(row["image_list"][idx], prefix=IMAGE_DATA_PREFIX)),
                        need_loss=True,
                        need_vae=False, 
                        need_vit=False,
                    )
        return data
