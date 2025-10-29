# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/some/path/SEED-Data-Edit-Part2-3/bagel_t2i_parquet', # path of the parquet files
            'num_files': 40, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 45670+28408, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/some/path/SEED-Data-Edit-Part2-3/bagel_parquet',
            'num_files': 8,
            'num_total_samples': 21382 + 51610,
            "parquet_info_path": '/some/path/SEED-Data-Edit-Part2-3/bagel_parquet_info.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}