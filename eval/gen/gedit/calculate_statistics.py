# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import megfile
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math

GROUPS = [
    "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
]

model_name = 'bagel'

skip_keys = set([
"365da3516f60dde11e8a362ceffceb38",
"680b385ed8595ff5e4bfdd9fc7c5bf1a",
"be45c39b3bcc9d082051c13b5300dde1",
"947855a8a1e0954deaa4dc826e9519db",
"3a9853285c981f9ec42fae7c9ba938f8",
"bdd77c99d54bdb14bcd48ee0ee3faafa",
"1bb0fbeaac87f6eff80e09d8fd409de1",
"5b8717b2209b784940f388864d5520f3",
"7eb55372802dfee7167a63e02728ca0e",
"0ca8f7be9422a84221920fccc6df2c4c",
"3a016977fd14367ffc324d12e965e961",
"8382cb3ebe2cfce4315f9ec944ee12c2",
"a095ed1661fa125363bf59a47d8e52e4",
"07c987a0a42790a9f5fed28a2bc6409e",
"75ba6223b2ab4de7f35cc0653d1df7da",
"f4f263ee11d7db0e483e1c10973e7b22",
"08fcf0e92aeea7e37931a6036a27174b",
"8ed283fe0c51659c06fd1de14420b544",
"db250fc8f05da99e1e9e57eb3ecd3920",
"a2a41ebb84be1126248d9cf65d8ed078",
"11f41f16aebac2a07f24432b8cbeefd7",
"4165fe6296cc4e75c9055794e5a13a10",
"b485856ffb2107eaf50b4b437dacbea3",
"90c04ff64d248a824f0cd65936a99bf0",
"ac11359a70c3950fce81d6c4a9665875",
"8155a4b0727c1a4176f5f70cc0810562",
"058a3b3c422dcefddb8deb2c61ded83d",
"58cf4f4a16cd16ffef55c170804be136",
])

def analyze_scores(save_path_dir, evaluate_group, language, preset_group_name=None):
    results = defaultdict(dict)
    save_path_new = save_path_dir
    model_total_score = defaultdict(dict)

    group_dict_sub = {}
    group_scores_semantics = defaultdict(lambda: defaultdict(list))
    group_scores_quality = defaultdict(lambda: defaultdict(list))
    group_scores_overall = defaultdict(lambda: defaultdict(list))

    group_scores_semantics_intersection = defaultdict(lambda: defaultdict(list))
    group_scores_quality_intersection = defaultdict(lambda: defaultdict(list))
    group_scores_overall_intersection = defaultdict(lambda: defaultdict(list))
    length_total = 0
    save_path_dir_raw = save_path_dir
    
    for group_name in GROUPS if preset_group_name is None else [preset_group_name]:

        csv_path = os.path.join(save_path_new, f"{evaluate_group[0]}_{group_name}_gpt_score.csv")
        csv_file = megfile.smart_open(csv_path)
        df = pd.read_csv(csv_file)
        
        filtered_semantics_scores = []
        filtered_quality_scores = []
        filtered_overall_scores = []
        filtered_semantics_scores_intersection = []
        filtered_quality_scores_intersection = []
        filtered_overall_scores_intersection = []
        
        for _, row in df.iterrows():
            source_image = row['source_image']
            edited_image = row['edited_image']
            instruction = row['instruction']
            semantics_score = row['sementics_score']
            quality_score = row['quality_score']
            intersection_exist = row['intersection_exist']
            instruction_language = row['instruction_language']

            key = os.path.basename(edited_image).split('.')[0]

            if key in skip_keys:
                continue

            if instruction_language == language:
                pass
            else:
                continue
            
            overall_score = math.sqrt(semantics_score * quality_score)
            
            filtered_semantics_scores.append(semantics_score)
            filtered_quality_scores.append(quality_score)
            filtered_overall_scores.append(overall_score)
            if intersection_exist:
                filtered_semantics_scores_intersection.append(semantics_score)
                filtered_quality_scores_intersection.append(quality_score)
                filtered_overall_scores_intersection.append(overall_score)
        
        avg_semantics_score = np.mean(filtered_semantics_scores)
        avg_quality_score = np.mean(filtered_quality_scores)
        avg_overall_score = np.mean(filtered_overall_scores)
        group_scores_semantics[evaluate_group[0]][group_name] = avg_semantics_score
        group_scores_quality[evaluate_group[0]][group_name] = avg_quality_score
        group_scores_overall[evaluate_group[0]][group_name] = avg_overall_score

        avg_semantics_score_intersection = np.mean(filtered_semantics_scores_intersection)
        avg_quality_score_intersection = np.mean(filtered_quality_scores_intersection)
        avg_overall_score_intersection = np.mean(filtered_overall_scores_intersection)
        group_scores_semantics_intersection[evaluate_group[0]][group_name] = avg_semantics_score_intersection
        group_scores_quality_intersection[evaluate_group[0]][group_name] = avg_quality_score_intersection
        group_scores_overall_intersection[evaluate_group[0]][group_name] = avg_overall_score_intersection


    # print("\n--- Overall Model Averages ---")

    # print("\nSemantics:")
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics[model_name][group] for group in (GROUPS if preset_group_name is None else [preset_group_name])]
        model_avg = np.mean(model_scores)
        group_scores_semantics[model_name]["avg_semantics"] = model_avg

    # print("\nSemantics Intersection:")
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics_intersection[model_name][group] for group in (GROUPS if preset_group_name is None else [preset_group_name])]
        model_avg = np.mean(model_scores)
        group_scores_semantics_intersection[model_name]["avg_semantics"] = model_avg
    
    # print("\nQuality:")
    for model_name in evaluate_group:
        model_scores = [group_scores_quality[model_name][group] for group in (GROUPS if preset_group_name is None else [preset_group_name])]
        model_avg = np.mean(model_scores)
        group_scores_quality[model_name]["avg_quality"] = model_avg

    # print("\nQuality Intersection:")
    for model_name in evaluate_group:
        model_scores = [group_scores_quality_intersection[model_name][group] for group in (GROUPS if preset_group_name is None else [preset_group_name])]
        model_avg = np.mean(model_scores)
        group_scores_quality_intersection[model_name]["avg_quality"] = model_avg

    # print("\nOverall:")
    for model_name in evaluate_group:
        model_scores = [group_scores_overall[model_name][group] for group in (GROUPS if preset_group_name is None else [preset_group_name])]
        model_avg = np.mean(model_scores)
        group_scores_overall[model_name]["avg_overall"] = model_avg

    # print("\nOverall Intersection:")
    for model_name in evaluate_group:
        model_scores = [group_scores_overall_intersection[model_name][group] for group in (GROUPS if preset_group_name is None else [preset_group_name])]
        model_avg = np.mean(model_scores)
        group_scores_overall_intersection[model_name]["avg_overall"] = model_avg

    return group_scores_semantics, group_scores_quality, group_scores_overall, group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["gpt4o", "qwen25vl"])
    parser.add_argument("--language", type=str, default="en", choices=["en", "cn"])
    parser.add_argument("--group_name", type=str, default=None, help="Specify a group name to evaluate, or leave as None to evaluate all groups.")
    args = parser.parse_args()
    save_path_dir = args.save_path
    evaluate_group = [model_name]
    backbone = args.backbone

    save_path_new = os.path.join(save_path_dir, backbone, "eval_results_new")

    print("\nOverall:")
   
    for model_name in evaluate_group:
        group_scores_semantics, group_scores_quality, group_scores_overall, group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection = analyze_scores(save_path_new, [model_name], language=args.language, preset_group_name=args.group_name)
    for group_name in GROUPS if args.group_name is None else [args.group_name]:
        print(f"{group_name}: {group_scores_semantics[model_name][group_name]:.3f}, {group_scores_quality[model_name][group_name]:.3f}, {group_scores_overall[model_name][group_name]:.3f}")

    print(f"Average: {group_scores_semantics[model_name]['avg_semantics']:.3f}\t{group_scores_quality[model_name]['avg_quality']:.3f}\t{group_scores_overall[model_name]['avg_overall']:.3f}")

    # print("\nIntersection:")

    # for group_name in GROUPS:
    #     print(f"{group_name}: {group_scores_semantics_intersection[model_name][group_name]:.3f}, {group_scores_quality_intersection[model_name][group_name]:.3f}, {group_scores_overall_intersection[model_name][group_name]:.3f}")

    # print(f"Average Intersection: {group_scores_semantics_intersection[model_name]['avg_semantics']:.3f}, {group_scores_quality_intersection[model_name]['avg_quality']:.3f}, {group_scores_overall_intersection[model_name]['avg_overall']:.3f}")
