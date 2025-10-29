import argparse
import glob
import os.path

import torch
from PIL import Image
from torchvision import transforms
from piq import brisque
import tqdm

def calc_score(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),  # Converts to [0, 1] and channels-first (C, H, W)
    ])

    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Compute NIQE
    with torch.no_grad():
        score = brisque(img_tensor)
        # print(f"brisque Score: {score.item():.2f}")
        return score.cpu().item()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    img_path = args.path
    score = 0.0
    cnt = 0
    files = glob.glob(os.path.join(img_path,  '**/*.png'), recursive=True)
    files = files + glob.glob(os.path.join(img_path, '**/*.webp'), recursive=True)

    care_files = []
    for filename in files:
        basename = os.path.basename(filename)
        if 'SRCIMG' not in basename and 'MASK' not in basename:
            care_files.append(filename)
    for care_file in tqdm.tqdm(care_files):
        ret = calc_score(care_file)
        score += ret
        cnt += 1
    print(f'get {img_path} brisque score: {score/cnt}')