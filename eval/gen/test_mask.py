import glob
import random
from os.path import basename

from PIL import Image
import numpy as np


def get_mask_rect(img):
    width, height = img.size
    max_y = 0
    max_x = 0
    min_y = height
    min_x = width
    for y in range(height):
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            # print(f'{r} {g} {b}')
            if r== 0 and g== 0 and b == 0 and a == 0:
                if y > max_y:
                    max_y = y
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if x < min_x:
                    min_x = x
    # print(f'[{min_x}, {max_x}], [{min_y}, {max_y}]')
    # mask_img = img.crop((min_x, min_y, max_x, max_y))
    # mask_img.show()
    return min_x, min_y, max_x, max_y


def get_noise(img: Image):
    width, height = img.size
    noise_gray = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    img_gray = Image.fromarray(noise_gray, mode='L').convert("RGBA")
    img_ret = Image.blend(img, img_gray, alpha=0.7)
    img_ret.show()
    img_ret.save('noise_blend.png')

def test_crop(img: Image, outpath):
    width, height = img.size
    print(f'img size {img.size}')
    img_ret = img.crop((0.7 * width//4, 0, width * 3.5 //4, height // 2))
    # img_ret.show()
    img_ret.save(outpath, format='PNG')