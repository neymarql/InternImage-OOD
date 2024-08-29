#coding=gbk
from PIL import Image
import numpy as np
import os

def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]

def convert_to_grayscale(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img_np = np.array(img)

    palette = cityscapes_palette()

    gray_img = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

    for i, color in enumerate(palette):
        mask = np.all(img_np == color, axis=-1)
        gray_img[mask] = i

    gray_img_pil = Image.fromarray(gray_img, mode='L')
    gray_img_pil.save(image_path)

def process_directory(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith("_pred.png"):
                input_path = os.path.join(root, filename)
                convert_to_grayscale(input_path)
                print(f"Processed and replaced {filename}")

input_directory = '/mnt/qianlong/InternImage/segmentation/bravo_eval/'

process_directory(input_directory)
