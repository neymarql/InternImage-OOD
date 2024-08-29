#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: dataaug
Author: 钱隆
Date: 2024-08-19
Email: neymarql0614@gmail.com

Modification History:
    - Date: 2024-08-19
      Author: 钱隆
"""
import os
import random
import cv2
import numpy as np
from tqdm import tqdm


original_data_dir = './data/cityscapes/'
gtFine_dir = os.path.join(original_data_dir, 'gtFine')
gtCoarse_dir = os.path.join(original_data_dir, 'gtCoarse')
leftImg8bit_dir = os.path.join(original_data_dir, 'leftImg8bit')
augmented_data_dir = './data/augmented_cityscapes/'

dirs = ['train', 'val', 'train_extra']
for split in dirs:
    gtFine_split_dir = os.path.join(gtFine_dir if split != 'train_extra' else gtCoarse_dir, split)
    leftImg8bit_split_dir = os.path.join(leftImg8bit_dir, split)

    for city in os.listdir(leftImg8bit_split_dir):
        city_gtFine_dir = os.path.join(gtFine_split_dir, city)
        city_leftImg8bit_dir = os.path.join(leftImg8bit_split_dir, city)

        os.makedirs(os.path.join(augmented_data_dir, 'gtFine', split, city), exist_ok=True)
        os.makedirs(os.path.join(augmented_data_dir, 'leftImg8bit', split, city), exist_ok=True)


def augment_with_ood(image, label_train_id, ood_image, ood_label_train_id):
    ood_mask = (ood_label_train_id == 255).astype(np.uint8)
    if np.sum(ood_mask) == 0:
        return image, label_train_id

    ood_image = cv2.resize(ood_image, (image.shape[1], image.shape[0]))
    ood_label_train_id = cv2.resize(ood_label_train_id, (label_train_id.shape[1], label_train_id.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

    augmented_image = image.copy()
    augmented_image[ood_mask == 1] = ood_image[ood_mask == 1]

    augmented_label_train_id = label_train_id.copy()
    augmented_label_train_id[ood_mask == 1] = 255

    return augmented_image, augmented_label_train_id


for split in dirs:
    gtFine_split_dir = os.path.join(gtFine_dir if split != 'train_extra' else gtCoarse_dir, split)
    leftImg8bit_split_dir = os.path.join(leftImg8bit_dir, split)

    for city in tqdm(os.listdir(leftImg8bit_split_dir)):
        city_gtFine_dir = os.path.join(gtFine_split_dir, city)
        city_leftImg8bit_dir = os.path.join(leftImg8bit_split_dir, city)

        for image_file in os.listdir(city_leftImg8bit_dir):
            img_path = os.path.join(city_leftImg8bit_dir, image_file)
            label_path = os.path.join(city_gtFine_dir, image_file.replace('leftImg8bit', 'gtFine_labelTrainIds'))

            image = cv2.imread(img_path)
            label_train_id = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            ood_split = random.choice(dirs)
            ood_city = random.choice(os.listdir(os.path.join(leftImg8bit_dir, ood_split)))
            ood_img_path = random.choice(os.listdir(os.path.join(leftImg8bit_dir, ood_split, ood_city)))

            ood_image = cv2.imread(os.path.join(leftImg8bit_dir, ood_split, ood_city, ood_img_path))
            ood_label_path = os.path.join(gtFine_dir if ood_split != 'train_extra' else gtCoarse_dir, ood_split,
                                          ood_city, ood_img_path.replace('leftImg8bit', 'gtFine_labelTrainIds'))
            ood_label_train_id = cv2.imread(ood_label_path, cv2.IMREAD_GRAYSCALE)

            augmented_image, augmented_label_train_id = augment_with_ood(image, label_train_id, ood_image,
                                                                         ood_label_train_id)

            augmented_img_path = os.path.join(augmented_data_dir, 'leftImg8bit', split, city, image_file)
            augmented_label_path = os.path.join(augmented_data_dir, 'gtFine', split, city,
                                                image_file.replace('leftImg8bit', 'gtFine_labelTrainIds'))

            cv2.imwrite(augmented_img_path, augmented_image)
            cv2.imwrite(augmented_label_path, augmented_label_train_id)

print("Completed")
