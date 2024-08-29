#coding=gbk
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from einops import rearrange
import random


class GROODNetKNMSoftMultiClass(nn.Module):
    def __init__(self, cfg, num_classes=19):
        super(GROODNetKNMSoftMultiClass, self).__init__()
        self.NUM_CLASSES = num_classes
        self.EMB_SIZE = cfg['EMB_SIZE']
        self.conv1x1 = nn.Conv2d(in_channels=sum([192, 384, 768, 1536]), out_channels=self.EMB_SIZE, kernel_size=1)

    def compute_2dspace_vectors(self, out):
        target_h, target_w = 1024, 2048
        emb = []
        for feature in out:
            feature = nn.functional.interpolate(feature, size=(target_h, target_w), mode="bilinear", align_corners=False)
            emb.append(feature)
        emb = torch.cat(emb, dim=1)
        emb = self.conv1x1(emb)
        emb = rearrange(emb, "b c h w -> (b h w) c")
        return emb

    def forward(self, feature_pyramid):
        embeddings = self.compute_2dspace_vectors(feature_pyramid)
        return embeddings


def load_cityscapes_data(data_root, label_dirs, img_dirs):
    all_image_paths = []
    all_label_paths = []
    
    for label_dir, img_dir in zip(label_dirs, img_dirs):
        cities = os.listdir(os.path.join(data_root, img_dir))
    
        for city in cities:
            city_img_dir = os.path.join(data_root, img_dir, city)
            city_label_dir = os.path.join(data_root, label_dir, city)
            for img_file in os.listdir(city_img_dir):
                if img_file.endswith("_leftImg8bit.png"):
                    label_file = img_file.replace("_leftImg8bit.png", "_leftImg8bit_gtFine_labelTrainIds.png")
                    all_image_paths.append(os.path.join(city_img_dir, img_file))
                    all_label_paths.append(os.path.join(city_label_dir, label_file))
    
    return all_image_paths, all_label_paths


def train_kmeans_on_cityscapes(data_root, cfg, feature_dir, num_classes=19, num_clusters_per_class=30, num_samples=30000):
    model = GROODNetKNMSoftMultiClass(cfg)
    kmeans_models = [KMeans(n_clusters=num_clusters_per_class, n_init='auto') for _ in range(num_classes)]

    label_dirs = ['gtFine/train','gtFine/val','gtCoarse/train_extra']
    img_dirs = ['leftImg8bit/train','leftImg8bit/val','leftImg8bit/train_extra']
    image_paths, label_paths = load_cityscapes_data(data_root, label_dirs, img_dirs)

    sampled_indices = random.sample(range(len(image_paths)), num_samples)
    sampled_image_paths = [image_paths[i] for i in sampled_indices]
    sampled_label_paths = [label_paths[i] for i in sampled_indices]

    for img_path, label_path in tqdm(zip(sampled_image_paths, sampled_label_paths), total=len(sampled_image_paths)):
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        preprocess = transforms.Compose([
            transforms.Resize((1024, 2048)),
            transforms.ToTensor(),
        ])
        
        img_tensor = preprocess(img).unsqueeze(0)  # [1, 3, H, W]
        label_tensor = torch.tensor(np.array(label), dtype=torch.int64)  # [H, W]

        feature_basename = os.path.basename(img_path).replace("_leftImg8bit.png", "_leftImg8bit_features.pth")
        feature_path = os.path.join(feature_dir, feature_basename)
        feature_pyramid = torch.load(feature_path, map_location='cpu')

        embeddings = model([feature_pyramid[f'feature_level_{i}'] for i in range(4)])

        for class_id in range(num_classes):
            class_mask = (label_tensor == class_id).view(-1)
            class_embeddings = embeddings[class_mask]
            
            if class_embeddings.shape[0] > 0:
                embeddings_np = class_embeddings.detach().numpy()
                kmeans_models[class_id].fit(embeddings_np)

    
    for class_id in range(num_classes):
        torch.save(kmeans_models[class_id], f'kmeans_class_{class_id}.pth')
    
    return kmeans_models


cfg = {
    'EMB_SIZE': 1536,
}


data_root = './data/cityscapes'
feature_dir = './feature'


print("Start training KMeans...")
kmeans_models = train_kmeans_on_cityscapes(data_root, cfg, feature_dir)
