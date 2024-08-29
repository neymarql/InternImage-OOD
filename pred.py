import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from einops import rearrange
import pickle
from tqdm import tqdm

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
            feature = nn.functional.interpolate(feature, size=(target_h, target_w), mode="bilinear",
                                                align_corners=False)
            emb.append(feature)
        emb = torch.cat(emb, dim=1)
        emb = self.conv1x1(emb)
        emb = rearrange(emb, "b c h w -> (b h w) c")
        return emb

    def forward(self, feature_pyramid):
        embeddings = self.compute_2dspace_vectors(feature_pyramid)
        return embeddings


def perform_ood_detection(img_path, pred_map_path, conf_map_path, feature_dir, kmeans_models, model, folder_index):
    pred_map = Image.open(pred_map_path).convert('L')
    pred_map = pred_map.resize((2048, 1024), Image.NEAREST)
    pred_map_tensor = torch.tensor(np.array(pred_map), dtype=torch.int16)
    conf_map = np.array(Image.open(conf_map_path))
    conf_map_resized = cv2.resize(conf_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)
    conf_map_np = conf_map_resized.astype(np.float32) / 65535.0
    conf_map_tensor = torch.tensor(conf_map_np)

    feature_basename = os.path.basename(img_path)
    feature_folder = os.path.join(feature_dir, str(folder_index))
    feature_path = os.path.join(feature_folder, feature_basename.replace(".png", "_features.pth"))

    feature_pyramid = torch.load(feature_path, map_location='cpu', pickle_module=pickle)

    embeddings = model([feature_pyramid[f'feature_level_{i}'] for i in range(4)])

    conf_threshold = 0.5
    low_conf_mask = conf_map_tensor < conf_threshold

    low_conf_mask_flat = low_conf_mask.view(-1)
    embeddings_flat = embeddings.view(-1, cfg['EMB_SIZE'])

    low_conf_pred_map = pred_map_tensor[low_conf_mask]
    low_conf_embeddings = embeddings_flat[low_conf_mask_flat]

    updated_conf_map = conf_map_tensor.clone()
    num_classes = len(kmeans_models)

    for class_id in range(num_classes):
        class_mask = (low_conf_pred_map == class_id).view(-1)
        low_conf_class_embeddings = low_conf_embeddings[class_mask]
        
        if low_conf_class_embeddings.shape[0] > 0:
            dists = kmeans_models[class_id].transform(low_conf_class_embeddings.cpu().detach().numpy())
            min_dists = np.min(dists, axis=1)
            ood_mask = torch.tensor(min_dists > dists.mean() * 1.1, dtype=torch.bool)

            updated_conf_map = updated_conf_map.float()

            low_conf_indices = torch.nonzero(low_conf_mask.view(-1), as_tuple=True)[0]
            class_indices = torch.nonzero(class_mask, as_tuple=True)[0]
            ood_indices = class_indices[ood_mask]
            updated_conf_map.view(-1)[low_conf_indices[ood_indices]] = 0.0001
            
    return updated_conf_map


def process_all_images(root_dir, eval_dir, feature_dir, kmeans_models, model, mapped_dirs):
    all_images = []
    selected_dirs = ['bravo_synobjs/armchair', 'bravo_synobjs/bathtub', 'bravo_synobjs/billboard', 'bravo_synobjs/cheetah',
    'bravo_synobjs/elephant', 'bravo_synobjs/baby', 'bravo_synobjs/bench', 'bravo_synobjs/box', 
    'bravo_synobjs/chimpanzee', 'bravo_synobjs/flamingo', 'bravo_synobjs/giraffe', 'bravo_synobjs/hippopotamus', 
    'bravo_synobjs/koala', 'bravo_synobjs/lion', 'bravo_synobjs/panda', 'bravo_synobjs/gorilla', 
    'bravo_synobjs/kangaroo', 'bravo_synobjs/penguin', 'bravo_synobjs/plant', 'bravo_synobjs/polar bear', 
    'bravo_synobjs/sofa', 'bravo_synobjs/tiger', 'bravo_synobjs/vase', 'bravo_synobjs/table', 
    'bravo_synobjs/toilet', 'bravo_synobjs/zebra', 'bravo_SMIYC']

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(img_path, root_dir)

                for selected_dir in selected_dirs:
                    if relative_path.startswith(selected_dir):
                        all_images.append(img_path)
                        break

    print(f"Total number of images found in SMIYC and synobjects: {len(all_images)}")

    for img_path in tqdm(all_images, desc="Processing Images"):
        relative_path = os.path.relpath(img_path, root_dir)

        folder_name = relative_path.split(os.sep)[0:2]
        if "bravo_synobjs" == folder_name[0]:
            folder_index = mapped_dirs.index(f'{folder_name[0]}/{folder_name[1]}')
        else:
            folder_name = relative_path.split(os.sep)[0]
            folder_index = mapped_dirs.index(folder_name)

        if "bravo_synobjs" in relative_path:
            relative_eval_path = os.path.join(relative_path.split(os.sep)[0], *relative_path.split(os.sep)[2:])
        else:
            relative_eval_path = relative_path

        pred_map_path = os.path.join(eval_dir, relative_eval_path.replace(".png", "_pred.png"))
        conf_map_path = os.path.join(eval_dir, relative_eval_path.replace(".png", "_conf.png"))

        if os.path.exists(pred_map_path) and os.path.exists(conf_map_path):
            updated_conf_map = perform_ood_detection(img_path, pred_map_path, conf_map_path, feature_dir, kmeans_models, model, folder_index)

            updated_conf_map_img = (updated_conf_map.cpu().numpy() * 65535).astype(np.uint16)
            Image.fromarray(updated_conf_map_img).save(conf_map_path)
        else:
            print(f"Missing pred or conf files for: {img_path}")


cfg = {
    'EMB_SIZE': 1536,
}

root_dir = './data/bravodataset'
eval_dir = './bravo_eval'
feature_dir = './bravo_eval_feature'

mapped_dirs = [
    'bravo_synobjs/armchair', 'bravo_synobjs/bathtub', 'bravo_synobjs/billboard', 'bravo_synobjs/cheetah',
    'bravo_synobjs/elephant', 'bravo_synobjs/baby', 'bravo_synobjs/bench', 'bravo_synobjs/box', 
    'bravo_synobjs/chimpanzee', 'bravo_synobjs/flamingo', 'bravo_synobjs/giraffe', 'bravo_synobjs/hippopotamus', 
    'bravo_synobjs/koala', 'bravo_synobjs/lion', 'bravo_synobjs/panda', 'bravo_synobjs/gorilla', 
    'bravo_synobjs/kangaroo', 'bravo_synobjs/penguin', 'bravo_synobjs/plant', 'bravo_synobjs/polar bear', 
    'bravo_synobjs/sofa', 'bravo_synobjs/tiger', 'bravo_synobjs/vase', 'bravo_synobjs/table', 
    'bravo_synobjs/toilet', 'bravo_synobjs/zebra', 'bravo_ACDC/fog/test', 'bravo_ACDC/night/test', 
    'bravo_ACDC/rain/test', 'bravo_ACDC/snow/test', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synflare', 
    'bravo_synrain'
]

kmeans_models = []
for class_id in range(19):
    kmeans_models.append(torch.load(f'./kmeans_class_{class_id}.pth'))

model = GROODNetKNMSoftMultiClass(cfg)

process_all_images(root_dir, eval_dir, feature_dir, kmeans_models, model, mapped_dirs)
