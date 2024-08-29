# InternImage-OOD

This repository contains the implementation of the InternImage-OOD framework, which integrates the InternImage model with a KMeans-based Out-of-Distribution (OOD) detection mechanism. The method achieved top performance in the BRAVO Challenge, with a notable Bravo Index of 0.6263.

## Overview

InternImage-OOD is a cutting-edge framework designed to enhance OOD detection by combining powerful image features with clustering techniques. This approach leverages the strengths of InternImage for feature extraction, followed by KMeans clustering to detect and refine OOD regions in the input images.

## Methodology

The core methodology is based on the following steps:
1. **Feature Extraction**: The InternImage model extracts feature embeddings (\( F \)) from input images, represented as:
    \[
    F = \{ f_k \in \mathbb{R}^{D \times H \times W} \}_{k=1}^{M} = \text{FeatureEmbedding}(I)
    \]
    where \( I \) is the input image, \( D \) is the feature dimension, and \( H \times W \) is the spatial resolution.

2. **OOD Detection**: KMeans clustering is applied to these feature embeddings to detect OOD regions:
    - \( P \) denotes the predicted class map, where each pixel is assigned a label by the InternImage model.
    - \( C \) is the confidence map, indicating the model's confidence in each prediction.
    - \( F \) represents the feature embeddings used for further refinement.

    After clustering, the OOD mask is generated, and the confidence map is updated to reflect low confidence (e.g., 0.0001) in detected OOD regions.

## Results

Considering the time constraints of the competition and to demonstrate the effectiveness of our method on smaller datasets, we used only the Mapillary Vistas and Cityscapes datasets for this experiment. Despite the limited data, InternImage-OOD achieved the top spot in the BRAVO Challenge, with a Bravo Index of 0.6263.

However, it is worth noting that while the overall performance was strong, there is still room for improvement in OOD detection. Future work will focus on advancing this area.

| Method             | Bravo Index | Semantic | OOD    |
|--------------------|-------------|----------|--------|
| InternImage-OOD    | **0.6263**  | 0.6934   | 0.5710 |
| Ablation (no KMeans)| 0.6208     | 0.6933   | 0.5621 |

Our method achieved the following results in the BRAVO Challenge:

## Example Usage

1. **Setup**: 
   - Ensure you have the required dependencies installed.
   - Clone this repository and navigate to the project directory.

2. **Feature Extraction**:
   - Run the feature extraction process on your dataset.
   - Save the extracted features in the specified directory structure.

3. **Training KMeans**:
   - Train the KMeans model on the extracted features.

4. **OOD Detection**:
   - Use the trained KMeans model to perform OOD detection on your test images.
   - Update the confidence map based on the detected OOD regions.

## Citation

If you find this project helpful in your research, please consider citing:

@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}

@inproceedings{zhu2022uni,
  title={Uni-perceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks},
  author={Zhu, Xizhou and Zhu, Jinguo and Li, Hao and Wu, Xiaoshi and Li, Hongsheng and Wang, Xiaohua and Dai, Jifeng},
  booktitle={CVPR},
  pages={16804--16815},
  year={2022}
}

@article{zhu2022uni,
  title={Uni-perceiver-moe: Learning sparse generalist models with conditional moes},
  author={Zhu, Jinguo and Zhu, Xizhou and Wang, Wenhai and Wang, Xiaohua and Li, Hongsheng and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2206.04674},
  year={2022}
}

@article{li2022uni,
  title={Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks},
  author={Li, Hao and Zhu, Jinguo and Jiang, Xiaohu and Zhu, Xizhou and Li, Hongsheng and Yuan, Chun and Wang, Xiaohua and Qiao, Yu and Wang, Xiaogang and Wang, Wenhai and others},
  journal={arXiv preprint arXiv:2211.09808},
  year={2022}
}

@article{yang2022bevformer,
  title={BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision},
  author={Yang, Chenyu and Chen, Yuntao and Tian, Hao and Tao, Chenxin and Zhu, Xizhou and Zhang, Zhaoxiang and Huang, Gao and Li, Hongyang and Qiao, Yu and Lu, Lewei and others},
  journal={arXiv preprint arXiv:2211.10439},
  year={2022}
}

@article{su2022towards,
  title={Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information},
  author={Su, Weijie and Zhu, Xizhou and Tao, Chenxin and Lu, Lewei and Li, Bin and Huang, Gao and Qiao, Yu and Wang, Xiaogang and Zhou, Jie and Dai, Jifeng},
  journal={arXiv preprint arXiv:2211.09807},
  year={2022}
}

@inproceedings{li2022bevformer,
  title={Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={ECCV},
  pages={1--18},
  year={2022},
}
