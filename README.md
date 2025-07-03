# OvaMTA:Ovarian Multi-Task Attention Network
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**Official PyTorch implementation of the deep learning pipeline for ovarian mass diagnosis from ultrasound screening**  
*(Retrospective Multicenter Study | Lancet 2024)*

## 📖 Overview
OvaMTA is an automated AI system for:
1. **Ovary & mass detection** in ultrasound images/videos
2. **Semantic segmentation** of ovarian structures
3. **Pathological classification** (benign/malignant)  
Achieving performance comparable to senior radiologists (AUC: 0.911 on video data).

![OvaMTA System Pipeline](docs/pipeline.png)  
*Fig.1: OvaMTA workflow combining segmentation and diagnosis models*

## 📊 Performance Highlights
| Metric       | Internal Test | External Test (Image) | External Test (Video) |
|--------------|---------------|----------------------|----------------------|
| **Segmentation (Dice)** | 0.887 ± 0.134 | 0.819 ± 0.201 | - |
| **Detection AUC**       | 0.970 | 0.877 | 1.00 |
| **Classification AUC**  | 0.941 | 0.941 | 0.911 |

## Environment
* Python 3.10.12
* PyTorch 1.13.1+cu117
* CUDA 11.7

## ⚙️ Installation
```bash
git clone https://github.com/SLYXDWI/OvaMTA.git
cd OvaMTA

# Create conda environment
conda create -n ovamta python=3.10
conda activate ovamta

# Install dependencies
pip install -r requirements.txt
```

## 🏥 Data Preparation
Data organization: Managed via Excel files (240108-image-based-BM.xlsx) with dedicated sheets for different data splits (train/val/test). Per-sample metadata:

* Tumor image path (tumor field)

* ROI mask path (roi field)

* Binary classification label (BBM: 0=Benign, 1=Malignant)

Clinical features:

* Normalized age (Age/100)

* CA125 tumor marker value

* CA125 availability flag (1=available, 0=missing)

* Filename identifier

## 🧠 Training OvaMTA-Seg Model 
```bash
python train_OvaMTA-Seg.py
```

## 🧠 Training OvaMTA-Diagnosis Model 
```bash
python train_OvaMTA-Diagnosis.py
```

## 🔍 Inference
```bash
python test_video.py --video_path <path of test video>
```

## 📜 Citation
If you use this work, please cite:
```bash
@article{dai2024development,
  title={Development and validation of a deep learning pipeline to diagnose ovarian masses using ultrasound screening},
  author={Dai, Wen-Li and Wu, Ying-Nan and Ling, Ya-Ting and Zhao, Jing and Zhang, Shuang and Gu, Zhao-Wen and Gong, Li-Ping and Zhu, Man-Ning and Dong, Shuang and Xu, Song-Cheng and Wu, Lei and Sun, Li-Tao and Kong, De-Xing},
  journal={The Lancet},
  volume={78},
  pages={102923},
  year={2024},
  doi={10.1016/j.eclimn.2024.102923}
}
```
```bash
@article{ling2024mtanet,
  title={{MTANet}: Multi-task attention network for automatic medical image segmentation and classification},
  author={Ling, Yating and Wang, Yuling and Dai, Wenli and Yu, Jie and Liang, Ping and Kong, Dexing},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={2},
  pages={674--685},
  year={2024},
  month={Feb},
  publisher={IEEE}
}  
```

## 📄 License
This project is licensed under the CC BY-NC-ND 4.0 license.
For commercial use, please contact the authors.

# Note
MTANet: https://github.com/yatingling/MTANet
