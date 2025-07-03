# OvaMTA
# Introduction
OvaMTA is an integrated system for the detection, segmentation, and diagnosis of ovaries and ovarian masses.

# Environment
* Python 3.10.12
* PyTorch 1.13.1+cu117
* CUDA 11.7

# Installation
```bash
pip install requirement.txt
```

# Data Preparation

# Training OvaMTA-Seg Model 
```bash
python train_OvaMTA-Seg.py
```

# Training OvaMTA-Diagnosis Model 
```bash
python train_OvaMTA-Diagnosis.py
```

# Testing
```bash
python test_video.py --video_path <path of test video>
```

# Reference
1.	Wenli Dai, Yingnan Wu, Yating Ling, Jing Zhao, Shuang Zhang, Zhaowen Gu, Liping Gong, Manning Zhu, Shuang Dong, Songcheng Xu, Lei Wu, Litao Sun, Dexing Kong. Development and validation of a deep learning pipeline to diagnose ovarian masses using ultrasound screening: a retrospective multicenter study[J]. eClinicalMedicine, 2024: 78, 102923.
2.	Yating Ling, Yuling Wang, Wenli Dai, Jie Yu, Ping Liang, Dexing Kong. MTANet: Multi-task attention network for automatic medical image segmentation and classification[J]. IEEE Trans Med Imaging. 2024 Feb, 43(2): 674-685.

# Note
MTANet: https://github.com/yatingling/MTANet
