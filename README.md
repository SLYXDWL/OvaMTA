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
MTANet: https://github.com/yatingling/MTANet
