# GAMMA

This repository provides source code for our paper:

## About this repository
    datasets/                   # contains train data for the project
    visual_model/                   # contains code and scripts

## 1. Install dependencies
### python 3.6
```bash
conda env create -f conda_env.yml
```
The backbone depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .


## 2. Train the model
```bash
 python train_gamma.py
```

## 3. Inference and visualization

```bash
 python test_demo.py
```