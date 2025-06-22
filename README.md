# DNNCS (Downscaling Neural Network for Coastal Simulations)

Downscaling Neural Network for Coastal Simulations (DNNCS) that leverages advanced neural networks for coastal downscaling.

By Zhi-Song Liu, Markus Buttner, Vadym Aizinger, Andreas Rupp

This repo provides training, simple testing codes and pretrained models.

Please check our [paper](https://arxiv.org/abs/2408.16553).

```text
@article{liu2025dnncs,
  title={Downscaling Neural Network for Coastal Simulations},
  author={Liu, Zhi-Song and Buttner, Markus and Aizinger, Vadym and Rupp, Andreas},
  journal={arXiv preprint arXiv:2202.13562},
  year={2025}
}
```

# Requirements
- Ubuntu 20.04 (18.04 or higher)
- NVIDIA GPU

# Dependencies
- Python 3.9 (> 3.0)
- PyTorch 2.5.0 (>= 1.13.0)
- NVIDIA GPU + CUDA 12.4 (or >=11.8)

Or you may create a new virtual python environment using Conda, as follows

```shell
conda create --name DNNCS python=3.9 -y
conda activate DNNCS
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```

# Installation
```sh
$ pip install -r requirements.txt
```

# Testing
## 1. Download Pre-trained Models
You can download the pretrained model from this [link](https://drive.google.com/file/d/1yVfbYR5qYP41QaeVlEQ2dM1n93d_DEjP/view?usp=sharing), and put it under "ckpt" folder.

## 2. Download example files for testing
You can download the Bahamas and Galveston Bay data from this [link](https://drive.google.com/file/d/1CsDtIJp83UpvKr5q5VFrq60JsQtXr1wD/view?usp=sharing), and put "demo_data" under the current working directory.

## 3. Run testing
You can now run the following script.

```sh
$ CUDA_VISIBLE_DEVICES=1 python demon_test.py
```

Before running this script, you may open it to change the root directory where you store the "demo_data". The results are stored under the new created folder "Pred_CNNCS".

## 4. Some downscaled results are available
You can download the data from here: [Galveston](https://drive.google.com/file/d/1Pt6ejGfrsypD0k0vCwXg0TW_6tU2IEyH/view?usp=sharing), [Bahamas](https://drive.google.com/file/d/1p23RPrM9X8Wi8TqH8Z3x2TG2TD8DX3Xa/view?usp=sharing). 

# Training
After preparing the training data, you can now run the following script.

```sh
$ CUDA_VISIBLE_DEVICES=1 python main_MPI.py
```

The model parameter is stored under "ckpt" folder
