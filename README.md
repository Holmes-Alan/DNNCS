# DNNCS (Downscaling Neural Network for Coastal Simulations)

Downscaling Neural Network for Coastal Simulations (DNNCS) that leverages advanced neural networks for coastal downscaling.

By Zhi-Song Liu, Markus Buttner, Vadym Aizinger, Andreas Rupp

This repo provides training, simple testing codes and pretrained models.

Please check our [paper](https://arxiv.org/abs/2408.16553)).

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
## Download Model Files
### 1. Pre-trained Models
You can download the pretrained model from this [link]()
