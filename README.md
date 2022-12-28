# DLCV Final Project - 3D Indoor Scene Long Tail Segmentation

<div align="center">
    <img src="docs/image-grid.png" width = 100% >
</div>

ScanNet200 is a 200-classed 3D semantic segmentation benchmark, 
whose number of categories is more than other benchmarks. 
The distribution of classes in this benchmark is very imbalanced. 
Since the mIoU and precision of previous work on ScanNet and 
SemanticKITTI perform poorly on the tail categories of ScanNet200, 
we follow the work of the Language-Grounded 3D Semantic Segmentation model.
 This model leverages pre-trained text embeddings from CLIP to enhance the
  robustness of feature extraction from 3D images and adopts the 
  weighted Focal loss to relieve the imbalance of data distribution.

## Reproduce Guideline

#### Build Enviroment

The following os and packages are required to reproduce this code. Nake sure
that you have correct version of packages and os before build the whole
environment.

- **ubuntu == 20.04**
- **conda == 22.11.1**

If these two os and package have been installed, you can run the following
command to cuild the whole environment.

```sh
  # create conda env
  sudo apt install build-essential python3-dev libopenblas-dev
  conda env create -f config/lg_semseg.yml
  conda activate lg_semseg

  # to install MinkowskiEngine
  pip install torch ninja
  wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
  sudo sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override
  export CUDA_HOME=/usr/local/cuda-11.1
  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas=openblas"

  # IF YOUR ENVIROMENT DOESN'T HAVE nvidia-driver-510, YOU HAVE TO RUN THE FOLLOWING COMMANDS.
  sudo apt-get install nvidia-driver-510
  sudo reboot
  cd third_party/pointnet2 && python setup.py install (TO BE CHECKED.)
```

#### Pretraining Command

```sh
  source 
```

#### Downstream Task Finetuning Command

```sh
  source
```

#### Inference Command

```sh
  source
```

## Reference

 - [LanguageGroundedSemseg](https://github.com/RozDavid/LanguageGroundedSemseg)

## Authors

- []()
