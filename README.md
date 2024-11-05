# D3-Stereo

This is the official repo for our work 'These Maps Are Made by Propagation: Adapting Deep Stereo Networks to Road Scenarios with Decisive Disparity Diffusion'.  

## Setup

We built and ran the repo with CUDA 11.8, Python 3.9.0, and Pytorch 2.1.0. For using this repo, please follow the instructions below:

```
pip install -r requirements.txt
```

## Pre-trained models

Pre-trained weights can be downloaded from [google drive](https://drive.google.com/file/d/1K9Hx-IGTWNTgWFemy_maz_dNiNNZf5B4/view?usp=sharing), and is supposed to be under dir: `toolkit/models/`.

The required VGG weights "vgg16-397923af.pth" can be found at [Graft](https://github.com/SpadeLiu/Graft-PSMNet), and should be under dir: `toolkit/models/graft`.

## Dataset Preparation

The created UDTIRI-Stereo dataset can be accessed form [google drive](https://pan.baidu.com/s/1mkmKGwgrvo0qT7W1xU3eXA?pwd=jxcn)

The used Stereo-Road dataset can be accessed from [google drive](https://drive.google.com/file/d/1s7wKvPNzPVTNQXIZRP5jZDOsTvUkqOls/view?usp=sharing)

Our folder structure is as follows:

```
├── datasets
    ├── real_road
    │   ├── dataset1
    │   ├── dataset2
    │   └── dataset3
    ├── Virtual_road
    │   ├── Tiled_V2
    │   └── WetRoad1
```

## Training & Evaluation

For using and evaluating our D3Stereo, please run the files  `toolkit/main.py` and ``toolkit/tool.py``, respectively. 
