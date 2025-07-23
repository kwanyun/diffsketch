##  DiffSketch: Stable Diffusion Feature Extraction for Sketching with One Example

<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->

<img width="800" height="auto" alt="Graphical_Abstract" src="https://github.com/user-attachments/assets/d442a3c7-1e96-43de-9fe5-cc30ee29e8e2" />

## :gear: Install Environment via Anaconda (Recommended)
    conda env create -f environment.yaml
    conda activate diffsketch

## download [weight file](https://drive.google.com/file/d/1zcjfofywsSB6zGVbZAPErA-ngGcrwE12/view?usp=drive_link) and put it on ./weight
## we also require sdv1.4 download from huggingface
    wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt



## :fast_forward: How to train diffsketch
    python train_diffsketch.py

    #change extractor to use different styles for example:
    python train_diffsketch.py --extractor xdog

    #change the config file to change the source image:
    python train_diffsketch.py --config configs/train_config/7-mountain.yaml


We tested our code on single Nvidia V100 GPU (32GB VRAM). 
