# ML Cropper
ml-cropper is a Python library for training an image cropping model with a few samples and batch cropping images in a directory.

# Installation
```
git clone https://github.com/royokello/ml-cropper.git
cd ml-cropper
pip install -r requirements.txt
```

# Usage

## Label

`python -m label.main -w "path to working directory"`

## Train

`python train.py -w "path to working directory" -e "epochs" -c "checkpoints"`

## Crop