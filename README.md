# ML Cropper
ml-cropper is a Python library for training an image cropping model with a few samples and batch cropping images in a directory.

# Installation
```
git clone https://github.com/royokello/ml-cropper.git
cd ml-cropper
pip install -r requirements.txt
```

# Usage

## Data

for training 256p input images and label.json are required.

for cropping a trained model, 256p and 1024p input images are required.

```
working_dir
├── input
│   ├── 256p
│   └── 1024p
├── models
│   ├── 1722688867_s=16_e=16.pth
│   └── 1722691040_b=1722688867_s=31_e=8.pth
├── output
│   ├── 256p
│   └── 512p
├── labels.json
└── training.txt

```

## Label

`python -m label.main -w "path to working directory"`

## Train

`python train.py -w "path to working directory" -e "epochs" -c "checkpoints" -b "base model name (optional)"`

## Crop

`python crop.py -w "path to working directory"`
