# Histopathy-ViT

**Histopathy-ViT** is a PyTorch-based implementation of Vision Transformer (ViT) models for histopathology image classification. This repository provides code, pretrained weights, and scripts to train and evaluate ViT models on histopathology datasets.

## Features
- Implementation of Vision Transformer (ViT) tailored for histopathology image analysis
- Training and evaluation scripts with data preprocessing
- Support for multiple ViT architectures and hyperparameter tuning
- Visualization tools for attention maps and model interpretability

## Installation

Clone the repository:

    git clone https://github.com/qyncist/Histopathy-ViT.git
    cd Histopathy-ViT

Create a Python environment (recommended with conda or venv):

    conda create -n histovit python=3.8 -y
    conda activate histovit

Install the required packages:

    pip install -r requirements.txt

## Usage

### Training

To train a Vision Transformer model on your histopathology dataset, run:

    python train.py --config configs/train_config.yaml

### Evaluation

To evaluate a pretrained model, run:

    python evaluate.py --model_path path/to/model.pth --data_path path/to/dataset

### Visualization

You can visualize attention maps or other model outputs using:

    python visualize.py --model_path path/to/model.pth --image_path path/to/image.png
