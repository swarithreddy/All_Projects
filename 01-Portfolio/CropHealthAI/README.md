# CropHealthAI

A crop-leaf image classifier built with **PyTorch** and a pretrained **Swin Transformer**. The project includes a Streamlit interface, a training pipeline, and a command-line prediction script.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Run the App](#run-the-app)
- [Train the Model](#train-the-model)
- [Predict from the Command Line](#predict-from-the-command-line)
- [How It Works](#how-it-works)
- [Limitations](#limitations)

## Overview

Upload a leaf image to receive the predicted PlantVillage class and the model's confidence score. The app also displays a short description and suggested treatment for a small set of explicitly configured classes.

The application loads class names from the dataset directory and loads weights from `saved_models/best_model.pth`. Both must be present and compatible to run inference.

## Features

- Leaf image classification through a Streamlit page
- Swin Transformer model created with `timm`
- Training, validation, and test data loaders
- Confidence score from the model's softmax output
- Optional class-specific descriptions and treatment text
- CPU or CUDA device selection

## Technology Stack

| Area | Technology |
|------|------------|
| Language | Python |
| Machine learning | PyTorch, torchvision, `timm` |
| Image processing | Pillow |
| Web interface | Streamlit |
| Dataset format | ImageFolder directory layout |

## Project Structure

```text
CropHealthAI/
├── app.py                         # Streamlit application
├── requirements.txt               # Currently empty; dependencies are listed below
├── saved_models/
│   └── best_model.pth             # Saved model weights
└── src/
    ├── config.py                  # Dataset path and training settings
    ├── dataset.py                 # ImageFolder and data loaders
    ├── model.py                   # Swin Transformer classifier
    ├── predict.py                 # Command-line image prediction
    ├── train.py                   # Model training loop
    └── transforms.py              # Image transforms
```

## Requirements

- Python 3.9 or later
- The PlantVillage dataset arranged as described below
- The saved checkpoint for inference, or a training run to create one

The repository's `requirements.txt` is currently empty. Install the runtime packages explicitly:

```bash
python -m pip install torch torchvision timm streamlit pillow
```

For CUDA acceleration, install a PyTorch build that matches your CUDA setup using the official PyTorch installation selector.

## Dataset

The code expects the dataset at `dataset/PlantVillage`, relative to the project root, in torchvision `ImageFolder` format. Each class must have its own directory:

```text
dataset/
└── PlantVillage/
    ├── Apple___Black_rot/
    │   ├── image_001.jpg
    │   └── ...
    ├── Apple___healthy/
    │   └── ...
    └── ...
```

The dataset is not included in this repository. Keep the directory name and class folder structure consistent: class names determine the output labels and must match the class ordering used to train the checkpoint.

## Installation

From the project root, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install torch torchvision timm streamlit pillow
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install torch torchvision timm streamlit pillow
```

## Run the App

Run from the project root so the relative dataset and checkpoint paths resolve:

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal. Open it in a browser, upload a `.jpg`, `.jpeg`, or `.png` leaf image, and review the predicted class and confidence.

## Train the Model

With the dataset in place, run:

```bash
python -m src.train
```

The training script uses the Swin Tiny model `swin_tiny_patch4_window7_224`, pretrained weights, Adam, cross-entropy loss, and 20 epochs. It creates an 80/10/10 train/validation/test split and saves the best validation checkpoint to `saved_models/best_model.pth`.

## Predict from the Command Line

After the checkpoint and dataset are available:

```bash
python -m src.predict
```

Enter an image path when prompted. The script prints the predicted class and confidence.

## How It Works

```mermaid
flowchart LR
  I[Leaf image] --> T[Resize and normalize]
  T --> M[Swin Transformer]
  M --> P[Class probabilities]
  P --> R[Predicted class and confidence]
  R --> D[Optional disease information]
```

## Limitations

- This is a prototype classifier, not a diagnostic tool. Predictions may be incorrect and should not replace expert agricultural advice.
- The disease descriptions and treatment suggestions are hard-coded for a small number of labels; other classes receive no additional information.
- The dataset is not bundled, and the app loads class names from it even when a checkpoint is already present.
- `requirements.txt` is empty, so dependencies must currently be installed manually.
