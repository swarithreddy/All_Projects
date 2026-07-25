# 🌿 Crop Disease Detection using PyTorch

## Overview

This project is a deep learning-based crop disease detection system built using PyTorch and a pretrained Swin Transformer model.

Users can upload a crop leaf image, and the model predicts the disease along with a confidence score.

---

## Features

- Crop disease classification
- Transfer Learning using Swin Transformer
- Image upload using Streamlit
- Confidence score
- Disease description
- Treatment suggestions

---

## Tech Stack

- Python
- PyTorch
- timm
- Streamlit
- OpenCV
- Pillow

---

## Dataset

PlantVillage Dataset

---

## Project Structure

CropHealthAI/
│
├── dataset/
├── saved_models/
├── outputs/
├── src/
│ ├── config.py
│ ├── transforms.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── predict.py
│
├── app.py
├── requirements.txt
└── README.md

---

## Run Training

python -m src.train

---

## Run Prediction

python -m src.predict

---

## Run Web App

streamlit run app.py