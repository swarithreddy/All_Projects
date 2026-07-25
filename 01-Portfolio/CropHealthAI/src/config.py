import torch

# Dataset
DATASET_PATH = "dataset/PlantVillage"

# Image Settings
IMAGE_SIZE = 224

# Training Settings
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001

# Model
MODEL_NAME = "swin_tiny_patch4_window7_224"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")