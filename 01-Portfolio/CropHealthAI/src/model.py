import timm
import torch.nn as nn

from src.config import MODEL_NAME


class CropDiseaseModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)