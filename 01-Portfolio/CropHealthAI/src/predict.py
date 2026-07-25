import torch
from PIL import Image

from src.config import DEVICE
from src.model import CropDiseaseModel
from src.dataset import get_dataloaders
from src.transforms import test_transform


def predict(image_path):

    # Get class names
    _, _, _, classes = get_dataloaders()

    # Load model
    model = CropDiseaseModel(num_classes=len(classes))
    model.load_state_dict(
        torch.load("saved_models/best_model.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply transforms
    image = test_transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    disease = classes[predicted.item()]

    print("\nPrediction Result")
    print("-" * 30)
    print(f"Disease   : {disease}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    print("-" * 30)


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predict(image_path)