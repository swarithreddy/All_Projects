import streamlit as st
import torch
from PIL import Image

from src.model import CropDiseaseModel
from src.dataset import get_dataloaders
from src.transforms import test_transform
from src.config import DEVICE

# ----------------------------------
# Load Class Names
# ----------------------------------
_, _, _, classes = get_dataloaders()

# ----------------------------------
# Load Trained Model
# ----------------------------------
model = CropDiseaseModel(num_classes=len(classes))
model.load_state_dict(
    torch.load(
        "saved_models/best_model.pth",
        map_location=DEVICE
    )
)
model.to(DEVICE)
model.eval()

# ----------------------------------
# Disease Information
# ----------------------------------
disease_info = {
    "Apple___Black_rot": {
        "description": "Fungal disease affecting apple fruits and leaves.",
        "treatment": "Remove infected leaves and spray a recommended fungicide."
    },
    "Apple___healthy": {
        "description": "The plant appears healthy.",
        "treatment": "Continue regular watering and nutrient management."
    },
    "Tomato___Late_blight": {
        "description": "A serious fungal disease caused by Phytophthora infestans.",
        "treatment": "Remove infected leaves and apply a suitable fungicide."
    },
    "Tomato___Early_blight": {
        "description": "A fungal disease causing brown spots on leaves.",
        "treatment": "Use disease-free seeds and apply fungicide if required."
    }
}

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("🌿 Crop Disease Detection")

st.write("Upload a crop leaf image to identify the disease.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    disease = classes[predicted.item()]

    st.success(f"Prediction: {disease}")

    st.write(f"Confidence: **{confidence.item()*100:.2f}%**")

    if disease in disease_info:

        st.subheader("Disease Description")

        st.write(disease_info[disease]["description"])

        st.subheader("Suggested Treatment")

        st.write(disease_info[disease]["treatment"])

    else:

        st.info("No additional information available for this disease.")