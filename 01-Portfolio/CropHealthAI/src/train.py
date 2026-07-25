import torch
import torch.nn as nn
import torch.optim as optim

from src.config import DEVICE, EPOCHS, LEARNING_RATE
from src.dataset import get_dataloaders
from src.model import CropDiseaseModel


def train():

    print("Loading dataset...")

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print(f"Number of classes: {len(classes)}")

    # Create model
    model = CropDiseaseModel(
        num_classes=len(classes)
    ).to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0

    print("\nTraining Started...\n")

    # Training Loop
    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        # -------------------------
        # Training
        # -------------------------
        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)

            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss : {train_loss/len(train_loader):.4f}")
        print(f"Train Accuracy : {train_accuracy:.2f}%")
        print(f"Validation Loss : {val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy : {val_accuracy:.2f}%")
        print("-" * 50)

        # Save Best Model
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "saved_models/best_model.pth"
            )

            print("✅ Best model saved!\n")

    print("\nTraining Completed!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    train()