from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from src.config import DATASET_PATH, BATCH_SIZE
from src.transforms import train_transform, test_transform


def get_dataloaders():

    # Load dataset with training transforms
    full_dataset = ImageFolder(
        root=DATASET_PATH,
        transform=train_transform
    )

    # Dataset sizes
    total_size = len(full_dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    # Validation and Test should NOT use augmentation
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_dataset.classes
    )