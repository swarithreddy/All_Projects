from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from configs.config import (
    DATASET_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)

from src.datasets.transforms import (
    train_transform,
    val_transform,
)


def get_dataloaders():

    # Create separate datasets with different transforms
    train_dataset = ImageFolder(DATASET_PATH, transform=train_transform)
    val_dataset = ImageFolder(DATASET_PATH, transform=val_transform)

    total_indices = list(range(len(train_dataset)))

    train_indices, temp_indices = train_test_split(
        total_indices,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
        stratify=train_dataset.targets,
    )

    temp_targets = [train_dataset.targets[i] for i in temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=SEED,
        shuffle=True,
        stratify=temp_targets,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(val_dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes,
    )