from src.datasets.dataset import get_dataloaders


def main():
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)

    print(f"Number of Classes : {len(classes)}")
    print(f"Training Batches  : {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")
    print(f"Testing Batches   : {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print("\nBatch Shape")
    print(images.shape)
    print(labels.shape)

    print("\nFirst 5 Labels")
    print(labels[:5])


if __name__ == "__main__":
    main()