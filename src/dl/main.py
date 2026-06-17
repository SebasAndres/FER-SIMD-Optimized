import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dl.constants import BATCH_SIZE, EPOCHS, LR, DEVICE
from src.dl.dataset import FERDataset
from src.dl.nn import SimpleCNN


def get_dataloaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    train_ds = FERDataset(split="train", transform=train_transform)
    val_ds = FERDataset(split="val")
    test_ds = FERDataset(split="test")

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader,
    criterion: nn.Module, 
    optimizer: optim.Optimizer
) -> float:

    model.train()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0

    print(f"Training on {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
