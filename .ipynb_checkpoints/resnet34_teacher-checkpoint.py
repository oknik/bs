import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from datasets.out import OUTDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_resnet(img_dir, task, fold=3, batch_size=16, epochs=50, checkpoint_path=None, patience=20):
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{task.lower()}_resnet34_fold{fold}.pth"

    # ===== Transforms =====
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ===== Dataset =====
    train_dataset = OUTDataset(img_root=img_dir, dataset='train', task=task, fold=fold, transform=transform)
    val_dataset   = OUTDataset(img_root=img_dir, dataset='valid', task=task, fold=fold, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ===== Model =====
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # ===== Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ===== Early Stopping =====
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ================= TRAIN =================
        model.train()
        train_loss = 0
        for img, y in train_loader:
            x = img.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for img, y in val_loader:
                x = img.to(device)
                y = y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)

        print(f"[{task}] Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        # ===== Early Stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({"model_state": model.state_dict()}, checkpoint_path)
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # 训练 T1 & T2
    train_resnet(img_dir="/root/autodl-tmp/bs/IN", task='T1', fold=0, epochs=50)
    train_resnet(img_dir="/root/autodl-tmp/bs/IN", task='T2', fold=0, epochs=50)
