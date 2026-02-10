import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from datasets.out import OUTDataset
from datasets.paired_transform import PairedTransform
import csv
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== class weights =====
def compute_class_weights(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = [total / counter[c] for c in sorted(counter.keys())]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


def train_resnet(img_dir, task, fold=0, batch_size=16, epochs=50,
                 checkpoint_path=None, patience=20, timestamp=None):

    # ===== Transform (paired) =====
    transform = PairedTransform()

    # ===== Dataset =====
    train_dataset = OUTDataset(img_root=img_dir, dataset='train', task=task, fold=fold, transform=transform)
    val_dataset   = OUTDataset(img_root=img_dir, dataset='valid', task=task, fold=fold, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ===== Model =====
    model = resnet18(pretrained=True)

    # ---- modify conv1 for 6-channel ----
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        6, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # copy pretrained weights
    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3:] = old_conv.weight

    # ---- classifier ----
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2)
    )

    # ---- freeze layers ----
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name and "conv1" not in name:
            param.requires_grad = False

    model = model.to(device)

    # ===== Loss =====
    weights = compute_class_weights(train_dataset.label_list).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ===== Early stopping =====
    best_val_loss = 1e9
    patience_counter = 0

    # 需要修改
    model_name = "resnet18_6ch"

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for img_C, img_G, y in train_loader:
            img_C, img_G, y = img_C.to(device), img_G.to(device), y.to(device)
            x = torch.cat([img_C, img_G], dim=1)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALID =================
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for img_C, img_G, y in val_loader:
                img_C, img_G, y = img_C.to(device), img_G.to(device), y.to(device)
                x = torch.cat([img_C, img_G], dim=1)

                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # ===== Metrics =====
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        print(f"[{task}] Epoch {epoch+1}/{epochs} "
              f"| Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} "
              f"| Acc {acc:.4f} | F1 {f1:.4f} | Prec {precision:.4f} | Rec {recall:.4f} "
              f"| Sen {sensitivity:.4f} | Spe {specificity:.4f}")

        # ===== Save best =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if checkpoint_path is None:
                checkpoint_path = f"checkpoints/teacher/{model_name}_{timestamp}/{task}_fold{fold}.pth"

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print("  -> Saved best model")

            # save predictions
            log_dir = f"logs/teacher/{model_name}_{timestamp}"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{task}_fold{fold}_best_val.csv")
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["gt", "pred"])
                for g, p in zip(all_labels, all_preds):
                    writer.writerow([g, p])

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("checkpoints", exist_ok=True)

    train_resnet(img_dir="/root/autodl-tmp/bs/IN", task="T1", fold=0, epochs=50, timestamp=timestamp)
    train_resnet(img_dir="/root/autodl-tmp/bs/IN", task="T2", fold=0, epochs=50, timestamp=timestamp)
