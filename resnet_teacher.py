import os
import csv
import datetime
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from datasets.teacher_dataset import TeacherDataset
from pair_generator import PairGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# class weights
# ===============================
def compute_class_weights(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = [total / counter[c] for c in sorted(counter.keys())]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


# ===============================
# model
# ===============================
def build_model():
    model = resnet18(pretrained=True)

    # ---------- modify for 6 channel ----------
    old_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3:] = old_conv.weight

    # ---------- embedding head ----------
    in_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 128))

    return model

# ===============================
# cosine regularization loss
# ===============================
def cosine_regularization(sim, label, margin=0.5):
    pos = label * (1 - sim)
    neg = (1 - label) * torch.relu(sim - margin)

    loss = torch.mean(pos + neg)

    return loss

# ===============================
# training
# ===============================
def train_resnet(img_dir, task, fold=0, batch_size=16, epochs=50, patience=20, timestamp=None):
    # ===============================
    # Dataset
    # ===============================
    train_dataset = TeacherDataset(img_root=img_dir, dataset='train', task=task, fold=fold)
    val_dataset = TeacherDataset(img_root=img_dir, dataset='valid', task=task, fold=fold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ===============================
    # pair generator
    # ===============================
    pair_generator = PairGenerator(train_dataset, None, None)

    # ===============================
    # model
    # ===============================
    model = build_model().to(device)

    # classification head
    classifier = nn.Linear(128, 2).to(device)

    # ===============================
    # loss
    # ===============================
    pair_loss = nn.BCELoss()

    weights = compute_class_weights(train_dataset.label_list).to(device)
    cls_loss = nn.CrossEntropyLoss(weight=weights)

    # ===============================
    # optimizer
    # ===============================
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # ===============================
    # log
    # ===============================
    model_name = "resnet18_pair_outp"

    log_dir = f"logs/teacher/{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{task}_fold{fold}.csv")

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "acc",
        "f1",
        "precision",
        "recall",
        "sensitivity",
        "specificity"
    ])

    # ===============================
    # early stopping
    # ===============================
    best_val_loss = 1e9
    patience_counter = 0

    for epoch in range(epochs):

        # ===============================
        # TRAIN
        # ===============================
        model.train()
        classifier.train()
        train_loss = 0

        for img_C, img_G, y in train_loader:
            img_C = img_C.to(device)
            img_G = img_G.to(device)
            y = y.to(device)

            # ---------- pair generation ----------
            data_shot_C, data_shot_G, data_query_C, data_query_G, pair_label = pair_generator.batch_generator(epoch, img_C, img_G, y)

            print(data_shot_C.shape, data_shot_G.shape, data_query_C.shape, data_query_G.shape, pair_label.shape)
            # ---------- 6 channel ----------
            shot = torch.cat([data_shot_C, data_shot_G], dim=1)
            query = torch.cat([data_query_C, data_query_G], dim=1)
            print(shot.shape, query.shape)

            # ---------- forward ----------
            optimizer.zero_grad()

            emb_shot = F.normalize(model(shot), dim=1)
            emb_query = F.normalize(model(query), dim=1)

            sim = torch.cosine_similarity(emb_shot, emb_query)
            sim_prob = torch.sigmoid(sim)

            loss_pair = pair_loss(sim_prob, pair_label)
            loss_cos = cosine_regularization(sim, pair_label)

            x_cls = torch.cat([img_C, img_G], dim=1)
            emb_cls = F.normalize(model(x_cls), dim=1)
            logits = classifier(emb_cls)
            loss_cls = cls_loss(logits, y)

            cosine_weight = 0.1
            cls_weight = 0.3
            loss = loss_pair + cosine_weight * loss_cos + cls_weight * loss_cls

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===============================
        # VALIDATION
        # ===============================
        model.eval()
        classifier.eval()

        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for img_C, img_G, y in val_loader:
                img_C = img_C.to(device)
                img_G = img_G.to(device)
                y = y.to(device)

                x = torch.cat([img_C, img_G], dim=1)

                emb = model(x)

                logits = classifier(emb)

                loss = cls_loss(logits, y)

                val_loss += loss.item()

                pred = logits.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # ===============================
        # metrics
        # ===============================
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        print(
            f"[{task} Fold{fold}] Epoch {epoch+1}/{epochs} "
            f"| Train {train_loss:.4f} "
            f"| Val {val_loss:.4f} "
            f"| Acc {acc:.4f} "
            f"| F1 {f1:.4f}"
        )

        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            acc,
            f1,
            precision,
            recall,
            sensitivity,
            specificity
        ])

        # ===============================
        # save best
        # ===============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_dir = f"checkpoints/teacher/{model_name}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)

            torch.save(
                {
                    "backbone": model.state_dict(),
                    "classifier": classifier.state_dict()
                },
                os.path.join(save_dir, f"{task}_fold{fold}.pth")
            )

            print("  -> Saved best model")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break
    log_file.close()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("checkpoints", exist_ok=True)

    train_resnet(
        img_dir="/root/autodl-tmp/bs/OUTP/teacher",
        task="T1",
        fold=0,
        epochs=50,
        timestamp=timestamp
    )

    train_resnet(
        img_dir="/root/autodl-tmp/bs/OUTP/teacher",
        task="T2",
        fold=0,
        epochs=50,
        timestamp=timestamp
    )