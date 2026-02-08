import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from datasets.tus import TUSDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_resnet(img_dir, dataset, task, transform=None, fold=3, batch_size=16, epochs=10, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{task.lower()}_resnet34.pth"

    train_dataset = TUSDataset(patch_level=1, img_root=img_dir, dataset=dataset, task=task, fold=fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = resnet34(pretrained=True)
    if task == 'S':
        num_classes = 3
    else:
        num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        for t1_img, t2_img, y in loader:
            # 支持 transform 后的 Tensor
            if transform is not None:
                t1_img = t1_img.float()
                t2_img = t2_img.float()

            if task=='T1':
                x = t1_img.to(device)
            elif task=='T2':
                x = t2_img.to(device)
            else:  # S 要把所有的图片传进去
                x = t1_img.to(device)  # 可改 fusion: torch.cat([t1_img, t2_img], dim=1)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        print(f"{task} Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # ✅ 修改：定义统一 transform
    tran = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_resnet(img_dir="IN/train", dataset='train', task='T1', transform=tran, fold=0, epochs=5)
    train_resnet(img_dir="IN/train", dataset='train', task='T2', transform=tran, fold=0, epochs=5)
    # train_resnet(img_dir="IN_dataset/train", dataset='train', task='S', transform=tran, fold=0, epochs=5)
