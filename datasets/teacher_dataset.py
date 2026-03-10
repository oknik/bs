import os
import csv
import random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data_utils
import torchvision.transforms.functional as tf


class TeacherDataset(data_utils.Dataset):

    def __init__(self, img_root, dataset, args, task, fold=0, few_shot=False, support_dataset=None):

        super().__init__()

        # 必须参数
        self.img_root = img_root
        self.dataset = dataset
        self.args = args
        self.task = task

        self.fold = fold
        self.few_shot = few_shot
        self.support_dataset = support_dataset

        self.data = []
        self.label = []
        self.img_map = {}

        train_dir = os.path.join(img_root, "train")

        # ========= 扫描图像 =========
        for fname in os.listdir(train_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                parts = fname.split("-")

                img_id = parts[0]
                mode = parts[2].split(".")[0]

                if img_id not in self.img_map:
                    self.img_map[img_id] = {}

                self.img_map[img_id][mode] = fname

        # ========= 读取CSV =========
        if dataset == 'train':
            folds = [i for i in range(5) if i != fold]
        else:
            folds = [fold]

        for f in folds:
            csv_path = os.path.join(img_root, f"fold{f}.csv")
            with open(csv_path) as file:
                reader = csv.reader(file)
                next(reader)

                for row in reader:
                    img_id = row[0]
                    raw_label = int(row[1])

                    if img_id not in self.img_map:
                        continue
                    if "C" not in self.img_map[img_id] or "G" not in self.img_map[img_id]:
                        continue

                    self.data.append(img_id)
                    self.label.append(raw_label)

        # label映射
        self._map_labels()

        # few-shot class index
        self.class_to_indices = defaultdict(list)

        for idx, lab in enumerate(self.label):
            self.class_to_indices[lab].append(idx)

        print("Label distribution:",
              {k: len(v) for k, v in self.class_to_indices.items()})
        print("Total samples:", len(self.data))

    # =================================
    # label remap
    # =================================
    def _map_labels(self):
        new_data = []
        new_label = []

        for img_id, raw_label in zip(self.data, self.label):
            if self.task == 'S':
                new_data.append(img_id)
                new_label.append(raw_label)
            elif self.task == 'T1':
                new_data.append(img_id)
                new_label.append(0 if raw_label == 0 else 1)
            elif self.task == 'T2':
                if raw_label in [1, 2]:
                    new_data.append(img_id)
                    new_label.append(raw_label - 1)

        self.data = new_data
        self.label = new_label

    # =================================
    # paired augmentation
    # =================================
    def get_aug_img(self, img1, img2):
        img1 = tf.resize(img1, [224, 224])
        img2 = tf.resize(img2, [224, 224])

        if random.random() > 0.5:
            img1 = tf.hflip(img1)
            img2 = tf.hflip(img2)

        if random.random() > 0.5:
            img1 = tf.vflip(img1)
            img2 = tf.vflip(img2)

        if random.random() > 0.3:
            degree = random.uniform(-10, 10)
            img1 = tf.rotate(img1, degree)
            img2 = tf.rotate(img2, degree)

        img1 = tf.to_tensor(img1)
        img2 = tf.to_tensor(img2)

        img1 = tf.normalize(img1, mean=[0.46]*3, std=[0.1582]*3)
        img2 = tf.normalize(img2, mean=[0.46]*3, std=[0.1582]*3)

        return img1, img2

    # =================================
    # 读取图像对
    # =================================
    def load_pair(self, img_id):
        fname_C = self.img_map[img_id]["C"]
        fname_G = self.img_map[img_id]["G"]

        path_C = os.path.join(self.img_root, "train", fname_C)
        path_G = os.path.join(self.img_root, "train", fname_G)

        img_C = Image.open(path_C).convert("RGB")
        img_G = Image.open(path_G).convert("RGB")

        img_C, img_G = self.get_aug_img(img_C, img_G)

        return img_C, img_G

    # =================================
    # few-shot support set
    # =================================
    def get_support_set(self):
        support_C = []
        support_G = []
        support_label = []

        classes = list(self.support_dataset.class_to_indices.keys())

        for new_label, cls in enumerate(classes):
            indices = self.support_dataset.class_to_indices[cls]

            chosen = random.sample(indices, self.args.shot)

            for idx in chosen:
                img_id = self.support_dataset.data[idx]
                img_C, img_G = self.support_dataset.load_pair(img_id)

                support_C.append(img_C)
                support_G.append(img_G)
                support_label.append(new_label)

        support_C = torch.stack(support_C)
        support_G = torch.stack(support_G)
        support_label = torch.tensor(support_label)

        return support_C, support_G, support_label

    # =================================
    # dataset接口
    # =================================
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data[idx]
        label = self.label[idx]

        query_C, query_G = self.load_pair(img_id)

        # 普通模式
        if not self.few_shot:
            return query_C, query_G, label

        # few-shot episode
        support_C, support_G, support_label = self.get_support_set()

        query_C = query_C.unsqueeze(0)
        query_G = query_G.unsqueeze(0)
        query_label = torch.tensor([label])

        return (
            query_C,
            query_G,
            query_label,
            support_C,
            support_G,
            support_label
        )