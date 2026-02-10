import os
import csv
from PIL import Image
import torch.utils.data as data_utils

class OUTDataset(data_utils.Dataset):

    def __init__(self, img_root, dataset, task, transform=None, fold=0):
        self.img_root = img_root
        self.transform = transform
        self.task = task
        self.data_list = []    # img_id list
        self.label_list = []   # mapped label
        self.img_map = {}      # img_id -> {"C": fname, "G": fname}

        # ===== scan train folder =====
        train_dir = os.path.join(img_root, "train")
        for fname in os.listdir(train_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                # filename format: id-label-mode.xxx
                parts = fname.split("-")
                img_id = parts[0]
                mode = parts[2].split(".")[0]   # C or G

                if img_id not in self.img_map:
                    self.img_map[img_id] = {}
                self.img_map[img_id][mode] = fname

        # ===== load CSV =====
        if dataset == 'train':
            data_num = [i for i in range(5) if i != fold]
        else:
            data_num = [fold]

        print(f"{dataset.upper()} fold: {fold}")

        for i in data_num:
            csv_path = os.path.join(img_root, f"fold{i}.csv")
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    img_id = row[0]
                    label = int(row[1])

                    # ensure both C and G exist
                    if img_id not in self.img_map or \
                       "C" not in self.img_map[img_id] or \
                       "G" not in self.img_map[img_id]:
                        raise FileNotFoundError(f"{img_id} missing C/G images")

                    self.data_list.append(img_id)
                    self.label_list.append(label)

        # ===== map labels T1/T2 =====
        self._map_labels()

        # stats
        from collections import Counter
        print("Label counts:", Counter(self.label_list))
        print("Total samples:", len(self.data_list))

    def _map_labels(self):
        new_data, new_labels = [], []
        for img_id, raw_label in zip(self.data_list, self.label_list):
            if self.task == 'T1':
                new_data.append(img_id)
                new_labels.append(0 if raw_label == 0 else 1)
            elif self.task == 'T2':
                if raw_label in [1, 2]:
                    new_data.append(img_id)
                    new_labels.append(raw_label - 1)
        self.data_list = new_data
        self.label_list = new_labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_id = self.data_list[idx]
        label = self.label_list[idx]

        fname_C = self.img_map[img_id]["C"]
        fname_G = self.img_map[img_id]["G"]

        path_C = os.path.join(self.img_root, "train", fname_C)
        path_G = os.path.join(self.img_root, "train", fname_G)

        img_C = Image.open(path_C).convert("RGB")
        img_G = Image.open(path_G).convert("RGB")

        # ===== paired transform =====
        if self.transform:
            img_C, img_G = self.transform(img_C, img_G)

        return img_C, img_G, label
