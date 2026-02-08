class OUTDataset(data_utils.Dataset):

    def __init__(self, patch_level, img_root, dataset, task, transform=None, fold=0):
        """
        task: 'T1', 'T2'
        dataset: 'train' or 'valid'
        fold: 哪一折作为验证
        """
        self.data_list = []
        self.transform = transform
        self.patch_level = patch_level
        self.task = task

        if dataset == 'train':
            data_num = [i for i in range(5) if i != fold]  # 排除验证 fold
        elif dataset == 'valid':
            data_num = [fold]  # 只使用验证 fold

        print(f"{dataset.upper()} fold: {fold}")

        # 读取对应 fold 的 CSV
        for i in data_num:
            f = open(f'{img_root}/fold{i}.csv', 'r')
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data_list.append(row)

        # 原始标签
        self.raw_label_list = [int(x[-1]) for x in self.data_list]

        # 标签映射
        self.label_list = self.map_labels(self.raw_label_list)

        # label_num
        self.label_num = [0, 0]
        for l in self.label_list:
            self.label_num[l] += 1

        print("Label counts:", self.label_num)
        print("Total samples:", len(self.data_list))

    def map_labels(self, labels):
        if self.task == 'T1':
            return [0 if l == 0 else 1 for l in labels]
        elif self.task == 'T2':
            filtered_labels = []
            filtered_idx = []
            for idx, l in enumerate(labels):
                if l in [1, 2]:
                    filtered_labels.append(l - 1)  # 1->0, 2->1
                    filtered_idx.append(idx)
            # 保留对应 data_list
            self.data_list = [self.data_list[i] for i in filtered_idx]
            return filtered_labels

    def __len__(self):
        return len(self.data_list)
