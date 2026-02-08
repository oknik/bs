class INDataset(data_utils.Dataset):

    def __init__(self, patch_level, img_root, dataset, task, transform=None, fold=3):
        """
        task: 'T1', 'T2', 'S'
        """
        self.data_list = []
        self.transform = transform
        self.patch_level = patch_level
        self.task = task

        if dataset == 'train':
            data_num = [i for i in range(5) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]
        print("Validation fold: {}".format(fold))

        for i in data_num:  
            f = open('/root/autodl-tmp/bs/IN/fold{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data_list.append(row)

        # label_list 原始标签
        self.raw_label_list = [int(x[-1]) for x in self.data_list]

        # 映射 label
        self.label_list = self.map_labels(self.raw_label_list)

        # patch_class
        if self.patch_level == 1:
            self.patch_class = self.get_token_class(fold)

        # label_num
        if task == 'S':
            self.label_num = [0,0,0]
        else:
            self.label_num = [0,0]  # 二分类
        for each in self.label_list:
            self.label_num[each] += 1

        print("Label counts:", self.label_num)
        print("Total samples:", len(self.data_list))

    def map_labels(self, labels):
        if self.task == 'T1':
            # 0 -> 0, 1/2 -> 1
            return [0 if l==0 else 1 for l in labels]
        elif self.task == 'T2':
            # 1 -> 0, 2 -> 1, ignore 0?
            # 如果 T2 只区分 1 和 2，可以过滤掉 0
            filtered_labels = []
            filtered_idx = []
            for idx, l in enumerate(labels):
                if l in [1,2]:
                    filtered_labels.append(l-1)  # 1->0, 2->1
                    filtered_idx.append(idx)
            # 保留对应 data_list
            self.data_list = [self.data_list[i] for i in filtered_idx]
            return filtered_labels
        else:  # S
            return labels

    def __len__(self):
        return len(self.data_list)
