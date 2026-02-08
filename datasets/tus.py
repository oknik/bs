# tus.py
import torch.utils.data as data_utils
import csv
import random
import pickle

class TUSDataset(data_utils.Dataset):

    def __init__(self, patch_level, img_root, dataset, transform=None, fold=3):
        self.data_list = []
        self.transform = transform
        self.patch_level = patch_level

        if dataset == 'train':
            data_num = [i for i in range(10) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]

        print("Use the data fold: {}".format(fold))

        for i in data_num:  
            # 需要修改
            f = open('/data2/dongchunlai/data/thyroid_dataset/ten_fold/fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                # 保持 row[1:] 结构
                self.data_list.append(row[1:])

        # ===============================
        # 标签映射（只改这里）
        # 原始 label: 1,2,3,4,5,6,7
        # 目标 student: 0=other(12347), 1=follicular(5), 2=medullary(6)
        # ===============================

        new_list = []
        for item in self.data_list:
            img_name = item[0]
            label = int(item[1])

            # other = 1,2,3,4,7
            if label in [1, 2, 3, 4, 7]:
                new_label = 0
            elif label == 5:
                new_label = 1
            elif label == 6:
                new_label = 2
            else:
                continue

            new_list.append([img_name, str(new_label)])

        self.data_list = new_list

        # label_list 保持一致
        self.label_list = [int(x[-1]) for x in self.data_list]

        if self.patch_level == 1:
            self.patch_class = self.get_token_class(fold)

        # 保持原 label_num 结构（扩展为3类）
        self.label_num = [0, 0, 0]
        for each in self.label_list:
            self.label_num[each] += 1

        print(self.label_num)
        print(len(self.data_list))
