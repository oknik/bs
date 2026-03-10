import numpy as np
import torch
        
class PairGenerator:

    def __init__(self, dataset, x_epoch_all, args):
        self.x_epoch_all = x_epoch_all
        self.dataset = dataset
        self.args = args

    def pair_load(self, _label, shot_idx, query_idx):
        img_shot, img_shot_start = self.dataset.get_aug_img(self.data[shot_idx], self.data_start[shot_idx])
        img_query, img_query_start = self.dataset.get_aug_img(self.data[query_idx], self.data_start[query_idx])

        self.data_shot.append(img_shot)
        self.data_shot_start.append(img_shot_start)
        self.data_query.append(img_query)
        self.data_query_start.append(img_query_start)
        self.label.append(_label)

    def pair_back(self):
        self.data_shot = torch.cat([_.unsqueeze(0) for _ in self.data_shot]).cuda()
        self.data_shot_start = torch.cat([_.unsqueeze(0) for _ in self.data_shot_start]).cuda()
        self.data_query = torch.cat([_.unsqueeze(0) for _ in self.data_query]).cuda()
        self.data_query_start = torch.cat([_.unsqueeze(0) for _ in self.data_query_start]).cuda()
        self.label = torch.Tensor(np.array(self.label)).cuda()
        return self.data_shot, self.data_shot_start, self.data_query, self.data_query_start, self.label


    def pair_generator_balance(self):
        for i in range(len(self.labels)):
            self.pair_load(1, i, i)# 每个样本与自己配对，标签为1
            for j in range(i+1, len(self.labels)):
                if self.labels[i] == self.labels[j]:
                    _label = 1
                    
                else: 
                    _label = 0
                
                self.pair_load(_label, j, i)# 每个样本与其他样本配对，标签根据是否同类确定
                self.pair_load(_label, i, j)# 每对样本配对两次，保证平衡，标签根据是否同类确定

    def pair_generator(self):
        for i in range(len(self.labels)):
            for j in range(i+1, len(self.labels)):
                if self.labels[i] == self.labels[j]:
                    _label = 1
                else: 
                    _label = 0
                self.pair_load(_label, i, j)
    
    def batch_generator(self, epoch, data, data_start, labels):
        self.data = data
        self.data_start = data_start
        self.data_shot = []
        self.data_query = []
        self.data_shot_start = []
        self.data_query_start = []
        self.label = []
        self.labels = np.array(labels.cpu())
        
        self.pair_generator_balance()
        return self.pair_back()