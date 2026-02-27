import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss.loss import SoftCELoss, CFLoss
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from models.cfl import CFL_ConvBlock
from datasets.in_dataset import INDataset
from datasets.paired_transform import PairedTransform
from utils import mkdir_if_missing, Logger
from models.resnet import *

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    #'densenet121': densenet121
}

# 计算熵（不确定性正则化）
def compute_entropy(probabilities):
    # 计算每个样本的熵 H(p)=−c=1∑C ​pc​log(pc)​
    return -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)  # 加上一个小常数避免log(0)
# 熵正则化项
def entropy_regularization(probabilities, lambda_entropy=0.1):
    entropy = compute_entropy(probabilities)
    # torch.mean 计算所有样本的平均熵作为正则化项
    # 乘以权重系数 lambda_entropy
    return lambda_entropy * torch.mean(entropy)

def get_parser():
    parser = argparse.ArgumentParser()
    # 需要修改
    parser.add_argument("--data_root", type=str, default='/root/autodl-tmp/bs/IN')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cfl_lr", type=float, default=None)
    # 需要修改
    parser.add_argument("--t1_ckpt", type=str, default='/root/autodl-tmp/bs/checkpoints/teacher/resnet18_6ch_20260210_155401/T1_fold0.pth')
    parser.add_argument("--t2_ckpt", type=str, default='/root/autodl-tmp/bs/checkpoints/teacher/resnet18_6ch_20260210_155401/T2_fold0.pth')

    parser.add_argument("--patience", type=int, default=10)
    return parser

# 训练学生模型一个 epoch，并返回平均 loss
def amal(cur_epoch, criterion, criterion_ce, criterion_cf, model, cfl_blk, teachers, optim, train_loader, device, scheduler=None, print_interval=100):
    """Train and return epoch loss"""
    # 两个教师模型
    t1, t2 = teachers

    #if scheduler is not None:
    #    scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    # avgmeter：一个用于统计和记录训练过程中指标平均值的工具类
    avgmeter = AverageMeter()
    #is_densenet = isinstance(model, DenseNet)
    is_densenet = False

    for cur_step, (images_C, images_G, labels) in enumerate(train_loader):
        images_C = images_C.to(device, dtype=torch.float32)
        images_G = images_G.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        images = torch.cat([images_C, images_G], dim=1)

        # get soft-target logits
        # 教师不更新参数（冻结）
        optim.zero_grad()
        with torch.no_grad():
            t1_out = t1(images)
            t2_out = t2(images)
            #print('labels:', labels[:10])
            #print('t1_out:', t1_out[:10])
            #print('t2_out:', t2_out[:10])

            # 拼接两个教师 logits(我的任务不应该直接拼接)
            t1_prob = F.softmax(t1_out, dim=1)  # [B,2]
            t2_prob = F.softmax(t2_out, dim=1)  # [B,2]

            # 提取教师特征（倒数第二层）
            ft1 = t1.layer4.output
            ft2 = t2.layer4.output
        
        # get student output
        # 学生 logits
        s_outs = model(images)
        s_prob = F.softmax(s_outs, dim=1)
        
        # 提取学生特征
        if is_densenet:
            fs = model.features.output
        else:
            fs = model.layer4.output

        ft = [ft1, ft2]

        (hs, ht), (ft_, ft) = cfl_blk(fs, ft)

        s_t1 = F.softmax(torch.stack([s_prob[:, 0], s_prob[:, 1] + s_prob[:, 2]], dim=1), dim=1)
        s_t2 = F.softmax(torch.stack([s_prob[:, 1], s_prob[:, 2]], dim=1), dim=1)

        # 计算熵正则化项
        #entropy_loss = entropy_regularization(t_outs)
        loss_1 = criterion(s_outs, labels)  #输出与真实标签之间计算损失
        #软目标损失
        loss_ce1 = criterion_ce(s_t1, t1_prob)
        loss_ce2 = criterion_ce(s_t2, t2_prob)
        loss_ce = loss_ce1 + loss_ce2
        
        loss_cf = 10*criterion_cf(hs, ht, ft_, ft) #MMD和重构损失

        loss = loss_1 + loss_ce + loss_cf

        loss.backward()
        optim.step()

        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())
        avgmeter.update('ce loss', loss_ce.item())
        avgmeter.update('cf loss', loss_cf.item())
        
        if (cur_step+1) % print_interval == 0:
            interval_loss = avgmeter.get_results('interval loss')
            ce_loss = avgmeter.get_results('ce loss')
            cf_loss = avgmeter.get_results('cf loss')

            print("Epoch %d, Batch %d/%d, Loss=%f (ce=%f, cf=%s)" %
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss, ce_loss, cf_loss))
            avgmeter.reset('interval loss')
            avgmeter.reset('ce loss')
            avgmeter.reset('cf loss')
    if scheduler is not None:
        scheduler.step()
    return avgmeter.get_results('loss')

def validate(model, loader, device):
    """Validate and compute all metrics"""
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images_C, images_G, labels in loader:
            images_C = images_C.to(device, dtype=torch.float32)
            images_G = images_G.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            images = torch.cat([images_C, images_G], dim=1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Sensitivity / Specificity（仅二分类有意义）
    if len(np.unique(all_labels)) == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        sensitivity, specificity = 0.0, 0.0

    return {
        'Overall Acc': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }


def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set up random seed
    mkdir_if_missing('checkpoints')
    mkdir_if_missing('logs')
    sys.stdout = Logger(os.path.join('logs', 'amal_%s.txt'%(opts.model)))
    print(opts)

    # 随机种子
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    cur_epoch = 0
    best_score = 0.0

    mkdir_if_missing('checkpoints')
    # 需要修改
    latest_ckpt = '/root/autodl-tmp/bs/checkpoints/student/%s_latest.pth'%opts.model
    best_ckpt = '/root/autodl-tmp/bs/checkpoints/student/%s_best.pth'%opts.model

    #  Set up dataloader
    #train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=opts.batch_size, download=opts.download)
    # 数据增强
    transform = PairedTransform()

    # 需要修改
    train_dataset = INDataset(img_root=opts.data_root, dataset='train', task='S', fold=0, transform=transform)
    val_dataset   = INDataset(img_root=opts.data_root, dataset='valid', task='S', fold=0, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    # pretrained teachers
    # 加载了教师模型，一个教师模型3分类，一个2分类。
    # 拆成了两个任务。数量多的一起分，数量少的一起分。缓解类别不均衡。
    t1 = _model_dict["resnet18"](num_classes=2)
    t2 = _model_dict["resnet18"](num_classes=2)

    t1.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    t2.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    in_features = t1.fc.in_features

    t1.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2)
    )

    t2.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2)
    )

    t1.to(device)
    t2.to(device)

    t1.load_state_dict(torch.load(opts.t1_ckpt, map_location=device))
    t2.load_state_dict(torch.load(opts.t2_ckpt, map_location=device))

    t1.eval()
    t2.eval()

    print("Target student: %s"%opts.model)
    #stu = _model_dict[opts.model](pretrained=True, num_classes=120+200).to(device)
    #metrics = StreamClsMetrics(120+200)
    stu = _model_dict[opts.model](pretrained=True, num_classes=3)
    stu.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    stu.to(device)

    metrics = StreamClsMetrics(3)

    # Setup Common Feature Blocks
    t1_feature_dim = t1.layer4[-1].conv2.out_channels
    t2_feature_dim = t2.layer4[-1].conv2.out_channels

    is_densenet = True if 'densenet' in opts.model else False

    if is_densenet:
        stu_feature_dim = stu.classifier.in_features
    else:
        stu_feature_dim = stu.fc.in_features #512
    
    # CFL Block（论文核心模块）
    cfl_blk = CFL_ConvBlock(stu_feature_dim, [t1_feature_dim, t2_feature_dim], 128).to(device)

    # forward hook（高级 PyTorch 技术）：  用于在前向传播时提取中间层的特征图
    def forward_hook(module, input, output):
        module.output = output # keep feature maps

    t1.layer4.register_forward_hook(forward_hook)
    t2.layer4.register_forward_hook(forward_hook)

    if is_densenet:
        stu.features.register_forward_hook(forward_hook)
    else:
        stu.layer4.register_forward_hook(forward_hook)

    params_1x = [] # backbone 参数（小学习率）
    params_10x = [] # 分类头 fc 参数（大学习率）
    for name, param in stu.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    # cfl 是新引入模块，随机初始化，必须快速收敛，所以用 10x lr
    cfl_lr = opts.lr*10 if opts.cfl_lr is None else opts.cfl_lr
    optimizer = torch.optim.Adam([{'params': params_1x,             'lr': opts.lr},
                                  {'params': params_10x,            'lr': opts.lr*10},
                                  {'params': cfl_blk.parameters(),  'lr': cfl_lr} ],
                                 lr=opts.lr, weight_decay=1e-4)
    # 每 15 个 epoch 学习率衰减为原来的 0.1 倍。
    # 除了 stepLR 还有 CosineAnnealing、MultiStepLR 等等
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1)

    # Loss
    # 三种 loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_ce = SoftCELoss(T=1.0)
    criterion_cf = CFLoss(normalized=True)

    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": stu.state_dict(),
            "cfl_state": cfl_blk.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(state, path)
        print("Model saved as %s" % path)

    print("Training ...")
    # ===== Train Loop =====#
    wait = 0
    while cur_epoch < opts.epochs:
        stu.train()
        # 调用 amal 函数训练一个 epoch
        epoch_loss = amal(cur_epoch=cur_epoch,
                            criterion=criterion,
                            criterion_ce=criterion_ce,
                            criterion_cf=criterion_cf,
                            model=stu,
                            cfl_blk=cfl_blk,
                            teachers=[t1, t2],
                            optim=optimizer,
                            train_loader=train_loader,
                            device=device,
                            scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f" %
              (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        save_ckpt(latest_ckpt)
        # =====  Validation  =====
        print("validate on val set...")
        stu.eval()
        val_score = validate(model=stu,
                             loader=val_loader,
                             device=device)
        print("Validation Metrics:")
        print(f"Accuracy      : {val_score['Overall Acc']:.4f}")
        print(f"F1 Score      : {val_score['F1']:.4f}")
        print(f"Precision     : {val_score['Precision']:.4f}")
        print(f"Recall        : {val_score['Recall']:.4f}")
        print(f"Sensitivity   : {val_score['Sensitivity']:.4f}")
        print(f"Specificity   : {val_score['Specificity']:.4f}")
        
        sys.stdout.flush()
        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > best_score:  # save best model
            best_score = val_score['Overall Acc']
            save_ckpt(best_ckpt)
            # 这里需要修改
            with open('/root/autodl-tmp/bs/checkpoints/tus/score_student.txt', mode='w') as f:
                f.write(metrics.to_str(val_score))
        elif val_score['Overall Acc'] <= best_score:
            wait += 1
            if wait == opts.patience:
                print('Early stopping at epoch: %d' % cur_epoch)
                break
        cur_epoch += 1

if __name__ == '__main__':
    main()
