import os
import random
import time
import copy
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from Dataset import Dataset
from Dataset import Dataset as MyDataset
import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import confusion_matrix
import torchvision.models as models
from torch.utils.data import WeightedRandomSampler

def get_class_counts(data_loader):
    counts = {}
    for _, y in data_loader:
        for label in y.tolist():
            counts[label] = counts.get(label, 0) + 1
    num_classes = len(counts)
    cls_num_list = [counts.get(i, 0) for i in range(num_classes)]
    return cls_num_list

def get_pretrained_model(model_name, is_train=True, device=None):
    import timm  # optional dependency

    model = timm.create_model(model_name, pretrained=True)
    if is_train:
        model.train()
    else:
        model.eval()
    if device:
        model = model.to(device)
    return model


def model_fc_fix(model, nb_classes, device=None):
    model.reset_classifier(num_classes=nb_classes)
    if device:
        model = model.to(device)
    return model


def get_kfold_img_idx(p, k, dataset, sample_type=None):
    '''
    KLSG_balance_num: set the balance number of each class for KLSG dataset
    LTSID_balance_num: set the balance number of each class for LTSID dataset
    FLSMDD_balance_num: set the balance number of each class for FLSMDD dataset
    KLSG_class_slices: index range of every class for KLSG dataset, like class 0: [0, 43], class 1: [43, 313]
    LTSID_class_slices: index range of every class for LTSID dataset
    FLSMDD_class_slices: index range of every class for FLSMDD dataset
    '''
    if sample_type in ('balance', 'balance_min'):
        KLSG_balance_num = 48
        NKSID_balance_num = 120
        FLSMDD_balance_num = 115
        KLSG_class_slices = [0, 62, 447]
        NKSID_class_slices = [0, 203, 491, 511, 1462, 1574, 1668, 1783, 2617]
        FLSMDD_class_slices = [0, 449, 816, 1042, 1391, 1524, 1661, 1760, 1825, 2156, 2364]

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    kfold_train_path = os.path.join(curr_dir, '..' , 'Data', dataset, 'kfold_train.txt')
    kfold_val_path = os.path.join(curr_dir, '..' , 'Data', dataset, 'kfold_val.txt')
    f_train = open(kfold_train_path, 'r')
    f_val = open(kfold_val_path, 'r')
    train_lines = f_train.readlines()
    val_lines = f_val.readlines()
    train_line = None
    val_line = None
    for i in range(len(train_lines)):
        if train_lines[i][0] == '#':
            str_cuts = train_lines[i].rstrip().split('-')
            p_read = int(str_cuts[0][2:])
            k_read = int(str_cuts[1][1:])
            if p_read == p and k_read == k:
                train_line = train_lines[i + 1]
                break
    for i in range(len(val_lines)):
        if val_lines[i][0] == '#':
            str_cuts = val_lines[i].rstrip().split('-')
            p_read = int(str_cuts[0][2:])
            k_read = int(str_cuts[1][1:])
            if p_read == p and k_read == k:
                val_line = val_lines[i + 1]
                break
    train_line = [int(str) for str in train_line.rstrip().split(' ')]
    val_line = [int(str) for str in val_line.rstrip().split(' ')]
    if sample_type:

        if sample_type == 'uniform':
            # NOTE:
            # 原实现是“带放回抽样”，会造成训练集大量重复、同时漏掉很多样本，
            # 这会让交叉验证结果不稳定/偏差很大。这里修正为“无放回shuffle”。
            train_line = list(train_line)
            random.shuffle(train_line)

        elif sample_type == 'balance':
            balance_train_line = []
            balance_num = 0
            if dataset == 'KLSG':
                balance_num = KLSG_balance_num
                class_slices = KLSG_class_slices
            elif dataset == 'NKSID':
                balance_num = NKSID_balance_num
                class_slices = NKSID_class_slices
            elif dataset == 'FLSMDD':
                balance_num = FLSMDD_balance_num
                class_slices = FLSMDD_class_slices
            else:
                print(f'ERROR! DATASET {dataset} IS NOT EXIST!')
                return None
            train_line.sort()
            for i in range(len(class_slices) - 1):
                tmp_arr = [tr for tr in train_line if tr >= class_slices[i] and tr < class_slices[i + 1]]
                balance_train_line = balance_train_line + [random.sample(tmp_arr, 1)[0] for n in range(balance_num)]
            train_line = balance_train_line

        elif sample_type == 'balance_min':
            # 按“最少类样本数”做不放回下采样：每类抽 min_count 个
            if dataset == 'KLSG':
                class_slices = KLSG_class_slices
            elif dataset == 'NKSID':
                class_slices = NKSID_class_slices
            elif dataset == 'FLSMDD':
                class_slices = FLSMDD_class_slices
            else:
                print(f'ERROR! DATASET {dataset} IS NOT EXIST!')
                return None

            train_line.sort()

            # 先统计每类在该fold里有多少
            per_class = []
            min_cnt = None
            for i in range(len(class_slices) - 1):
                tmp_arr = [tr for tr in train_line if tr >= class_slices[i] and tr < class_slices[i + 1]]
                per_class.append(tmp_arr)
                if min_cnt is None or len(tmp_arr) < min_cnt:
                    min_cnt = len(tmp_arr)

            if min_cnt is None or min_cnt == 0:
                print(f'ERROR! balance_min got empty class in dataset {dataset}, p={p}, k={k}')
                return None

            # 每类不放回抽 min_cnt 个（头类会被丢弃一部分，尾类通常会全用上）
            balance_train_line = []
            for tmp_arr in per_class:
                # 不放回抽样
                balance_train_line += random.sample(tmp_arr, min_cnt)

            random.shuffle(balance_train_line)
            train_line = balance_train_line

        else:
            print(f'ERROR! sample_type {sample_type} is not exist!')
            return None
    return train_line, val_line


'''
Get k-fold image data iters.
    @ batch_size : batch size
    @ data_dir : dataset direction
    @ train_idx : index list(s) of train images you want, like [[1,2,3],[4,5,6],...] or [1,2,3,...]
    @ val_idx : index list(s) of validation images you want
    @ mean : the mean of the ImageNet dataset
    @ std : the variance of the ImageNet dataset  
'''


def get_kfold_img_iters(batch_size, data_dir, train_idx, val_idx, mean, std, num_workers=0):
    """
    Unified CV preprocessing (ImageNet normalize):
      train: Resize 224 -> AutoAugment(IMAGENET) -> RandomHorizontalFlip -> RandomRotation(15) -> ToTensor -> normalize
      val:   Resize 224 -> ToTensor -> normalize
    """
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])

    if isinstance(train_idx[0], list):
        train_iter = []
        for idx in train_idx:
            train_imgs = MyDataset(os.path.join(data_dir, 'train.txt'), idx,
                                   transform=train_transform)
            train_iter.append(torch.utils.data.DataLoader(train_imgs,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          num_workers=num_workers))
    elif isinstance(train_idx[0], int):
        train_imgs = MyDataset(os.path.join(data_dir, 'train.txt'), train_idx,
                               transform=train_transform)
        train_iter = torch.utils.data.DataLoader(train_imgs,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)

    val_imgs = MyDataset(os.path.join(data_dir, 'train.txt'), val_idx,
                         transform=val_transform)
    val_iter = torch.utils.data.DataLoader(val_imgs,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers)
    return train_iter, val_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None: device = next(net.parameters()).device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            output = net(X.to(device)) # Test时只返回一个 logits
            acc_sum += (output.argmax(dim=1) == y.to(device)).float().sum().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n

def cal_y_hat(fcs, features, method, device, fusion_net = None, if_get_logits=False):
    y_hat = torch.tensor([]).to(device)
    if method == 'normal':
        logits = []
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval()
                logits.append(fcs[i](features).argmax(dim=1).tolist())
        y_hat = torch.tensor(logits).to(device)
    elif method == 'hard_voting':
        labels_sum = torch.tensor([0]).to(device)
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval()
                logits = fcs[i](features)
                labels_sum = labels_sum + F.one_hot(logits.argmax(dim=1), logits.shape[1])
        y_hat = labels_sum.argmax(dim=1)
    elif method == 'soft_voting':
        logits_sum = torch.tensor([0]).to(device)
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval()
                logits_sum = logits_sum + fcs[i](features)
        y_hat = logits_sum.argmax(dim=1)
    elif method == 'stacking':
        if fusion_net == None:
            print('ERROR! FUSION NET SHOULD NOT BE NONE!')
            return None
        logits = torch.tensor([]).to(device)
        for fc in fcs:
            fc.eval()  # open evaluate mode
            logits = torch.cat((logits, fc(features)), 1)
        fusion_net.eval() # open evaluate mode
        y_hat = fusion_net(logits).argmax(dim=1)
    elif method == 'weight_averaging':
        fus_param = copy.deepcopy(fcs[0].state_dict()) # fusion params of fcs
        for i in range(1, len(fcs)):
            fus_param['fc.weight'] += fcs[i].state_dict()['fc.weight']
            fus_param['fc.bias'] += fcs[i].state_dict()['fc.bias']
        fus_param['fc.weight'] /= len(fcs)
        fus_param['fc.bias'] /= len(fcs)
        fus_fc = copy.deepcopy(fcs[0]) # use deepcopy to incase fcs[0] be changed
        fus_fc.load_state_dict(fus_param)
        y_hat = fus_fc(features).argmax(dim=1)
        if if_get_logits:
            logits = fus_fc(features)
            return y_hat, logits
    else:
        print(f'ERROR! Method {method} is not exist!')
        return None
    return y_hat

def cal_gmean(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu()
    conf_matrix = confusion_matrix(y_true, y_pred)
    diag = np.diagonal(conf_matrix) # right classifying number per class
    n_per_class = np.sum(conf_matrix, axis=1) # original number per class
    if(0 not in n_per_class):
        acc_per_class = diag / n_per_class
        gm = gmean(acc_per_class)
        return gm
    else:
        print('ERROR: Number of classes can not be 0 !')
        return None


def evaluate_gmean(data_iter, net, device=None, if_get_y=False, if_get_logits=False):
    if device is None: device = next(net.parameters()).device
    y_true = torch.tensor([]).to(device)
    y_hat = torch.tensor([]).to(device)
    logits_store = torch.tensor([]).to(device)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            net.eval()
            output = net(X)
            logits = output # eval模式只返回一个
            logits_store = torch.cat((logits_store, logits), 0)
            y_hat = torch.cat((y_hat, logits.argmax(dim=1)), 0)
    gmean_fc = cal_gmean(y_true, y_hat)
    if if_get_logits: return gmean_fc, y_hat, y_true, logits_store
    if if_get_y: return gmean_fc, y_hat, y_true
    return gmean_fc

# Stage2改变抽样分布，但epoch总抽样数不变
def build_balanced_sampler_by_idx(train_txt_path, idx_list, nb_classes,
                                  alpha=1.0, num_samples=None, replacement=True):
    """
    用 WeightedRandomSampler 改变抽样分布：
      - alpha=0   -> 等价原始分布
      - alpha=1   -> 类概率近似均匀（期望接近平衡，如 51/51）
    且 num_samples 默认 = len(idx_list)，即每个 epoch 总样本数不变。

    Returns:
      sampler: WeightedRandomSampler
      counts:  原始 idx_list 内各类计数（不是采样后的）
    """
    with open(train_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    labels = np.array([int(lines[i].rstrip().split()[1]) for i in idx_list], dtype=np.int64)
    counts = np.bincount(labels, minlength=nb_classes)

    # 每个样本权重：w_i = 1 / (n_{y_i} ^ alpha)
    denom = np.power(counts[labels].astype(np.float32), alpha)
    denom = np.maximum(denom, 1e-6)
    weights = (1.0 / denom).astype(np.float32)
    weights = torch.from_numpy(weights)

    if num_samples is None:
        num_samples = len(idx_list)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=replacement
    )
    return sampler, counts


def evaluate_gmean_optional(data_iter, backbone, fcs, method, fusion_net=None, device=None, if_get_y=False, if_get_logits=False):
    if device is None and isinstance(backbone, torch.nn.Module):
        device = list(backbone.parameters())[0].device
    y_hat = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    logits = torch.tensor([]).to(device)
    cat_dim = 1 if method == 'normal' else 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            backbone.eval() # open evaluate mode
            features = backbone(X)
            y_hat = torch.cat((y_hat, cal_y_hat(fcs, features, method, device, fusion_net)), cat_dim)
            if method == 'weight_averaging' and if_get_logits:
                _, lo = cal_y_hat(fcs, features, method, device, fusion_net, if_get_logits=True)
                logits = torch.cat((logits, lo), cat_dim)
        if method == 'normal':
            gmean = [cal_gmean(y_true, y_hat[i]) for i in range(len(y_hat))]
        else:
            gmean = cal_gmean(y_true, y_hat)
    if method == 'weight_averaging' and if_get_logits:
        return gmean, y_hat, y_true, logits
    if if_get_y:
        return gmean, y_hat, y_true
    return gmean


def evaluate_acc_optional(data_iter, backbone, fcs, method, fusion_net=None, device=None):
    if device is None and isinstance(backbone, torch.nn.Module):
        device = list(backbone.parameters())[0].device
    y_hat = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            backbone.eval()  # open evaluate mode
            features = backbone(X)
            y_hat = torch.cat((y_hat, cal_y_hat(fcs, features, method, device, fusion_net)), 0)
        acc_sum = (y_hat == y_true.to(device)).float().sum().cpu().item()
        acc_ave = acc_sum / len(y_true)
    return acc_ave

def remove_fold_results(file_dir, p, k):
    """Remove existing #p{p}-k{k} block so re-runs do not duplicate or leave stale first blocks."""
    marker = f'#p{p}-k{k}\n'

    def _filter(path):
        if not os.path.isfile(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        out = []
        i = 0
        while i < len(lines):
            if lines[i] == marker:
                i += 2
                continue
            out.append(lines[i])
            i += 1
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(out)

    _filter(os.path.join(file_dir, 'y_hat.txt'))
    _filter(os.path.join(file_dir, 'y_true.txt'))
    _filter(os.path.join(file_dir, 'logits.txt'))


def write_result_to_file(file_dir, y_hat, y_true, logits=None, p=0, k=0):
    y_hat_path = os.path.join(file_dir, 'y_hat.txt')
    y_true_path = os.path.join(file_dir, 'y_true.txt')
    f_y_hat = open(y_hat_path, 'a')
    f_y_true = open(y_true_path, 'a')
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    f_y_hat.write(f'#p{p}-k{k}\n')
    for y_h in y_hat:
        f_y_hat.write(str(y_h.astype(np.int32)) + ' ')
    f_y_hat.write('\n')

    f_y_true.write(f'#p{p}-k{k}\n')
    for y_t in y_true:
        f_y_true.write(str(y_t.astype(np.int32)) + ' ')
    f_y_true.write('\n')

    if logits is not None and len(logits):
        logits_path = os.path.join(file_dir, 'logits.txt')
        f_logits = open(logits_path, 'a')
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        f_logits.write(f'#p{p}-k{k}\n')
        logits = np.around(logits, 2)
        for lo in logits:
            for l in lo:
                f_logits.write(str(l) + ' ')
            f_logits.write(',')
        f_logits.write('\n')

    print(f'Write result to file {file_dir} finished!')


# def read_result_from_file(file_dir, p, k, if_get_logits=False):
#     y_hat_path = os.path.join(file_dir, 'y_hat.txt')
#     y_true_path = os.path.join(file_dir, 'y_true.txt')
#
#     f_y_hat = open(y_hat_path, 'r')
#     f_y_true = open(y_true_path, 'r')
#
#     y_hat_lines = f_y_hat.readlines()
#     y_true_lines = f_y_true.readlines()
#
#     y_hat_line = None
#     y_true_line = None
#
#     for i in range(len(y_hat_lines)):
#         if y_hat_lines[i][0] == '#':
#             str_cuts = y_hat_lines[i].rstrip().split('-')
#             p_read = int(str_cuts[0][2:])
#             k_read = int(str_cuts[1][1:])
#             if p_read == p and k_read == k:
#                 y_hat_line = y_hat_lines[i + 1]
#                 break
#     for i in range(len(y_true_lines)):
#         if y_true_lines[i][0] == '#':
#             str_cuts = y_true_lines[i].rstrip().split('-')
#             p_read = int(str_cuts[0][2:])
#             k_read = int(str_cuts[1][1:])
#             if p_read == p and k_read == k:
#                 y_true_line = y_true_lines[i + 1]
#                 break
#     y_hat_line = [int(str) for str in y_hat_line.rstrip().split(' ')]
#     y_true_line = [int(str) for str in y_true_line.rstrip().split(' ')]
#
#     if if_get_logits:
#         logits_path = os.path.join(file_dir, 'logits.txt')
#         f_logits = open(logits_path, 'r')
#         logits_lines = f_logits.readlines()
#         logits_line = None
#         for i in range(len(logits_lines)):
#             if logits_lines[i][0] == '#':
#                 str_cuts = logits_lines[i].rstrip().split('-')
#                 p_read = int(str_cuts[0][2:])
#                 k_read = int(str_cuts[1][1:])
#                 if p_read == p and k_read == k:
#                     logits_line = logits_lines[i + 1]
#                     break
#         logits_line = [[float(s) for s in str.rstrip().split(' ')] for str in
#                        logits_line.rstrip().rstrip(',').split(',')]
#         return y_hat_line, y_true_line, logits_line
#     return y_hat_line, y_true_line

def read_result_from_file(file_dir, p, k, if_get_logits=False):
    def _find_block_line(lines, file_path, p, k):
        target_line = None
        for i in range(len(lines)):
            line = lines[i].strip()
            if not line or line[0] != '#':
                continue

            str_cuts = line.split('-')
            if len(str_cuts) != 2:
                continue

            try:
                p_read = int(str_cuts[0][2:])
                k_read = int(str_cuts[1][1:])
            except ValueError:
                continue

            if p_read == p and k_read == k:
                if i + 1 >= len(lines):
                    raise ValueError(
                        f'Missing content line after marker #p{p}-k{k} in {file_path}'
                    )
                target_line = lines[i + 1].strip()
                break

        if target_line is None:
            raise ValueError(
                f'Cannot find block #p{p}-k{k} in {file_path}'
            )
        if target_line == '':
            raise ValueError(
                f'Empty content for block #p{p}-k{k} in {file_path}'
            )
        return target_line

    y_hat_path = os.path.join(file_dir, 'y_hat.txt')
    y_true_path = os.path.join(file_dir, 'y_true.txt')

    with open(y_hat_path, 'r', encoding='utf-8') as f_y_hat:
        y_hat_lines = f_y_hat.readlines()
    with open(y_true_path, 'r', encoding='utf-8') as f_y_true:
        y_true_lines = f_y_true.readlines()

    y_hat_line = _find_block_line(y_hat_lines, y_hat_path, p, k)
    y_true_line = _find_block_line(y_true_lines, y_true_path, p, k)

    y_hat_line = [int(s) for s in y_hat_line.split() if s]
    y_true_line = [int(s) for s in y_true_line.split() if s]

    if len(y_hat_line) != len(y_true_line):
        raise ValueError(
            f'Length mismatch at #p{p}-k{k}: '
            f'len(y_hat)={len(y_hat_line)}, len(y_true)={len(y_true_line)}'
        )

    if if_get_logits:
        logits_path = os.path.join(file_dir, 'logits.txt')
        with open(logits_path, 'r', encoding='utf-8') as f_logits:
            logits_lines = f_logits.readlines()

        logits_line = _find_block_line(logits_lines, logits_path, p, k)
        logits_line = [
            [float(s) for s in row.split() if s]
            for row in logits_line.rstrip(',').split(',')
            if row.strip()
        ]

        if len(logits_line) != len(y_true_line):
            raise ValueError(
                f'Length mismatch at #p{p}-k{k}: '
                f'len(logits)={len(logits_line)}, len(y_true)={len(y_true_line)}'
            )

        return y_hat_line, y_true_line, logits_line

    return y_hat_line, y_true_line
