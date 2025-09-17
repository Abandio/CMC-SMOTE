import math
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
from collections import Counter


class CompleteDataset(Dataset):
    def __init__(self, data_list, labels=None):
        """
        初始化多视图完整数据集
        :param data_list: 多视图数据列表
        :param labels: 标签（可选）
        """
        self.data = [torch.tensor(view_data, dtype=torch.float32) for view_data in data_list]
        self.n_views = len(data_list)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data[0])

    def __getitem__(self, index):
        """
        返回指定索引的多视图数据
        """
        data = [self.data[v][index] for v in range(self.n_views)]
        if self.labels is not None:
            label = self.labels[index]
            return data, label
        else:
            return data
class SingleviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


class Incomplete_MultiviewDataset(Dataset):
    def __init__(self, data_list, mask_matrix, labels, n_views):
        self.n_views = n_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, n_views, axis=0)

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.n_views)]
        mask = [torch.tensor(self.mask_list[v][0][index], dtype=torch.float32, requires_grad=False) for v in range(self.n_views)]
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return data, mask, label
        else:
            return data, mask


def get_mask(n_samples,view_mr, mr):
    """
    view_mr: 平均缺失率
    mr(list): 视图缺失率 
    视图idx缺失率 = view_mr * mr[idx]
    """

    masks = []

    for view in range(len(view_mr)):

        miss_sample_num =  math.floor( n_samples * (view_mr[view] * mr))

        data_ind = list(range(n_samples))
        random.shuffle(data_ind)
        miss_ind = data_ind[:miss_sample_num]
        mask = np.ones([n_samples])

        mask[miss_ind] = 0
        masks.append(mask)
        
    return np.array(masks,dtype=np.int32)




def load_multiview_data(config):
    X_list = []
    
    main_dir = "T:\\Files\\projs\\datasets\\转换后的数据\\"

    mat = sio.loadmat(os.path.join(main_dir, config['dataname'] + '.mat'))
    n_views = mat['X'].shape[1]
    config['n_views'] = n_views
    for idx in range(n_views):
        if config['toArray']:
            X_list.append(mat['X'][0][idx].astype('float32').toarray())
        else:
            X_list.append(mat['X'][0][idx].astype('float32'))

    scaler = MinMaxScaler()
    for idx in range(n_views):
        X_list[idx] = scaler.fit_transform(X_list[idx])

    labels = np.array(mat['Y'].squeeze()).astype(np.int32)
    
        
    dims = [xv.shape[1] for xv in X_list]
    n_samples = labels.shape[0]
    n_clusters = len(np.unique(labels))
    if np.max(labels) == n_clusters:
        labels = labels - 1
    config['in_feas'] = dims
    config['n_clusters'] = n_clusters
    config['n_samples'] = n_samples
    tmp = config['hidden_feas']
    config['hidden_feas'] = [tmp for _ in range(config['n_views'])]
    return X_list, labels


def pixel_normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def generate_imbalanced_data(data_list, labels, seed,type='argmax', imbalance_factor_range=(0.1, 1.0)):
    """
    通过递减的方式调整类别不平衡，生成不平衡的数据。
    
    data_list: 包含多个视图的数据列表（每个视图为一个 numpy 数组）
    labels: 训练数据的标签
    imbalance_factor_range: 不平衡系数的范围, 默认为(0.1, 1.0)
    """
    np.random.seed(seed)
    # 统计每个类别的样本数
    class_counts = Counter(labels)
    
    if(type == 'argmax'):
        # 获取类别及其对应的样本数量，并按数量排序
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        sorted_class_labels, sorted_class_counts = zip(*sorted_classes)
    elif(type == 'random'):
        # 将类别和对应样本数转换为列表
        class_items = list(class_counts.items())  # [(label1, count1), (label2, count2), ...]

        # 随机打乱这些项的顺序
        random_indices = np.random.permutation(len(class_items))
        shuffled_items = [class_items[i] for i in random_indices]

        # 解压成两个列表，确保一一对应
        sorted_class_labels, sorted_class_counts = zip(*shuffled_items)
            
    # 计算每个类别的不平衡系数
    n_classes = len(sorted_class_counts)
    imbalance_factors = np.linspace(imbalance_factor_range[0], imbalance_factor_range[1], n_classes)

    # 生成新的标签数量
    new_class_counts = []
    for i, count in enumerate(sorted_class_counts):
        imbalance_factor = imbalance_factors[i]
        new_class_count = int(count * imbalance_factor)  # 通过不平衡系数调整样本数量
        new_class_counts.append(new_class_count)

    # 根据新样本数量生成不平衡数据集
    new_data = []
    new_labels = []

    # 对每个类别进行采样
    for i, label in enumerate(sorted_class_labels):
        class_data = [data[labels == label] for data in data_list]  # 获取每个视图该类的样本
        class_size = new_class_counts[i]
        
        # 对每个视图中的样本进行采样
        selected_data = []
        for data in class_data:
            selected_indices = np.random.choice(len(data), class_size, replace=False)
            selected_data.append(data[selected_indices])
        
        new_data.append(selected_data)
        new_labels.extend([label] * class_size)

    # 转换成 numpy 数组并合并
    new_data = [np.concatenate(view_data, axis=0) for view_data in zip(*new_data)]
    new_labels = np.array(new_labels)

    return new_data, new_labels



# 示例：生成不平衡数据
def get_data(config):
    data_list, labels = load_multiview_data(config)  # 加载多视图数据
   

    return data_list, labels
    

