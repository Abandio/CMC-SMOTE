import random
import numpy as np
from scipy.io import savemat
import datetime
import os
import math
import torch
from torch import nn
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def save_data_to_mat(file_name, synthetic_views, synthetic_labels):
    # 直接将 synthetic_views 转换为列表，确保每个元素都被正确处理
    X = np.empty(len(synthetic_views),dtype=np.object_)
    for v in range(len(synthetic_views)):
        X[v] = synthetic_views[v] 

    
    # 将 synthetic_labels 转换为 NumPy 数组
    Y = np.array(synthetic_labels)
    
    # 创建字典以保存数据
    save_dict = {
        'X': X,
        'Y': Y
    }
    
    # 保存为 .mat 文件
    savemat(file_name, save_dict)


def setup_seed(seed_n):
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


class InstanceLoss(nn.Module):
    def __init__(self):
        super(InstanceLoss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss2 = mse(gt, P)

        return Loss2


class ClusterLoss(nn.Module):
    """类簇级别的对比损失"""
    def __init__(self, config):
        super(ClusterLoss, self).__init__()
        self.class_num = config['n_clusters']
        self.temperature = config['temperature_clu']
        self.device = config['device']

        self.mask = self.mask_correlated_clusters(self.class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + alpha * ne_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()  # 创建MSE损失计算器
    
    def forward(self, x1, x2):
        # 计算两个输入之间的均方误差
        return self.criterion(x1, x2)



def smote(cmv_data, cmv_label, config):
    # 获取每个类别的样本数
    label_counts = Counter(cmv_label)
    print(f"SMOTE前类别分布: {np.bincount(cmv_label)}")

    # 检查标签中是否只有一个类别
    if len(label_counts) == 1:
        print("标签中只有一个类别，无法进行SMOTE过采样，跳过当前循环")
        return None, None  # 返回 None，表示跳过当前循环处理
    if min(label_counts.values()) == 1:
        print("存在某个标签只有一个样本，无法达到最低2邻居要求，跳过当前循环")
        return None, None  # 返回 None，表示跳过当前循环处理

    # 提取每个视图的数据
    X_views = [x for x in cmv_data]  # 假设 cmv_data 是一个列表

    # 合并所有视图的数据
    X_combined = np.hstack(X_views)  # (samples, total_features)

    # 获取每个类别的样本数
    min_samples = min(label_counts.values())

    # 安全地设置 k_neighbors
    k_neighbors = config.get('k_neighbors', 5)
    safe_k = min(k_neighbors, min_samples - 1)
    safe_k = max(safe_k, 1)  # SMOTE 至少需要 k=1

    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=safe_k)

    # 对合并后的数据集进行 SMOTE 过采样处理
    X_combined_smote, Y_smote = smote.fit_resample(X_combined, cmv_label)

    # 打印合成样本的类别分布，确保同步
    print(f"SMOTE后类别分布: {np.bincount(Y_smote)}")

    # 拆分合成后的数据为每个视图
    X_views_smote = []
    start_idx = 0
    for i in range(config['n_views']):
        # 每个视图的特征数
        n_features_per_view = X_views[i].shape[1]
        # 获取每个视图的合成样本
        X_view_smote = X_combined_smote[:, start_idx:start_idx + n_features_per_view]
        X_views_smote.append(X_view_smote)
        start_idx += n_features_per_view

    # 返回合成后的每个视图的样本数据
    return X_views_smote, Y_smote

def visualize_tsne(data, labels, title, save_path):
    """
    使用t-SNE进行可视化并保存结果
    """
    # 使用perplexity参数来优化聚类效果
    tsne = TSNE(
        n_components=2,
        perplexity=50, 
        learning_rate='auto',
        n_iter=1000,
        early_exaggeration=50,
        init='random',
        random_state=3407,
        method='exact'
    )
    data_tsne = tsne.fit_transform(data)
    
    plt.figure(figsize=(12, 10))
    plt.style.use('default')  # 使用matplotlib的默认样式
    
    # 使用更鲜明的颜色方案
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # 创建散点图
    for label in unique_labels:
        idx = labels == label
        plt.scatter(data_tsne[idx, 0], data_tsne[idx, 1], 
                    label=f'Cluster {label}', 
                    color=label_to_color[label], 
                    s=50,  # 增加点的大小
                    alpha=0.7)  # 透明度
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE dimension 1', fontsize=14)
    plt.ylabel('t-SNE dimension 2', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格线
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(title='Clusters', fontsize=12, title_fontsize=14, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_prediction_heatmap(true_labels, pred_probs, title, save_path):
    """
    绘制预测概率的热力图
    
    参数:
        true_labels: 真实标签 (n_samples,)
        pred_probs: 预测概率 (n_samples, n_classes) 或者预测标签 (n_samples,)
        title: 图表标题
        save_path: 保存路径
    """
    # 如果输入的是预测标签（1D数组）
    if len(pred_probs.shape) == 1:
        pred_labels = pred_probs
    else:
        # 如果输入的是概率（2D数组）
        pred_labels = np.argmax(pred_probs, axis=1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)  # 添加小值避免除以0
    
    # 获取唯一的标签
    unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    
    # 创建热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                vmin=0, vmax=1)  # 确保颜色范围在0-1之间
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 返回混淆矩阵用于进一步分析
    return cm_normalized