from utils import InstanceLoss, ClusterLoss, ReconstructionLoss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval import eval
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

EPS = sys.float_info.epsilon

def train(imbalance_dataset, model, optim,config,is_eval=True):
    # 创建 DataLoader 实例
    dataloader = DataLoader(
        imbalance_dataset,
        batch_size=config['bs'],
        shuffle= False,
        drop_last=False,
        pin_memory=True
    )
    
    device = config['device']
    criterion_instance = InstanceLoss().to(device)
    criterion_cluster = ClusterLoss(config).to(device)
    criterion_rec = ReconstructionLoss().to(device)
    pbar = tqdm(range(config['train_epochs']), desc='Training Progress', leave=True)  # 只在外层显示训练进度条
    acc_list, nmi_list, ari_list, pur_list = [], [], [], []
    total_losses = []
    for epoch in pbar:
        model.train()
        train_loss = []

        # 通过 tqdm 处理训练数据
        for _, (xs, _) in enumerate(dataloader):
            ins_loss = 0.0
            clu_loss = 0.0
            rec_loss = 0.0
            
            # 数据转移到设备
            xs = [x.to(device) for x in xs]
            
            # 梯度清零
            optim.zero_grad()
            
            # 前向传播
            zs, rxs, ys, common_y = model(xs)


            def target_l2(q):
                return ((q ** 2).t() / (q ** 2).sum(1)).t()

            # 计算多视图损失
            n_views = config['n_views']

            y_max = ys[0]
            for i in range(1,n_views):
                y_max = torch.maximum(y_max,ys[i])
            y_max = torch.maximum(y_max, common_y)
            y_max = target_l2(y_max)
            common_y = torch.where(common_y < EPS, torch.tensor([EPS], device=common_y.device), common_y)
            hc_loss = F.kl_div(common_y.log(), y_max.detach(), reduction='batchmean')


            for i in range(n_views):
                rec_loss += criterion_rec(xs[i], rxs[i])
                
                


                for j in range(i,n_views):
                    # ! 只对正对进行相似度矩阵，得到的是N， 大小的相似度系数
                    Dx = F.cosine_similarity(zs[i], zs[j], dim=1)
                    #! 创建一个 N， 大小的1向量
                    gt = torch.ones_like(Dx).to(device)
                    ins_loss += criterion_instance(gt, Dx)
                    clu_loss += criterion_cluster(ys[i], ys[j])

            
            # 加权总损失
            epoch_loss = rec_loss + config['alpha']*ins_loss + config['beta']*clu_loss + config['gamma'] * hc_loss
            
            # 反向传播
            epoch_loss.backward()
            optim.step()
            
            # 记录损失值
            batch_total = epoch_loss.item()

            
            train_loss.append(batch_total)
                # 保存train_loss为mat

        if is_eval:
            # 每隔 interval 个 epoch 进行评估
            if (epoch + 1) % config['interval'] == 0:
                eval_results = eval(model, imbalance_dataset, config)
                acc_list.append(eval_results[0])
                nmi_list.append(eval_results[1])
                ari_list.append(eval_results[2])
                pur_list.append(eval_results[3])
        
        # 记录每个epoch的总损失
        epoch_total_loss = sum(train_loss) / len(train_loss) if train_loss else 0
        total_losses.append(epoch_total_loss)
        
    # 根据is_eval返回不同的结果
    if is_eval:
        # 训练结束时记录所有评估结果
        print("#------------------Training Completed------------------#")
        print("Evaluation Results over all epochs:")
        print(f"ACC List: {acc_list}")
        print(f"NMI List: {nmi_list}")
        print(f"ARI List: {ari_list}")
        print(f"PUR List: {pur_list}")
        
        # 保存模型检查点
        if config['save_model']:
            save_path = f"{config['checkpoint_dir']}/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': epoch_loss,
            }, save_path)
            
        return {
            'acc': acc_list,
            'nmi': nmi_list,
            'ari': ari_list,
            'pur': pur_list,
            'interval': config['interval']
        }
    else:
        return total_losses




def plot_convergence(pretrain_metrics, train_metrics, save_dir, dataset_name):
    """绘制预训练和训练阶段的收敛曲线
    
    参数:
        pretrain_metrics: 预训练阶段的指标
        train_metrics: 训练阶段的指标
        save_dir: 保存图像的目录
        dataset_name: 数据集名称，用于保存文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    metrics = ['acc', 'nmi', 'ari', 'pur']
    titles = ['Accuracy (ACC)', 'Normalized Mutual Info (NMI)',
              'Adjusted Rand Index (ARI)', 'Purity (PUR)']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        # 获取预训练和训练阶段的数据
        pretrain_vals = pretrain_metrics[metric]
        train_vals = train_metrics[metric]
        
        # 计算x轴坐标
        pretrain_x = np.arange(0, len(pretrain_vals)) * pretrain_metrics['interval']
        train_x = np.arange(0, len(train_vals)) * train_metrics['interval'] + (pretrain_x[-1] + pretrain_metrics['interval'] if len(pretrain_x) > 0 else 0)
        
        # 绘制预训练阶段（黄色系）
        axes[idx].plot(pretrain_x, pretrain_vals, 
                      color='goldenrod', linewidth=2, label='Pretraining')
        
        # 绘制训练阶段（绿色系）
        axes[idx].plot(train_x, train_vals, 
                      color='darkgreen', linewidth=2, label='Training')
        
        # 添加分界线
        if len(pretrain_x) > 0:
            split_x = pretrain_x[-1] + pretrain_metrics['interval']/2
            axes[idx].axvline(x=split_x, color='gray', linestyle='--', alpha=0.7)
        
        # 设置图表标题和标签
        axes[idx].set_title(title, fontsize=14)
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
        
        # 设置y轴范围
        all_vals = pretrain_vals + train_vals
        if all_vals:
            min_val = max(0, min(all_vals) - 0.1)
            max_val = min(1, max(all_vals) + 0.1)
            axes[idx].set_ylim(min_val, max_val)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'{dataset_name}_convergence.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved convergence plot to {save_path}")
            

