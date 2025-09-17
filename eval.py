
import torch
import numpy as np
from Nmetrics import evaluate
from torch.utils.data import DataLoader


def eval(model, imbalance_dataset, config, return_probs=False):
    # 创建 DataLoader 实例
    dataloader = DataLoader(
        imbalance_dataset,
        batch_size=config['bs'],
        shuffle=True,  
        drop_last=False,        
        pin_memory=True
    )
    
    device = config['device']
    model.eval()  # 设置模型为评估模式
    
    # 存储预测结果、概率和真实标签
    all_pred = []
    all_probs = []
    all_labels = []
    all_common = []
    with torch.no_grad():  # 评估时不需要计算梯度
        for batch_idx, (xs, y) in enumerate(dataloader):
            xs = [x.to(device) for x in xs]

            # 前向传播
            _, _, _, common_y = model(xs)


            # 存储预测概率和标签
            probs = torch.softmax(common_y, dim=1)
            all_probs.append(probs.cpu().detach().numpy())
            all_labels.append(y)
            all_common.append(common_y.cpu().detach().numpy())
    # 将批次结果堆叠成一个数组
    all_probs = np.vstack(all_probs)
    all_pred = all_probs.argmax(axis=1)
    all_labels = np.concatenate(all_labels)
    all_common = np.concatenate(all_common, axis=0) 
    
    # 调用评估函数
    results,true,y_pred_ajusted = evaluate(all_labels, all_pred)
    
    if return_probs:
        return results, true,y_pred_ajusted
    return results,all_common,all_pred
