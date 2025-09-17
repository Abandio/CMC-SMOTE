
import torch
import numpy as np
from Nmetrics import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_pesudo(model, cmv_dataset, config):
    # 创建 DataLoader 实例
    dataloader = DataLoader(
        cmv_dataset,
        batch_size=config['bs'],
        shuffle=False,  
        drop_last=False,
        pin_memory=True
    )
    
    device = config['device']
    model.eval()  # 设置模型为评估模式
    
    # 存储预测结果和真实标签
    all_pred = []
    
    with torch.no_grad():  # 评估时不需要计算梯度
        
        for batch_idx, (xs,_) in enumerate(dataloader):
            xs = [x.to(device) for x in xs]

            # 前向传播
            _, _, _, common_y = model(xs)

            # 存储预测结果和标签
            all_pred.append(common_y.cpu().detach().numpy())
            
    
    # 将批次结果堆叠成一个数组
    all_pred = np.vstack(all_pred)
    all_pred = all_pred.argmax(1)
    

    

    return all_pred


