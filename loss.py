import torch

def elastic_loss(x, rx,delta=5):
    # 计算 y_true 和 y_pred 的平方误差
    a = (x - rx) ** 2
    # 计算部分损失 b，考虑 delta 和平方误差 a 的缩放比
    b = (delta * a) / (delta + a)
    # 计算部分损失 c，使用一种不同的缩放方式
    c = a / (a + delta)
    # 将 b 和 c 相加，得到最终的损失项
    d = b + c
    # 将所有的 d 元素求和，得到总的损失值
    el_loss = torch.sum(d)
    return el_loss