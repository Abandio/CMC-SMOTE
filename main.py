
from utils import smote 
from config import init_config 
from datasets import get_data,generate_imbalanced_data 
from models import Model 
import torch
import os

from train import train
from datasets import CompleteDataset
from get_pesudo import get_pesudo
from eval import eval
from utils import setup_seed,visualize_tsne
import numpy as np
import scipy.io
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # config初始化
    """
    0: MSRC-v2
    1: Scene15
    2: NoisyMNIST30000
    3: CCV
    4: NUSWIDEOBJ
    5: Caltech256_fea
    6: 100leaves
    7: COIL20
    8: 3sources
    9: NGs
    10: handwritten
    11: BBCSport
    12: Caltech101-7
    13: YaleB
    """

    # for dataid in [7]:
    for dataid in [0]:
        saved_results = []
        config = init_config(dataid)
        data,labels = get_data(config)
        # 设备选择（CUDA 或 CPU）
        device_str = "cuda:0"
        device = torch.device(device_str)
        # 设置 CUDA 可见设备，确保 GPU 的选择与 device 一致m
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config['device'] = device
        setup_seed(config['seed'])
        # 设定数据不平衡
        imbalance_data, imbalance_labels = generate_imbalanced_data(data, labels,config['seed'],'argmax')



        print("#--------------------PRETRAINING--------------------#")
        # 模型搭建
        model = Model(config)
        model = model.to(device) 
        pre_optim = torch.optim.Adam(model.parameters(),lr=config['train_lr'])

        imbalance_dataset = CompleteDataset(imbalance_data, imbalance_labels)
        #dual pretrain 首先用不平衡数据训练，之后用smote构造的平衡数据训练
        pretrain_loss = train(imbalance_dataset, model, pre_optim, config,is_eval=False)

        #! smote数据生成
        print("#--------------------CONSTRUCT smote_dataset--------------------#")
        pesudo_label = get_pesudo(model, imbalance_dataset, config)
        smote_data, smote_label = smote(imbalance_data, pesudo_label, config)
        if smote_data == None and smote_label == None:
            continue
            
        #! smote_dataset生成 
        print("#--------------------TRAINING--------------------#")
        smote_dataset = CompleteDataset(smote_data, smote_label)
        
        # 训练模型并获取损失值
        train_loss = train(smote_dataset, model, pre_optim, config, is_eval=False)
        
        # 绘制损失曲线
        print("#--------------------PLOTTING LOSS CURVE--------------------#")
        plt.figure(figsize=(16, 8))
        
        # 绘制预训练阶段（黄色系）
        plt.plot(pretrain_loss, label='Pretraining Loss', color='goldenrod', linewidth=2)
        
        # 绘制训练阶段（绿色系）
        plt.plot(range(len(pretrain_loss), len(pretrain_loss) + len(train_loss)), 
                train_loss, label='Training Loss', color='darkgreen', linewidth=2)
        
        # 添加分界线
        plt.axvline(x=len(pretrain_loss)-0.5, color='gray', linestyle='--', alpha=0.7)
        
        # 添加分界线标签
        plt.text(len(pretrain_loss)/2, plt.ylim()[1]*0.95, 'Pretraining', 
                ha='center', va='top', fontsize=12, fontweight='bold', color='darkorange')
        plt.text(len(pretrain_loss) + len(train_loss)/2, plt.ylim()[1]*0.95, 'Training', 
                ha='center', va='top', fontsize=12, fontweight='bold', color='darkgreen')
        
        # 设置图表样式
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f"{config['dataname']} - Pretraining and Training Loss Curve", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 确保目录存在
        os.makedirs('visualization/loss_curves', exist_ok=True)
        
        # 保存图像
        save_path = f"visualization/loss_curves/{config['dataname']}_loss_curve.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved loss curve to {save_path}")
        
        #! tsne可视化
        print("#--------------------VISUALIZATION--------------------#")
        # 确保输出目录存在
        os.makedirs('visualization/tsne', exist_ok=True)
        
        # 准备数据
        print("Preparing combined visualization...")
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 合并原始和SMOTE数据
        combined_data = np.vstack([imbalance_data[0], smote_data[0]])
        
        # 使用t-SNE降维
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
        
        print("Fitting t-SNE...")
        combined_tsne = tsne.fit_transform(combined_data)
        
        # 分离原始数据和SMOTE数据
        n_original = len(imbalance_data[0])
        original_tsne = combined_tsne[:n_original]
        smote_tsne = combined_tsne[n_original:]
        
        # 创建可视化
        plt.figure(figsize=(14, 12))
        plt.style.use('default')
        
        # 获取唯一的标签和颜色
        unique_labels = np.unique(np.concatenate([imbalance_labels, smote_label]))
        palette = sns.color_palette("husl", len(unique_labels))
        
        # 绘制原始数据（浅色）
        for label in unique_labels:
            idx = imbalance_labels == label
            plt.scatter(
                original_tsne[idx, 0], original_tsne[idx, 1],
                label=f'Original {label}',
                color=palette[label],
                alpha=1,  # 半透明
                s=80,       # 较小点
                marker='x',  # 不同标记
            )
        
        # 绘制SMOTE数据（深色）
        for label in unique_labels:
            idx = smote_label == label
            plt.scatter(
                smote_tsne[idx, 0], smote_tsne[idx, 1],
                label=f'SMOTE {label}',
                color=palette[label],
                alpha=0.5,  # 不透明
                s=30,       # 较大点
                linewidth=1.5,
                edgecolors='none'
            
            )
        
        plt.title(f'{config["dataname"]} - Original vs SMOTE Data (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE dimension 1', fontsize=14)
        plt.ylabel('t-SNE dimension 2', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加图例（每个类别只显示一个图例项）
        handles, labels1 = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels1, handles))
        plt.legend(unique_labels.values(), unique_labels.keys(),
                  title='Class',
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  fontsize=10)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = f"visualization/tsne/{config['dataname']}_combined_tsne.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved combined visualization to {save_path}")

        print("#--------------------TESTING--------------------#")

        # 初始化结果列表
        all_results = []

        # 创建可视化目录
        os.makedirs('visualization/heatmaps', exist_ok=True)
        
        for i in range(config['test_times']):
            # 生成测试集
            test_data, test_labels = generate_imbalanced_data(
                data, labels, seed=config['seed'] + i, type='random'
            )
            test_dataset = CompleteDataset(test_data, test_labels)
            
            visualize_tsne(test_data[0], test_labels, config['dataname'], f"visualization/tsne/{config['dataname']}_test_tsne.png")
            eval_results,common_y,all_labels = eval(model, test_dataset, config)
            visualize_tsne(common_y, all_labels, config['dataname'], f"visualization/tsne/{config['dataname']}_common_tsne.png")
            
            # 保存评估结果
            all_results.append(eval_results)
            acc, nmi, ari, pur, p, r, f = eval_results
            print(f'Test {i+1}: ACC = {acc:.4f} ; NMI = {nmi:.4f} ; ARI = {ari:.4f} ; PUR = {pur:.4f} ; P = {p:.4f} ; R = {r:.4f} ; F = {f:.4f}')

        
        # 转换为 numpy 数组，便于计算均值
        all_results = np.array(all_results)  # shape: (test_times, 4)
        mean_results = np.mean(all_results, axis=0)
        std_results = np.std(all_results, axis=0)


        mean_acc, mean_nmi, mean_ari, mean_pur, mean_p, mean_r, mean_f = mean_results
        std_acc, std_nmi, std_ari, std_pur,std_p,std_r,std_f = std_results

        print("=" * 60)
        print(f"{'ACC':<12}{'NMI':<12}{'ARI':<12}{'PUR':<12}{'P':<12}{'R':<12}{'F':<12}")
        print(f"{mean_acc * 100:.2f}+/-{std_acc * 100:.2f}  "
                f"{mean_nmi * 100:.2f}+/-{std_nmi * 100:.2f}  "
                f"{mean_ari * 100:.2f}+/-{std_ari * 100:.2f}  "
                f"{mean_pur * 100:.2f}+/-{std_ari * 100:.2f}  "
                f"{mean_p * 100:.2f}+/-{std_p * 100:.2f}  "
                f"{mean_r * 100:.2f}+/-{std_r * 100:.2f}  "
                f"{mean_f * 100:.2f}+/-{std_f * 100:.2f}  ")
        print("=" * 60)
        # 准备所有参数，包括配置和结果
        params = [
            # 基础配置
            config['seed'],
            # 超参数
            config['alpha'],
            config['beta'],
            config['gamma'],
            # 评估结果
            mean_acc, std_acc,
            mean_nmi, std_nmi,
            mean_ari, std_ari,
            mean_pur, std_pur,
            mean_p, std_p,
            mean_r, std_r,
            mean_f, std_f
        ]
        
        # 添加当前参数行到结果列表
        saved_results.append(params)

        # 保存结果到文件
        dataname = config['dataname']
        # 转换为numpy数组并确保是2D
        results_array = np.array(saved_results, dtype=object)
        scipy.io.savemat(f'./results/{dataname}_results.mat', {'results': results_array})

