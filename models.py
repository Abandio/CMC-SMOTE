import torch.nn as nn
import torch
import torch.nn.functional as F

class view_specific_encoder(nn.Module):
    def __init__(self, in_fea, out_fea, hidden_fea):
        super(view_specific_encoder, self).__init__()
        # 初始化一个空的模块列表
        layers = []
        # 第一层输入层
        layers.append(nn.Linear(in_fea, hidden_fea[0]))
        layers.append(nn.BatchNorm1d(hidden_fea[0]))
        layers.append(nn.ReLU())
        # 中间的隐藏层，使用 for 循环
        for i in range(1, len(hidden_fea)):
            layers.append(nn.Linear(hidden_fea[i-1], hidden_fea[i]))  # 从上一层到当前层的线性变换
            layers.append(nn.BatchNorm1d(hidden_fea[i]))  # 批归一化
            layers.append(nn.ReLU())  # 激活函数
        # 输出层
        layers.append(nn.Linear(hidden_fea[-1], out_fea))  # 最后一层的输出
        layers.append(nn.BatchNorm1d(out_fea))
        layers.append(nn.ReLU())

        # 将所有的层组合成一个 Sequential 模块
        self.encoder = nn.Sequential(*layers)

    def forward(self, xv):
        return self.encoder(xv)
    
class view_specific_decoder(nn.Module):
    def __init__(self, in_fea, out_fea, hidden_fea):
        super(view_specific_decoder, self).__init__()

        # 反转 hidden_fea，以便解码器层按反向顺序构建
        hidden_fea = hidden_fea[::-1]
        
        # 初始化一个空的模块列表
        layers = []

        # 第一层输入
        layers.append(nn.Linear(out_fea, hidden_fea[0]))
        layers.append(nn.BatchNorm1d(hidden_fea[0]))
        layers.append(nn.ReLU())
        
        # 中间的隐藏层，使用 for 循环
        for i in range(1, len(hidden_fea)):
            layers.append(nn.Linear(hidden_fea[i-1], hidden_fea[i]))  # 从上一层到当前层的线性变换
            layers.append(nn.BatchNorm1d(hidden_fea[i]))  # 批归一化
            layers.append(nn.ReLU())  # 激活函数

        # 输出层
        layers.append(nn.Linear(hidden_fea[-1], in_fea))  # 最后一层的输出

        # 将所有的层组合成一个 Sequential 模块
        self.decoder = nn.Sequential(*layers)

    def forward(self, zv):
        return self.decoder(zv)

class ClusterProject(nn.Module):
    def __init__(self, out_fea,n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = out_fea
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),

        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, z):
        h = self.cluster_projector(z)
        y = self.cluster(h)
        return y
class AttentionLayer(nn.Module):
    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self._latent_dim = config['out_fea']
        self.n_views = config['n_views']
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * self.n_views, self._latent_dim * self.n_views),
            nn.BatchNorm1d(self._latent_dim * self.n_views),
            nn.ReLU(),
            nn.Linear(self._latent_dim * self.n_views, self._latent_dim * self.n_views),
            nn.BatchNorm1d(self._latent_dim * self.n_views),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self._latent_dim * self.n_views, self.n_views, bias=True)

    def forward(self, zs, tau=10.0):
        h = torch.cat(zs, dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1)
        # weights = torch.mean(e, dim=0)
        # h = weights[0] * h1 + weights[1] * h2
        # 对于多个视图的情况：
        p = 0
        for idx, z in enumerate(zs):
            p += e[:, idx].unsqueeze(1) * z  # 对每个视图加权求和
        
        return p
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_views = config['n_views']
        self.encoders = nn.ModuleList(
            [view_specific_encoder(in_fea, config['out_fea'],hidden_fea) for in_fea,hidden_fea in zip(config['in_feas'],config['hidden_feas'])]
        )
        self.decoders = nn.ModuleList(
            [view_specific_decoder(in_fea, config['out_fea'],hidden_fea) for in_fea,hidden_fea in zip(config['in_feas'],config['hidden_feas'])]
        )
        self.clusters = nn.ModuleList(
            [ClusterProject(config['out_fea'],config['n_clusters']) for _ in range(config['n_views'])]
        )
        # 并确保类型为 float32
        fusion_weights = torch.ones(config['n_views'], 1, config['n_clusters'], dtype=torch.float32)
        # 使用 nn.Parameter 将其转化为模型的可训练参数
        self.fusion_weights = nn.Parameter(fusion_weights)



    def forward(self, xs):  
        # 编码每个视图
        zs = [self.encoders[v](xs[v]) for v in range(self.num_views)]
        # 得到伪标签
        ys = [self.clusters[v](zs[v]) for v in range(self.num_views)]
        # 重构每个视图
        rxs = [self.decoders[v](zs[v]) for v in range(self.num_views)]
        
        # 将 ys 转换为张量，形状为 [n_views, batch_size, n_clusters]
        ys_stack = torch.stack(ys, dim=0)  # 形状为 (n_views, batch_size, n_clusters)
        
        # 使用 fusion_weights 对 ys 进行加权
        weighted_ys = ys_stack * self.fusion_weights  # 广播相乘，得到 (n_views, batch_size, n_clusters)
        
        # 沿着第一个维度（视图）聚合，加权后得到的 uni_y
        uni_y = weighted_ys.sum(dim=0)  # 结果的形状为 (batch_size, n_clusters)
        return zs, rxs, ys, uni_y



