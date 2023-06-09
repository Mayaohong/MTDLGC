from scipy.sparse.linalg import eigs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from logging import getLogger
from TPsource.model.abstract_traffic_state_model import AbstractTrafficStateModel
from TPsource.model import loss
from torch.nn import BatchNorm2d, Conv2d, Parameter, LayerNorm, BatchNorm1d

"""
一堆硬编码 输入数据的时间长度没法变了，必须输入维度=60，不能少
后边看看如何改他的结构 去除硬编码
"""


def scaled_laplacian(weight):
    """
    compute \tilde{L} (scaled laplacian matrix)

    Args:
        weight(np.ndarray): shape is (N, N), N is the num of vertices

    Returns:
        np.ndarray: shape (N, N)
    """
    assert weight.shape[0] == weight.shape[1]
    diag = np.diag(np.sum(weight, axis=1))
    lap = diag - weight
    lambda_max = eigs(lap, k=1, which='LR')[0].real
    return (2 * lap) / lambda_max - np.identity(weight.shape[0])


def cheb_polynomial(l_tilde, k):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Args:
        l_tilde(np.ndarray): scaled Laplacian, shape (N, N)
        k(int): the maximum order of chebyshev polynomials

    Returns:
        list(np.ndarray): cheb_polynomials, length: K, from T_0 to T_{K-1}
    """
    num = l_tilde.shape[0]
    cheb_polynomials = [np.identity(num), l_tilde.copy()]
    for i in range(2, k):
        cheb_polynomials.append(2 * l_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


class T_cheby_conv_ds(nn.Module):
    """
    x : [batch_size, feat_in, num_node ,seq_len] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    seq_len: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]

    Conv2d: the input must be 4D, batch_size x channels x height x width
    """

    def __init__(self, c_in, c_out, K, Kt, device):
        super(T_cheby_conv_ds, self).__init__()
        self.device = device
        c_in_new = K * c_in  # k*64
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape  # x:(B=nSample, c_in, N, Tw+Td+Th); adj: b,n,n

        Ls = []
        L1 = adj  # [b,n,n]
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).to(self.device)  # 重复生成Batch(nSample)个adj
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        # stack堆叠后的张量会增加一个新维度，新维度的大小等于待堆叠的张量的个数
        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        Lap = Lap.transpose(-1, -2)
        # b,c_in, n, Tw+Td+Th =l； bknn
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()  # (batch_size, c_in, K, nNode, l) 删除掉n维
        x = x.view(nSample, -1, nNode, length)  # (batch_size, c_in*K, nNode, l)
        out = self.conv1(x)  # (batch_size, c_out, nNode, l)
        return out


class SATT_3(nn.Module):
    """
    通过对两个特征矩阵进行点积，生成注意力矩阵。然后通过对注意力矩阵进行均值池化，生成每个节点的权重
    """

    def __init__(self, c_in, num_nodes):
        super(SATT_3, self).__init__()
        self.conv1 = Conv2d(c_in * 12, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_in * 12, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.bn = LayerNorm([num_nodes, num_nodes, 4])
        self.c_in = c_in

    def forward(self, seq):
        shape = seq.shape  # b,c_in,n,l[0:48]
        # input:b,c,l[0:48],n ;  out:b, c*12, l//12, n; // 表示整数除法，返回相除后向下取整的整数结果
        seq = seq.permute(0, 1, 3, 2).contiguous().view(shape[0], shape[1] * 12, shape[3]//12, shape[2])
        seq = seq.permute(0, 1, 3, 2)  # input: b, c*12, l//12, n; output:
        shape = seq.shape  # b,c*12,n, l//12

        # input:b,c,n,l[0:48]; [b, c_in//4, 4, n, l//12];  [B, n,c_in//4, l//12,4]
        f1 = self.conv1(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 3, 1, 4, 2).contiguous()
        # input:b,c,n,l[0:48]; [b, c_in//4, 4, n, l//12];  [B, c_in//4,n, l//12,4]
        f2 = self.conv2(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        # logits[b, i, j, :, m]表示 i 与 j 之间的注意力权重，对应输入张量中第 m 个时间点
        logits = torch.einsum('bnclm,bcqlm->bnqlm', f1, f2)  # b,n,n,l//12,4=m
        logits = logits.permute(0, 3, 1, 2, 4).contiguous()  # b,l//12,n,n,m
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits, -1)  # b,l//12, n, n
        return logits


class SATT_2(nn.Module):
    def __init__(self, c_in, num_nodes):
        super(SATT_2, self).__init__()
        self.conv1 = Conv2d(c_in, c_in, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_in, c_in, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.bn = LayerNorm([num_nodes, num_nodes, 12])
        self.c_in = c_in

    def forward(self, seq):
        shape = seq.shape  # [b, c_in, n, l：12]
        # b, c_in, n, l：12;        b, c//4, 4, n, l;           b, n, c//4, l, 4
        f1 = self.conv1(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 3, 1, 4, 2).contiguous()
        # b, c_in, n, l：12;        b, c//4, 4, n, l;           b, c//4, n, l, 4
        f2 = self.conv2(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        logits = torch.einsum('bnclm,bcqlm->bnqlm', f1, f2)  # b, n, n, l, 4
        logits = logits.permute(0, 3, 1, 2, 4).contiguous()  # b, l, n, n, 4
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits, -1)  # b,l, n, n
        return logits


class TATT_1(nn.Module):
    # seq_len = self.len_period + self.len_trend + self.len_closeness  # l
    def __init__(self, c_in, num_nodes, seq_len, device):
        super(TATT_1, self).__init__()
        # l = 60 硬编码；MASK
        A = np.zeros((60, 60))
        for i in range(12):
            for j in range(12):
                A[i, j] = 1
                A[i + 12, j + 12] = 1
                A[i + 24, j + 24] = 1
        for i in range(24):
            for j in range(24):
                A[i + 36, j + 36] = 1
        self.B = (-1e13) * (1 - A)  # 乱给的吧
        self.B = (torch.tensor(self.B)).type(torch.float32).to(device)
        self.device = device
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(seq_len, seq_len), requires_grad=True).to(device)
        self.v = nn.Parameter(torch.rand(seq_len, seq_len), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(seq_len)

    def forward(self, seq):
        # seq:(b, c_out, n, l=60);
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,l

        # f1*w=(b,l,c)，再与f2相乘，得(b,l,n)。
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)  # (b,l,l)+(l, l)
        logits = torch.matmul(self.v, logits)  # (b,l,l)

        # (b,l,l);  permute: b,l, l;
        logits = logits.permute(0, 2, 1).contiguous()
        #b,n,l;  permute: bln;
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits + self.B, -1)  # b,l,l; self.B is Mask matrix
        return coefs


class ST_BLOCK_2(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, seq_len, K, Kt, device):
        super(ST_BLOCK_2, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)  # 用于线性变化
        self.TATT_1 = TATT_1(c_out, num_nodes, seq_len, device)
        self.SATT_3 = SATT_3(c_out, num_nodes)
        self.SATT_2 = SATT_2(c_out, num_nodes)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt, device)
        self.LSTM = nn.LSTM(num_nodes, num_nodes, batch_first=True)  # b*n,l,n
        self.K = K
        self.seq_len = seq_len
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, seq_len])
        self.device = device

    def forward(self, x, supports):
        # feature sampling in paper.
        # x:(B, F_in, N_nodes, Tw+Td+Th)
        x_input = self.conv1(x)  # (B, c_out:64, N_nodes, Tw+Td+Th)使用1x1的卷积核，在时间和空间维不会压缩，在特征维从F_in映射到c_out
        x_1 = self.time_conv(x)  # (B, c_out, N_nodes, Tw+Td+Th)
        x_1 = F.leaky_relu(x_1)
        # feature sample, "len_closeness": 2,"len_period": 1,"len_trend": 2
        x_tem1 = x_1[:, :, :, 0:48]  # b,c,n,l[0:48]
        x_tem2 = x_1[:, :, :, 48:60]  # b,c,n,l[48:60]

        # Sequence of Laplace matrixes
        # spatial attention
        S_coef1 = self.SATT_3(x_tem1)  # b, l[0:48]//12= 4, n,n
        S_coef2 = self.SATT_2(x_tem2)  # b,l[48:60], n, n
        S_coef = torch.cat((S_coef1, S_coef2), 1)  # b,u=4+12,n,n; 相当于分开处理后合并
        shape = S_coef.shape  # b,u,n,n

        # LSTM
        h = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).to(self.device)  # 1,b*n, n
        c = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).to(self.device)  # 1,b*n, n
        hidden = (h, c)

        # b,16=u,n,n；   permute:b, n,u,n;               view: b*n, u, n
        S_coef = S_coef.permute(0, 2, 1, 3).contiguous().view(shape[0] * shape[2], shape[1], shape[3])
        S_coef = F.dropout(S_coef, 0.5, self.training)  # b*n, u, n
        # output，(h_n, c_n) = (l, b, h * num_dir), h_n：(num_layers*num_dir, b, h) ; output, (hn, cn) = rnn(input, (h0, c0))
        _, hidden = self.LSTM(S_coef, hidden)  # u为隐藏层单元的最后一个单元：hidden[0]: 1, b*n, n

        # 相当于得到了每个时间步的邻接矩阵, 形状为n,n ; 1,b,n,n; seueeze:bnc; view:b,n,n
        adj_out = hidden[0].squeeze().view(shape[0], shape[2], shape[3]).contiguous()

        adj_out1 = adj_out * supports  # [b, n, n]; L_p = L_d * L_res

        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, adj_out1)  # (batch_size, 2*c_out, nNode, l)

        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)  # (batch_size, c_out, nNode, l)
        T_coef = self.TATT_1(x_1)  # b,l,l
        T_coef = T_coef.transpose(-1, -2)  # b,l,l
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)  # bcnl 删掉l维

        out = self.bn(F.leaky_relu(x_1) + x_input)  # x1:bcnl +bcnl 残差连接
        return out, adj_out, T_coef


class DGCN(AbstractTrafficStateModel):
    """
    {"scaler": self.scaler, "adj_mx": self.adj_mx,
    "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
    "output_dim": self.output_dim, "ext_dim": self.ext_dim,
    "len_closeness": self.len_closeness * self.output_window,
    "len_period": self.len_period * self.output_window,
    "len_trend": self.len_trend * self.output_window,
    "num_batches": self.num_batches}
    """

    def __init__(self, config, data_feature):
        super(DGCN, self).__init__(config, data_feature)
        self.data_feature = data_feature
        self.c_out = config.get('c_out', 64)
        self.K = config.get('K', 3)  # adj
        self.Kt = config.get('Kt', 3)  # conv
        self.device = config.get('device', torch.device('cpu'))

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.len_period = self.data_feature.get('len_period', 1)
        self.len_trend = self.data_feature.get('len_trend', 2)
        self.len_closeness = self.data_feature.get('len_closeness', 2)
        if self.len_period == 0 and self.len_trend == 0 and self.len_closeness == 0:
            raise ValueError('Num of days/weeks/hours are all zero! Set at least one of them not zero!')
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.adj_mx = self.data_feature.get('adj_mx')
        self.supports = torch.tensor(scaled_laplacian(self.adj_mx)).type(torch.float32).to(self.device)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.seq_len = self.len_period + self.len_trend + self.len_closeness  # Seq_len
        self.block1 = ST_BLOCK_2(self.feature_dim, self.c_out, self.num_nodes,
                                 self.seq_len, self.K, self.Kt, self.device)
        self.block2 = ST_BLOCK_2(self.c_out, self.c_out, self.num_nodes,
                                 self.seq_len, self.K, self.Kt, self.device)
        self.bn = BatchNorm2d(self.feature_dim, affine=False)

        self.conv1 = Conv2d(self.c_out, self.output_dim, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv2 = Conv2d(self.c_out, self.output_dim, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv3 = Conv2d(self.c_out, self.output_dim, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv4 = Conv2d(self.c_out, self.output_dim, kernel_size=(1, 2), padding=(0, 0),
                            stride=(1, 2), bias=True)

        self.h = Parameter(torch.zeros(self.num_nodes, self.num_nodes), requires_grad=True).to(self.device)  # L_par
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(self, batch):
        # batch：［B,Tw+Td+Th, N, F] , closeness, period and weekly are added
        x = batch['X'].permute(0, 3, 2, 1)  # (B, F_in, N_nodes, Tw+Td+Th)
        x_list = []
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            x_r = x[:, :, :, begin_index:end_index]
            x_r = self.bn(x_r)
            x_list.append(x_r)  # (B, F_in, N_nodes,Th)
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            x_d = x[:, :, :, begin_index:end_index]
            x_d = self.bn(x_d)
            x_list.append(x_d)  # (B, F_in, N_nodes, Td)
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            x_w = x[:, :, :, begin_index:end_index]
            x_w = self.bn(x_w)
            x_list.append(x_w)  # (B, F_in, N_nodes, Tw)
        x = torch.cat(x_list, -1)  # (B, F_in, N_nodes, Tw+Td+Th)

        A = self.h + self.supports  # L_par + L~
        d = 1 / (torch.sum(A, -1) + 0.0001)  # num_nodes
        D = torch.diag_embed(d)  # 函数将以为Tensor转化为一个二维张量
        A = torch.matmul(D, A)
        A1 = F.dropout(A, 0.5, self.training)

        x, _, _ = self.block1(x, A1)  # x: b, c, n, l
        x, d_adj, t_adj = self.block2(x, A1)  # x: b, c, n, l

        # "len_closeness": 2,"len_period": 1,"len_trend": 2
        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]
        x3 = x[:, :, :, 24:36]
        x4 = x[:, :, :, 36:60]  # trend

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)  # B ,output_dim, N, 1
        x = x1 + x2 + x3 + x4
        x = x.permute(0, 3, 2, 1)  # B, 12, N, output_dim=1
        return x

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
