import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
import torch
import torch.nn as nn
from logging import getLogger
import math
import torch.nn.functional as F
from TPsource.model import loss
from torch.autograd import Variable
from TPsource.model.abstract_traffic_state_model import AbstractTrafficStateModel


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


def scaled_laplacian_sparse(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = eigs(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32)


def normalized_laplacian(weight):
    '''
    1st-order approximation function.
    :param W: weighted adjacency matrix of G. Not laplacian matrix.
    :return: np.ndarray
    '''
    # TODO:
    n = weight.shape[0]
    adj = weight + np.identity(n)
    d = np.sum(adj, axis=1)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    # lap = np.expand_dims(lap, axis=0)              # 1*n*n
    return lap


def normalized_laplacian_sparse(adj):
    """
    A = A + I
    L = D^-1/2 A D^-1/2
    """
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

class MTDLGC(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(MTDLGC, self).__init__(config, data_feature)
        self._scaler = data_feature.get('scaler')
        self.input_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.K= config.get('K', 3) #adj
        self.Kt = config.get('Kt', 3)
        self.lgc = config.get('lgc', False)
        self.lgc_type = config.get('lgc_type', 'randW')
        self.num_nodes = data_feature.get('num_nodes', '')
        config['num_nodes'] = self.num_nodes
        self.batch_size = config.get('batch_size', 64)
        self.hidden_size = int(config.get('rnn_units', 64))
        self.residual = config.get('residual', False)
        self.dropout = config.get('dropout', 0)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))
        self.len_period = data_feature.get('len_period', 1)
        self.len_trend = data_feature.get('len_trend', 1)
        self.len_closeness = data_feature.get('len_closeness', 1)

        self.rnn_type = config.get('rnn_type', 'RNN')
        self.rnn_layers = config.get('rnn_layers', 1)
        self.lgc_layers = config.get('lgc_layers', 1)
        self.bidirectional = config.get('bidirectional', False)
        self._logger = getLogger()

        self.adj_mx = data_feature.get('adj_mx')
        self.laplacian =[]
        if self.lgc == True:
            self.w_adj = nn.Parameter(torch.zeros(self.batch_size,self.num_nodes, self.num_nodes),
                                      requires_grad=True).to(self.device)  # LGC
            nn.init.xavier_normal_(self.w_adj)
            for adj in self.adj_mx:
                normalized_adj = torch.tensor(normalized_laplacian(adj)).type(torch.float32).to(self.device)
                self.laplacian.append(normalized_adj*self.w_adj)
        else:
            self.w_adj = nn.Parameter(torch.ones(self.batch_size, self.num_nodes, self.num_nodes),
                                      requires_grad=True).to(self.device)
            for adj in self.adj_mx:
                normalized_adj = torch.tensor(normalized_laplacian(adj)).type(torch.float32).to(self.device)
                self.laplacian.append(normalized_adj)

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.input_size = self.input_dim
        self._logger.info('You select rnn_type {} in RNN!'.format(self.rnn_type))
        if self.rnn_type.upper() == 'LSTM':
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.rnn_layers,
                               bidirectional=self.bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))
        
        self.lstm_adj = nn.LSTM(self.hidden_size, self.num_nodes,  batch_first=True) 
        
        self._W_theta = torch.nn.Parameter(torch.zeros((1*(self.K+1), self.input_dim),
                                                       device=self.device), requires_grad=True)
        self._bias = torch.nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=True)
        torch.nn.init.xavier_normal_(self._W_theta)
        torch.nn.init.constant_(self._bias, 0)

        self.convw = nn.Conv2d(self.hidden_size, self.output_window, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)
        self.convd = nn.Conv2d(self.hidden_size, self.output_window, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)
        self.convh = nn.Conv2d(self.hidden_size, self.output_window, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)
        self.time_conv = nn.Conv2d(self.input_dim, self.hidden_size, kernel_size=(
            1, self.Kt), padding=(0, 1), stride=(1, 1), bias=True)  # 特征提取
        
    def dynamic_adj(self, inputs):
        # inputs:[B, Tw+Td+Th=L, N, F]
        x = inputs.permute(0, 3, 2, 1)  # (B, F_in, N, L)
        batch_size, input_dim, num_nodes, seq_len = x.shape
        x_1 = self.time_conv(x)
        # x_1 = F.leaky_relu(x_1)  # (B, H, N, L)

        # LSTM
        h = torch.zeros((1, batch_size * num_nodes, num_nodes)).to(self.device)  # 1,b*n, n
        c = torch.zeros((1, batch_size * num_nodes, num_nodes)).to(self.device)  # 1,b*n, n

        x_1 = x_1.permute(0, 2, 3, 1)  # [B,N,L,H]
        x_1 = x_1.reshape((batch_size*num_nodes, seq_len, self.hidden_size))  # [B*N,L,H]
        _, (hidden, _) = self.lstm_adj(x_1, (h, c))  # hidden:(num_layers, B, H) [1,b*n, n]
        dynamic_adj = hidden.squeeze().view(batch_size, num_nodes, num_nodes).contiguous()  # b,n,n
        return dynamic_adj 
    
    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
    
    def LGC_gc(self, inputs, dynamic_adj):
        batch_size, num_nodes, _ = inputs.shape  # inputs:(batch_size, num_nodes, F)
        X0 = inputs  # (B, N, 1)
        X = torch.unsqueeze(X0, dim=0)  # X:(1, B, N, 1)

        for k in range(1, self.K+1):
            #X1: (B, N, 1); laplacian:[B,N,N]
            X1 = torch.einsum('bnm,bml->bnl', self.laplacian[k-1]* dynamic_adj, X0) # [B, N, 1]
            # X: (K, B, N, 1)
            X = self._concat(X, X1)  # (K, B, num_nodes, 1)
            X0 = X1 
       
        X = X.permute(1, 2, 3, 0)  # [B, N, 1, K+1]
        # X: (B * N, (1)*K+1)
        X = X.reshape((batch_size * num_nodes, -1))
        # (B * N, input_dim)
        outputs = torch.matmul(X, self._W_theta)+self._bias
        # (B, N, input_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self.input_dim))
        return outputs
    
    def forward(self, batch):
        """
        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window:Tw+Td+Th, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n
        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        target = batch['y']  # [batch_size, output_window, num_nodes, feature_dim]
        batch_size, input_window, num_nodes, input_dim = inputs.shape
        target = target.permute(1, 0, 2, 3)  # [output_window, batch_size, num_nodes, output_dim]
        
        input_chunk = inputs.chunk(3, dim=1)
        week_inputs = input_chunk[2].permute(1, 0, 2, 3)  # [Tw, B, N, F]
        daily_inputs = input_chunk[1].permute(1, 0, 2, 3)  # [Td, B,  N, F]
        hour_inputs = input_chunk[0].permute(1, 0, 2, 3)  # [Th, B,  N, F]

        dynamic_adj = self.dynamic_adj(input_chunk[0])

        for index_lgc in range(self.lgc_layers):
            week_gc_outputs = []
            daily_gc_outputs = []
            hour_gc_outputs = []
            for i in range(int(input_window / 3)):
                week_gc = self.LGC_gc(week_inputs[i,:, :, :], dynamic_adj) # [B, N, F]
                daily_gc = self.LGC_gc(daily_inputs[i,:, :, :], dynamic_adj)
                hour_gc = self.LGC_gc(hour_inputs[i,:, :, :], dynamic_adj)
                week_gc_outputs.append(week_gc)
                daily_gc_outputs.append(daily_gc)
                hour_gc_outputs.append(hour_gc)
            week_inputs = torch.stack(week_gc_outputs,dim=0) # [12, B, N, F]
            daily_inputs = torch.stack(daily_gc_outputs, dim=0)
            hour_inputs = torch.stack(hour_gc_outputs, dim=0)

        week_gc_outputs = week_inputs.reshape((self.input_window, batch_size*num_nodes, self.input_dim)) # [12, B*N, F]
        daily_gc_outputs = daily_inputs.reshape((self.input_window, batch_size*num_nodes, self.input_dim)) 
        hour_gc_outputs = hour_inputs.reshape((self.input_window, batch_size*num_nodes, self.input_dim)) 
        # week_gc_outputs: [input_window, batch_size*num_nodes,  hidden_size]
        # out: [input_window, batch_size*num_nodes, hidden_size * num_directions]
        output_week, _ = self.lstm(week_gc_outputs)  
        output_daily, _ = self.lstm(daily_gc_outputs)
        output_hour, _ = self.lstm(hour_gc_outputs)

        if self.residual:
            output_week = output_week[-1] + self.LGC_gc(week_inputs[-1,:,:,:], dynamic_adj)
            output_daily = output_daily[-1] + self.LGC_gc(daily_inputs[-1,:,:,:], dynamic_adj)
            output_hour = output_hour[-1] + self.LGC_gc(hour_inputs[-1,:, :, :], dynamic_adj)
        else:
            output_week = output_week[-1]
            output_daily = output_daily[-1]
            output_hour = output_hour[-1]
        
        output_week = output_week.reshape((batch_size, self.num_nodes,
                                        self.hidden_size*self.num_directions)).unsqueeze(dim=-1) # [B, N, H*direction, F]
        output_daily = output_daily.reshape((batch_size, self.num_nodes,
                                            self.hidden_size*self.num_directions)).unsqueeze(dim=-1)
        output_hour = output_hour.reshape((batch_size, self.num_nodes,
                                        self.hidden_size*self.num_directions)).unsqueeze(dim=-1)
        
        # carry out convlution
        output_week = self.convw(output_week.permute(0, 2, 1, 3).contiguous())  # (B, output_window, N, F)
        output_daily = self.convd(output_daily.permute(0, 2, 1, 3).contiguous())
        output_hour = self.convh(output_hour.permute(0, 2, 1, 3).contiguous())
        last_output = (output_week + output_daily + output_hour)/3 ## [B, output_window, N, F]
        return last_output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
