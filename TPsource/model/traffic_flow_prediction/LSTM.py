import torch
import torch.nn as nn
import random
import math
from logging import getLogger
from TPsource.model import loss
from TPsource.model.abstract_traffic_state_model import AbstractTrafficStateModel

class LSTMCell(nn.Module):
    def __init__(self, feature_dim, hidden_size,device):
        super(LSTMCell,self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        #i_t
        self.W_xi = nn.Parameter(torch.FloatTensor(self.feature_dim, self.hidden_size), requires_grad=True).to(device)
        self.W_hi = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size), requires_grad=True).to(device)
        self.b_i = nn.Parameter(torch.FloatTensor(self.hidden_size)).to(device)
        #f_t
        self.W_xf = nn.Parameter(torch.FloatTensor(self.feature_dim, self.hidden_size), requires_grad=True).to(device)
        self.W_hf = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size), requires_grad=True).to(device)
        self.b_f = nn.Parameter(torch.FloatTensor(self.hidden_size)).to(device)
        #c_hat_t
        self.W_xc_hat = nn.Parameter(torch.FloatTensor(self.feature_dim, self.hidden_size), requires_grad=True).to(device)
        self.W_hc_hat = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size), requires_grad=True).to(device)
        self.b_c_hat = nn.Parameter(torch.FloatTensor(self.hidden_size)).to(device)

        # o_t
        self.W_xo = nn.Parameter(torch.FloatTensor(self.feature_dim, self.hidden_size), requires_grad=True).to(device)
        self.W_ho = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size), requires_grad=True).to(device)
        self.b_o = nn.Parameter(torch.FloatTensor(self.hidden_size)).to(device)
        
        #PyTorch默认值中的权重初始化nn.Module
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_xi)
        nn.init.xavier_uniform_(self.W_xf)
        nn.init.xavier_uniform_(self.W_xo)
        nn.init.xavier_uniform_(self.W_xc_hat)
        nn.init.xavier_uniform_(self.W_hi)
        nn.init.xavier_uniform_(self.W_hf)
        nn.init.xavier_uniform_(self.W_ho)
        nn.init.xavier_uniform_(self.W_hc_hat)
        nn.init.constant_(self.b_i, 0)
        nn.init.constant_(self.b_f, 0)
        nn.init.constant_(self.b_o, 0)
        nn.init.constant_(self.b_c_hat, 0)

    def forward(self,x, init_state=None):
        # assume x.shape =[batch_size, seq_len, feature_dim]
        bs, num_nodes, feature_dim = x.size()
        x = x.reshape((bs*num_nodes, feature_dim))
        #init_states参数是（h_t，c_t）参数的元组，如果不引入，则设置为零
        h_t, c_t = init_state #

        i_t = torch.sigmoid(x @ self.W_xi + h_t @ self.W_hi + self.b_i) # matrix multiply #[batch_size*N hidden_szie]
        f_t = torch.sigmoid(x @ self.W_xf + h_t @ self.W_hf + self.b_f) #[batch_size*N hidden_szie]
        c_hat_t = torch.tanh(x @ self.W_xc_hat + h_t @ self.W_hc_hat +self.b_c_hat)
        o_t = torch.sigmoid(x @ self.W_xo + h_t @ self.W_ho + self.b_o)

        c_t = f_t * c_t + i_t * c_hat_t 
        h_t = o_t * torch.tanh(c_t) # [batch_size*N, hidden_szie]
        return h_t,c_t

class LSTM(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(LSTM, self).__init__(config, data_feature)
        self._scaler = data_feature.get('scaler')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = data_feature.get('scaler')

        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        self.bidirectional = config.get('bidirectional', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)

        self._logger.info('You select rnn_type {} in RNN!'.format(self.rnn_type))
        self.lstm = LSTMCell(self.feature_dim, self.hidden_size, self.device)
        self.conv1 = nn.Conv2d(self.hidden_size, self.output_window, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)
        self.fc = nn.Linear(self.hidden_size, self.output_window)

    def forward(self, batch):
        src = batch['X'].clone()  # [batch_size, input_window, num_nodes, feature_dim]
        batch_size, input_window, num_nodes, feature_dim = src.shape
        target = batch['y']  # [batch_size, output_window, num_nodes, feature_dim]

        outputs = list()
        hidden_state, cell_state = torch.zeros(batch_size*num_nodes, self.hidden_size).to(self.device), torch.zeros(batch_size*num_nodes,self.hidden_size).to(self.device)
        for i in range(int(input_window)):
            hidden_state, cell_state = self.lstm(src[:, i, :, :], (hidden_state, cell_state))  # [B*N, H]
            output = hidden_state.view(batch_size, num_nodes, self.hidden_size).contiguous().unsqueeze(dim=-1)  # [B, N, H, F]

            # carry out convlution
            output = self.conv1(output.permute(0, 2, 1, 3).contiguous())  # (B, output_window, N, F)
            outputs.append(output)

        last_output = outputs[-1]  # [B, output_window, N, F]
        return last_output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
