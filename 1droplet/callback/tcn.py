import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np
from collections import OrderedDict

def _debug_print(debug, *content):
    if debug:
        print(*content)
        
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
        
class myTCN(nn.Module):
    def __init__(self, input_size=2, output_size=1, num_channels=[5]*2, kernel_size=4, dropout=0.0):
        super(myTCN, self).__init__()
        self.layers = OrderedDict()
        self.num_levels = len(num_channels)
        
        for i in range(self.num_levels):
            dilation = 2 ** i
            n_in = input_size if i == 0 else num_channels[i-1]
            n_out = num_channels[i]
            padding = (kernel_size-1) * dilation
            
            # ========== TemporalBlock ==========
            self.layers['conv1_' + str(i)] = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation))
            self.layers['chomp1_' + str(i)] = Chomp1d(padding)
            self.layers['relu1_' + str(i)] = nn.ReLU()
            self.layers['dropout1_' + str(i)] = nn.Dropout(dropout)
            self.layers['conv2_' + str(i)] = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation))
            self.layers['chomp2_' + str(i)] = Chomp1d(padding)
            self.layers['relu2_' + str(i)] = nn.ReLU()
            self.layers['dropout2_' + str(i)] = nn.Dropout(dropout)
            self.layers['downsample_' + str(i)] = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
            self.layers['relu_' + str(i)] = nn.ReLU()
            # ===================================
        
        self.network = nn.Sequential(self.layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        
    def init_weights(self):
        for i in range(self.num_levels):
            self.layers['conv1_' + str(i)].weight.data.normal_(0, 0.01)
            self.layers['conv2_' + str(i)].weight.data.normal_(0, 0.01)
            if self.layers['downsample_' + str(i)] is not None:
                self.layers['downsample_' + str(i)].weight.data.normal_(0, 0.01)
        self.linear.weight.data.normal_(0, 0.01)
    
    def forward(self, x, debug=False):
        _debug_print(debug, '========== forward ==========')
        _debug_print(debug, x.size())
        for i in range(self.num_levels):
            _debug_print(debug, '---------- block {} ----------'.format(i))
            _debug_print(debug, 'in : ', x.size())
            res = x if self.layers['downsample_' + str(i)] is None else self.layers['downsample_' + str(i)](x) # Residual Connection
            out = self.layers['conv1_' + str(i)](x)
            out = self.layers['chomp1_' + str(i)](out)
            out = self.layers['relu1_' + str(i)](out)
            out = self.layers['dropout1_' + str(i)](out)
            out = self.layers['conv2_' + str(i)](out)
            out = self.layers['chomp2_' + str(i)](out)
            out = self.layers['relu2_' + str(i)](out)
            out = self.layers['dropout2_' + str(i)](out)
            _debug_print(debug, 'out: ', out.size())
            _debug_print(debug, 'res: ', res.size())
            x = self.layers['relu_' + str(i)](out + res)
            _debug_print(debug, x.size())
            _debug_print(debug, '-----------------------------')
        _debug_print(debug, x.size())
        x = self.linear(x[:, :, -1])
        _debug_print(debug, x.size())
        _debug_print(debug, '=============================')
        return x