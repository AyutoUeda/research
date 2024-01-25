import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 重みの初期値設定 --> 勾配消失を防ぐ
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        
    def forward(self, x):
        y_rnn, h = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])
        
        return y