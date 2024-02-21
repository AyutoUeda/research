from torch import nn

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self, n_nearest_neighbors):
        super(Net, self).__init__()
        self.input_dim = n_nearest_neighbors * 4
        
        self.layers1 = nn.Sequential(
            nn.Linear(self.input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.BatchNorm1d(40),
            nn.Linear(40, 8)
        )

    # 順伝播
    def forward(self, x):
        
        x = self.layers1(x)
        out = self.layers2(x)
        return out    