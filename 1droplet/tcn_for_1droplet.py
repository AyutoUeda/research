""" tcnを用いた運動予測
Parse Args:
    -h, --help: show this help message and exit
    --epochs: epoch数
    --batch_size: batch size
    --time_step: time step
    --level: level
    
Example:
    >>>$ python3 tcn_for_1droplet.py --epochs 5 --batch_size 100 --time_step 300
    
Note:
    * 使用するデータはコード内で指定
    * lr, level, h_dim, kernel_sizeはコード内で指定
    * 乱数固定
    * 1つのドロップレットの運動を予測。入力数を拡張する際はmodelのinput_sizeを変更

"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import os
import random
import pandas as pd
import csv

import matplotlib.pyplot as plt

# 正規化
from sklearn.preprocessing import MinMaxScaler


from callback import tcn
from callback import earlystopping
from callback.train_eval import TrainEval

import argparse

parser = argparse.ArgumentParser(description='Set hyper parameters')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='epochs (default: 5)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 100)')
parser.add_argument('--time_step', type=int, default=300, metavar='N',
                    help='time step (default: 300)')
parser.add_argument('--level', type=int, default=15, metavar='N',
                    help='level (default: 15)')

args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 乱数固定用の処理 ===================================
# 同じ学習結果を得る
seed = 42


os.environ['PYTHONHASHSEED'] = str(seed)
# Python random
random.seed(seed)
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(seed):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
# ====================================================

# read data
def mk_dataframe(path):
    df = pd.read_table(path, sep=" ", header=None)
    df = df.drop(columns=0, axis=1)
    df = df.drop(columns=3, axis=1)
    return df

df = mk_dataframe("230724/pos-7.dat")
df_ = df.values

# get position by 1sec
pos_by1sec = df_[::10]
# scaled
scaler = MinMaxScaler(feature_range = (0, 1))

pos_by1s_scaled = scaler.fit_transform(pos_by1sec)

# time step
time_step = args.time_step # default: 300

len_seq = len(pos_by1s_scaled)
data_size = len_seq - time_step

# =====create data to use=====
data = np.zeros((data_size, 2, time_step))
t = np.zeros((data_size, 2))

for i in range(data_size):
    data[i, 0] = pos_by1s_scaled[i:i+time_step, 0]
    data[i, 1] = pos_by1s_scaled[i:i+time_step, 1]
    t[i] = pos_by1s_scaled[i+time_step]
    
# Tensor
X = torch.Tensor(np.array(data).reshape(-1, 2, time_step)).to(device)
t = torch.Tensor(np.array(t).reshape(-1, 2)).to(device)

n_seq = X.size(0)

# =====create train and test data=====
train_size = int(n_seq * 0.9)
test_size = n_seq - train_size
print("train size:{}, test size: {}".format(train_size, test_size))

X_train, Y_train = X[:train_size], t[:train_size]
X_test, Y_test = X[train_size:], t[train_size:]

# =====create datasets=====
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

# =====create dataloaders=====
g = torch.Generator()
g.manual_seed(seed)

def createDataloader(batch_size:int, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   worker_init_fn=seed_worker,
                                                   generator=g,
                                                   )
    return train_dataloader

batch_size = args.batch_size # default: 100
train_dataloader = createDataloader(batch_size, train_dataset)
test_dataloader = createDataloader(test_size, test_dataset)


# =====Model define=====
# epochs = args.epochs
# level = args.level
epochs = 100
lr = 1e-3
level = 15
h_dim = 11
kernel_size = 8


model = tcn.myTCN(input_size=2, output_size=2, num_channels=[h_dim]*level, kernel_size=kernel_size, dropout=0.0)
optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
criterion = nn.MSELoss()

es = earlystopping.EarlyStopping(patience=5, verbose=1)

# =====train=====
train_eval = TrainEval(
    model, train_dataloader, test_dataloader, criterion=criterion, optimizer=optimizer, clip=-1, early_stopping=es
    )
loss_dict, cnt_epochs = train_eval.train(epochs)


# =====plot loss=====
def plot_loss(loss_dict: dict, save: bool=False) -> str:
    """plot loss
    
    Args:
        loss_dict (dict): lossを格納したdict(train, valの2つのkeyを持つ)
        save (bool, optional): saveするかどうか. Default: False.
    
    Returns:
        loss_path (str): lossのplotを保存したpath
    """
    plt.rcParams['font.family'] ='sans-serif'#使用するフォント
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 8 #フォントの大きさ
    plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

    plt.xlim(0, len(loss_dict["val"]))
    plt.ylim(0, max(loss_dict["val"]))

    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)

    plt.plot(loss_dict["train"], label="train loss")
    plt.plot(loss_dict["val"], label="val loss")
    plt.legend()
    
    plt.title("TCN \n batch:{}, t_step:{}, epoch:{} \n level:{}, h_dim:{}, k_size:{}, lr:{}".format(
        batch_size, time_step, cnt_epochs, level, h_dim, kernel_size, lr), fontsize=13)

    plt.tight_layout()

    loss_path = "outputs/oscillated_es5/loss/tcnloss_batch{}_tstep{}_epoch{}_level{}_hdim{}_ksize{}_lr{}.png".format(
        batch_size, time_step, cnt_epochs,level, h_dim, kernel_size, lr)
    if save:
        plt.savefig(loss_path)
    # plt.show()
    
    return loss_path

save_bool = True
loss_path = plot_loss(loss_dict, save=save_bool)


# =====prediction=====
def generate_pred(model, gen_time:int, z: torch.Tensor)-> list:
    gen = [[None, None] for i in range(time_step)]
    
    if type(z) != torch.Tensor:
        z = torch.Tensor(z)
        
    z = z.reshape(1,2,-1)
    
    for i in range(gen_time):
        model.eval()
        pred = model(z).data.cpu().numpy()
        z = np.concatenate([z.numpy().reshape(2,-1), pred.reshape(2,-1)], 1)
        z = z[:,1:]
        # print(z.shape)
        z = torch.Tensor(z.reshape(1,2,-1))
        gen.append([pred[0,0], pred[0,1]])
    
    return np.array(gen)

gen_time = len(data) - time_step
# input to predict
z = torch.Tensor(data[0].reshape(1,2,-1))
gen = generate_pred(model, gen_time, z)


# =====plot gen=====
def plot_gen(gen: np.ndarray, save: bool=False) -> str:
    plt.figure(figsize=(10,10))
    plt.plot(gen[:,0], gen[:,1], label="gen", linewidth=3)
    plt.scatter(gen[time_step,0], gen[time_step, 1], marker="x")
    # plt.scatter(gen[train_size,0], gen[train_size, 1], marker="x", s=100)


    plot_train = X_train[:,:,0].numpy()
    plt.plot(plot_train[:,0], plot_train[:,1], label="train")
    # plt.plot(X_test[0,0], X_test[0,1], label="given")
    plt.plot(pos_by1s_scaled[:time_step,0], pos_by1s_scaled[:time_step,1], label="given")
    plt.plot(pos_by1s_scaled[:,0], pos_by1s_scaled[:,1], color="gray", alpha=0.3, label="val")


    # plt.title("TCN \n epoch:{} \n val loss:{:6f}".format(epochs, loss["val_loss"][-1]))
    plt.title("TCN \n val loss:{:6f} \n batch:{}, t_step:{}, epoch:{} \n level:{}, h_dim:{}, k_size:{}, lr:{}".format(
        loss_dict["val"][-1], batch_size, time_step, cnt_epochs, level, h_dim, kernel_size, lr), fontsize=13)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

    pred_path = "outputs/oscillated_es5/pred/tcn_batch{}_tstep{}_epoch{}_level{}_hdim{}_ksize{}_lr{}.png".format(
        batch_size, time_step, cnt_epochs, level, h_dim, kernel_size, lr)
    
    if save:
        plt.savefig(pred_path)

    plt.show()
    
    return pred_path

pred_path = plot_gen(gen, save=save_bool)


# =====model save and log========
model_path = "model/tcn_with_es/tcn_batch{}_tstep{}_epoch{}_level{}_hdim{}_ksize{}_lr{}.pth".format(
    batch_size, time_step, cnt_epochs, level, h_dim, kernel_size, lr)

if save_bool:
    print("model save")
    torch.save(model, model_path)

# =====log=====
save_output = [{
    "model": "tcn",
    "batch size": batch_size,
    "time step": time_step,
    "epoch": cnt_epochs,
    "level": level,
    "hidden dim": h_dim,
    "kernel size": kernel_size,
    "learning rate": lr,
    "val loss": loss_dict["val"][-1],
    "model path": model_path,
    "loss path": loss_path,
    "pred_path": pred_path,
}]


if save_bool:
    with open('tcn_log_oscillated(es5).csv','a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = list(save_output[0]))
        # writer.writeheader()
        writer.writerows(save_output)

