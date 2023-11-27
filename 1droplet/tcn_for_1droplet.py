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

import matplotlib.pyplot as plt

# 正規化
from sklearn.preprocessing import MinMaxScaler


from callback import tcn
from callback import earlystopping

import argparse

parser = argparse.ArgumentParser(description='Set hyper parameters')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='epochs (default: 5)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 100)')

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
time_step = 300

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

""" check reproducibility
for i, (x,t) in enumerate(train_dataloader):
    print(x)
    print(t)
    
    if i == 1:
        break

debag = ok
[1]
0.7542, 0.7518, 0.7512,  ..., 0.7702, 0.7681, 0.7664

[2]
0.2995, 0.2994, 0.2972,  ..., 0.4175, 0.4165, 0.4156

0.3641, 0.3638, 0.3614,  ..., 0.5012, 0.5006, 0.5012
"""

# =====train and evaluate func=====
def train_model(model, optimizer, batch_size: int=0, clip=-1):
    model.train()
    batch_idx = 1 # バッチが何回カウントされたか
    total_loss = 0
    
    for i, (x,t) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(x, debug=False)
        
        # 出力と正解の誤差をバッチの数で割る（バッチにおける誤差の平均）
        loss = F.mse_loss(output, t)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            cur_loss = total_loss / batch_idx
            processed = min(i+batch_size, X_train.size(0))
            # print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
            #     epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            # total_loss = 0
            
    return total_loss / batch_idx
            
def evaluate(model, epoch=None, train_loss=None):
    model.eval()
    with torch.no_grad():
        test_loss = .0
        for i, (x,t) in enumerate(test_dataset):
            output = model(x.reshape(1,2,-1))
            t_loss = F.mse_loss(output, t.reshape(1,2))
            test_loss += t_loss.item()
            
        test_loss /= len(test_dataset)    

        if epoch % 5 == 0 or epoch == 1:
            print('\n [epoch: {}] Train loss: {:.6f} \t Test set: Average loss: {:.6f}\n'.format(epoch, train_loss, test_loss))
        
        return test_loss



epoch_list = [20, 30, 50, 100]
    

# =====Model define=====
epochs = args.epochs
lr = 1e-3
level = 3
h_dim = 15
kernel_size = 5

model = tcn.myTCN(input_size=2, output_size=2, num_channels=[h_dim]*level, kernel_size=kernel_size, dropout=0.0)
optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)

# =====train=====
loss = {"val_loss":[], "train_loss":[]}
es = earlystopping.EarlyStopping(patience=5, verbose=1)

for epoch in range(1, epochs + 1):
    train_loss = train_model(model, optimizer, clip=-1)
    tloss = evaluate(model, epoch, train_loss)
    
    loss["train_loss"].append(train_loss)
    loss["val_loss"].append(tloss)

"""check reproducibility ==> ok
[epoch: 1] Train loss: 0.125471         Test set: Average loss: 0.156144


[epoch: 5] Train loss: 0.002935         Test set: Average loss: 0.017688


[epoch: 10] Train loss: 0.000454        Test set: Average loss: 0.005820


[epoch: 15] Train loss: 0.000072        Test set: Average loss: 0.004358


[epoch: 20] Train loss: 0.000051        Test set: Average loss: 0.003631


epoch 30
[epoch: 1] Train loss: 0.125471         Test set: Average loss: 0.156144


[epoch: 5] Train loss: 0.002935         Test set: Average loss: 0.017688


[epoch: 10] Train loss: 0.000454        Test set: Average loss: 0.005820


[epoch: 15] Train loss: 0.000072        Test set: Average loss: 0.004358


[epoch: 20] Train loss: 0.000051        Test set: Average loss: 0.003631


[epoch: 25] Train loss: 0.000048        Test set: Average loss: 0.003686


[epoch: 30] Train loss: 0.000054        Test set: Average loss: 0.003820
"""

# =====plot loss=====
plt.rcParams['font.family'] ='sans-serif'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 8 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

plt.xticks(np.arange(1, len(loss["val_loss"])+1, 2))
plt.xlim(0, len(loss["val_loss"]))
plt.ylim(0, max(loss["val_loss"]))

plt.xlabel("epoch", fontsize=12)
plt.ylabel("loss", fontsize=12)

plt.plot(loss["train_loss"], label="train loss")
plt.plot(loss["val_loss"], label="val loss")
plt.legend()
plt.title("TCN \n batch:{}, t_step:{}, epoch:{} \n level:{}, h_dim:{}, k_size:{}, lr:{}".format(
    batch_size, time_step, epochs, level, h_dim, kernel_size, lr), fontsize=13)

plt.tight_layout()

loss_path = "outputs/oscillated_loss/tcnloss_batch{}_tstep{}_epoch{}_level{}_hdim{}_ksize{}_lr{}.png".format(
    batch_size, time_step, epochs, level, h_dim, kernel_size, lr)
plt.savefig(loss_path)
plt.show()


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

# generate prediction
gen = generate_pred(model, gen_time, z)

# plot
plt.figure(figsize=(10,10))
gen = np.array(gen)
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
    loss["val_loss"][-1], batch_size, time_step, epochs, level, h_dim, kernel_size, lr), fontsize=13)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

pred_path = "outputs/pred_oscillated/tcn_batch{}_tstep{}_epoch{}_level{}_hdim{}_ksize{}_lr{}.png".format(
    batch_size, time_step, epochs, level, h_dim, kernel_size, lr)

plt.savefig(pred_path)

plt.show()

# =====model save and log========
print("model save")

model_path = "model/tcn_batch{}_tstep{}_epoch{}_level{}_hdim{}_ksize{}_lr{}.pth".format(
    batch_size, time_step, epochs, level, h_dim, kernel_size, lr)
# torch.save(model, model_path)

# =====log=====
save_output = [{
    "model": "tcn",
    "batch size": batch_size,
    "time step": time_step,
    "epoch": epochs,
    "level": level,
    "hidden dim": h_dim,
    "kernel size": kernel_size,
    "learning rate": lr,
    "val loss": loss["val_loss"][-1],
    "model path": model_path,
    "loss path": loss_path,
    "pred_path": pred_path,
}]

import csv

with open('new_model_log_oscillated.csv','a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = list(save_output[0]))
    # writer.writeheader()
    writer.writerows(save_output)
