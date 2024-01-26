import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchinfo import summary # ニューラルネットワークの中身を見る

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import csv

from callback import earlystopping

from sklearn.preprocessing import MinMaxScaler

from callbacks_for_RNN.data_create import DataCreate
from callbacks_for_RNN.rnn import RNN



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





    
def train_func(net, epochs=100, hidden_dim=30, save_bool=False):
    loss_func = nn.MSELoss(reduction="mean")

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    device = torch.device("cuda:0" if torch.cuda. is_available() else "cpu")  #デバイス(GPU or CPU)設定 
    
    hist = {"train_loss":[], 
            "val_loss":[]}

    net.to(device)

    es = earlystopping.EarlyStopping(patience=5, verbose=1)
    log_epochs = 0
    
    for i in range(epochs):
        net.train()
        running_loss = 0.0
        running_val_loss = 0.0

        for j, (x, t) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            y = net(x)
            y = y.to(device)

            loss = loss_func(y, t)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        running_loss /= j + 1
        hist["train_loss"].append(running_loss)
        
        for j, (x, t) in enumerate(val_loader):
            x, t = x.to(device), t.to(device)
            
            net.eval()
            
            pred = net(x)
            loss = loss_func(pred, t)
            running_val_loss += loss.item()
            
        running_val_loss /= j + 1
        hist["val_loss"].append(running_val_loss)


        if i%5 == 0 or i==epochs-1:
            print("Epoch:{}, Train_Loss:{:3f}, Val_Loss:{:3f}".format(i, running_loss, running_val_loss))

        if es(running_val_loss):
            print("Epoch:{}, Train_Loss:{:3f}, Val_Loss:{:3f}".format(i, running_loss, running_val_loss))
            log_epochs = i
            break
        
        log_epochs = i
    
    #lossの推移を確認
    plt.plot(range(len(hist["train_loss"])), hist["train_loss"], label='train loss')
    plt.plot(range(len(hist["val_loss"])), hist["val_loss"], label='val loss')
    
    plt.legend()
    plt.title("Change of Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if save_bool:
        plt.savefig(("1droplet/outputs_rnn/loss/loss_hdim{}.png".format(hidden_dim)))
    plt.show()
    
    return hist["val_loss"][-1], log_epochs+1
    
def future_prediction(net, data, start, pred_len, hidden_dim, save_bool=False):
    """
    Args:
        net: model to use
        data: dataset to use
        start: start prediction at this point
        pred_len: how many seconds the model predict
    """
    X = data
    net.eval()

    gen = [[None, None] for i in range(start)] # 予測値を時系列で保持するためのリスト

    # z = X[start:start+time_step].reshape(-1, time_step, 2) # 予測用に未知の部分の最初10個
    
    z = X[:time_step].reshape(-1, time_step, 2) # 予測用に最初の50データtimestep分を与える
    

    # pred_lenは何回予測を繰り返すか＝何秒後まで予測するか
    for i in range(pred_len):
        z_ = torch.Tensor(z[-1:, :])

        preds = net(z_).data.cpu().numpy()
        z = np.concatenate([z, preds.reshape(-1, 1, 2)], 1)
        z = z[:,1:,:]

        gen.append([preds[0,0], preds[0,1]])
        
    print(net)
    
    plt.figure(figsize=(7,7))
    
    gen = np.array(gen)
    plt.plot(gen[:,0], gen[:,1], label="gen", linewidth=3)
    plt.scatter(gen[time_step,0], gen[time_step, 1], marker="x")

    plt.plot(X[:train_size,0], X[:train_size,1], label="train data")
    
    plt.plot(X[:,0], X[:,1], color="gray", alpha=0.3, label="val")
    # plt.plot(X[:,0], X[:,1], label="Correct", alpha=0.6)
    
    plt.plot(X[:time_step,0], X[:time_step,1], label="given")

    plt.legend()

    # plt.scatter(X[start+time_step,0], X[start+time_step,1], marker="x")
    # plt.scatter(X[train_size+time_step,0], X[train_size+time_step,1], marker="x")
    plt.title("hidden_dim={} \n start prediction at {} s".format(net.hidden_dim, start))
    
    if save_bool:
        plt.savefig(("1droplet/outputs_rnn/pred/pred_hdim{}.png".format(hidden_dim)))
    # plt.savefig("outputs/1droplet/pred_1droplts_hdim{}_at{}.png".format(net.hidden_dim, start))
    plt.show()
    
    return gen



if __name__ == "__main__":
    data_path = "1droplet/230724/pos-7.dat"
    time_range = 10
    time_step = 50
    batch_size = 100

    data_for_pred = DataCreate(data_path, time_range=time_range,time_step_for_pred=time_step,batch_size=batch_size, seed=seed)

    df_for_pred = data_for_pred.scaled_data
    train_size = data_for_pred.train_size
    test_size = data_for_pred.test_size

    train_dataset = data_for_pred.train_dataset
    val_dataset = data_for_pred.val_dataset
    
    # dataloader
    train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            worker_init_fn=seed_worker(seed),
                            generator=torch.Generator().manual_seed(seed),
                        )

    val_loader = DataLoader(
                                val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                worker_init_fn=seed_worker(seed),
                                generator=torch.Generator().manual_seed(seed),
                            )

    input_dim = 2
    output_dim = 2
    hidden_dim = 64
    n_layers = 1

    torch.manual_seed(seed)
    net = RNN(input_dim, output_dim, hidden_dim, n_layers)
    
    save_bool = True

    val_loss, epochs = train_func(net, 100, hidden_dim=hidden_dim, save_bool=save_bool)
    gen = future_prediction(net, df_for_pred, start=train_size, pred_len=len(df_for_pred)-time_step,hidden_dim=hidden_dim, save_bool=save_bool)
    
    # =====log=====
    save_output = [{
        "model": "rnn",
        "hidden dim": hidden_dim,
        "epochs": epochs,
        "time step": time_step,   
        "val loss": round(val_loss, 4),
        "batch size": batch_size,
    }]


    if save_bool:
        with open('1droplet/outputs_rnn/rnn_log(es5).csv','w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = list(save_output[0]))
            writer.writeheader()
            writer.writerows(save_output)