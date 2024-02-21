import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# from callback
from callback.data_processing import DataProcessing
from callback.directon_pred_model import Net
from callback.train_eval import train_func, evaluate_func

import itertools

# 正規化
from sklearn.preprocessing import MinMaxScaler

import csv

# データ作成用関数群
def mk_dataframe(path):
    df = pd.read_table(path, sep=" ", header=None)
    df = df.drop(columns=0, axis=1)
    df = df.drop(columns=df.shape[1], axis=1)
    return df.values


def delete_label_stop(data, labels):
    """ ラベルがstopを意味するデータを削除する
    delete_index: 削除するデータのインデックス
    """
    delete_index = np.where(labels == 8)[0]
    data = np.delete(data, [i for i in delete_index], axis=0)
    labels = np.delete(labels, [i for i in delete_index], axis=0)
    return labels, data


def create_data_instance(data, n_nearest_neighbors, target_list, split_angle, time_range)->dict:
    data_dict = {}
    for target in target_list:
        data_instace = DataProcessing(data, n_nearest_neighbors=n_nearest_neighbors, target_no=target, split_angle=split_angle, time_range=time_range)
        labels, data_d_and_angle = data_instace()
        
        # delete labels stop
        labels, data_d_and_angle = delete_label_stop(data_d_and_angle, labels)
        
        data_dict[target] = labels, data_d_and_angle
        
    return data_dict


if __name__ == "__main__":
    # =====使用するデータの選定=====
    df = mk_dataframe("230728/pos-0.dat")

    # =====ハイパーパラメータ=====
    # データ設定に関するもの
    time_range = 6.0 # データの時間幅
    target_list = [0, 1, 5, 9, 10, 11, 12, 15, 17, 18, 19, 23] # 追跡する粒子の番号
    # 分割する角度
    # 45ならラベルは全部で8つ
    split_angle = 45
    n_nearest_neighbors = 5 # 学習する近傍点の数
    
    # 学習に関するもの
    batch_size_list = [30, 50, 100] # テストするバッチサイズ
    epochs = 1000 # エポック数
    lr = 0.001 # 学習率
    
    
    
    # 正規化するかどうか
    norm_boolean = True
    # ログを保存するかどうか
    save_log_bool = False
    
    
    # =====データ作成=====
    df_to_use = df[:10000:int(time_range*10)]
    
    data_dict = create_data_instance(df_to_use, n_nearest_neighbors, target_list, split_angle, time_range)
    
    labels_list = [data[0] for data in data_dict.values()]
    data_list = [data[1] for data in data_dict.values()]


    # Concatenate the arrays along the rows (axis=0)
    data_d_and_angle = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    

    # 正規化
    if norm_boolean:
        scaler = MinMaxScaler()
        data_d_and_angle = scaler.fit_transform(data_d_and_angle)
    

    X = torch.tensor(data_d_and_angle, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.int64) 

    # 目的変数と入力変数をまとめてdatasetに変換
    dataset = torch.utils.data.TensorDataset(X,target)

    # 各データセットのサンプル数を決定
    # train : val : test = 80% : 10% : 10%
    n_train = int(len(dataset) * 0.8)
    n_val = int((len(dataset) - n_train) * 0.5)
    n_test = len(dataset) - n_train - n_val

    print("train size: {}, val size: {}, test size: {}".format(n_train, n_val, n_test))

    # データセットの分割
    torch.manual_seed(0) #乱数を与えて固定
    train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val,n_test])


    # 指定したバッチサイズごとにデータローダを作成
    train_loaders = [torch.utils.data.DataLoader(train, batch_size, shuffle=True) for batch_size in batch_size_list]
    val_loaders = [torch.utils.data.DataLoader(val, batch_size) for batch_size in batch_size_list]
    test_loaders = [torch.utils.data.DataLoader(test, batch_size) for batch_size in batch_size_list]
    
    
    # =====学習=====
    # バッチサイズごとに学習させる
    for i, batch_size in enumerate(batch_size_list):
    
        train_loader = train_loaders[i]
        val_loader = val_loaders[i]
        test_loader = test_loaders[i]


        # インスタンス化
        net = Net(n_nearest_neighbors=n_nearest_neighbors)

        # 損失関数の設定
        criterion = nn.CrossEntropyLoss()

        # 最適化手法の選択
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss = {
            "train": [],
            "val": [],
            "val acc": []
        }
        
        print("====batch size: {}====".format(batch_size))

        for epoch in range(epochs):
            train_loss, train_acc = train_func(net, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_func(net, val_loader, criterion, device)
            
            loss["train"].append(train_loss)
            loss["val"].append(val_loss)
            loss["val acc"].append(val_acc)
            
            if epoch == 0 or (epoch+1) % 50 == 0:
                print("epoch: {:>3}, train_loss: {:.4f}, train_acc: {:.2f}%, val_loss: {:.4f}, val_acc: {:.2f}%".format(
                    epoch+1, train_loss, train_acc, val_loss, val_acc))
                

        # =====log=====
        save_output = [{
            "target list": target_list,
            "n nearest neighbors": n_nearest_neighbors,
            "time range": time_range,
            "train size": n_train,
            "batch size": batch_size,
            "norm": norm_boolean,
            "epoch": epochs,
            "learning rate": lr,
            "min val loss": min(loss["val"]),
            "max val acc": max(loss["val acc"]),
        }]
        
        if save_log_bool:

            with open('pos0_compare_hyperparams.csv','a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = list(save_output[0]))
                # writer.writeheader()
                writer.writerows(save_output)