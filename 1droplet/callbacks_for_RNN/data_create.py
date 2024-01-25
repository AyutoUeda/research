import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
import random

class DataCreate:
    def __init__(self, data_path, time_range, time_step_for_pred, batch_size:100, seed):
        self.data_path = data_path
        self.time_range = time_range
        self.scaled_data = self.create_scaled_array(data_path, time_range)
        self.train_dataset, self.val_dataset = self.create_dataset(self.scaled_data, time_step_for_pred, seed)
        # self.train_loader, self.val_loader = self.create_dataloader(batch_size, self.train_dataset, self.val_dataset, seed)
        
        
    def create_scaled_array(self, data_path, time_range):
        df = pd.read_table(data_path, sep=" ", header=None)
        df = df.drop(columns=0, axis=1)
        df = df.drop(columns=3, axis=1)
        df_ = df.values # array
        
        # scaled
        scaler = MinMaxScaler(feature_range = (0, 1))
        df_scaled = scaler.fit_transform(df_)
        return df_scaled[:36000:time_range]
    
    def create_dataset(self, df_for_pred, time_step, seed):
        train_size = int(len(df_for_pred) * 0.9)
        test_size = len(df_for_pred) - train_size
        
        self.train_size = train_size
        self.test_size = test_size
        
        n_sample = train_size - time_step - 1 # train データセットのサイズ

        #シーケンシャルデータを格納する箱を用意(入力)
        input_data = np.zeros((n_sample, time_step, 2))
        #シーケンシャルデータを格納する箱を用意(正解)
        correct_input_data = np.zeros((n_sample, 2))


        for i in range(n_sample):
            input_data[i] = df_for_pred[i:i+time_step, :].reshape(-1, 2)
            correct_input_data[i] = df_for_pred[i+time_step:i+time_step+1, :]
            
            
        # -----データローダ-----
        input_data = torch.tensor(input_data, dtype=torch.float) #Tensor化(入力)
        correct_data = torch.tensor(correct_input_data, dtype=torch.float) #Tensor化(正解)

        dataset = torch.utils.data.TensorDataset(input_data, correct_data) #データセット作成

        n_sample_train = len(dataset)
        n_train = int(n_sample * 0.8)
        n_val = n_sample_train - n_train

        print("n_train:{}, n_val:{}".format(n_train, n_val))

        train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                                [n_train, n_val], 
                                                                generator=torch.Generator().manual_seed(seed))
        
        return train_dataset, val_dataset
    
    def create_dataloader(self, batch_size, train_dataset, val_dataset, seed):
        # datasetからバッチごとに取り出す
        train_loader = DataLoader(
                                    train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2,
                                    worker_init_fn=self.seed_worker(seed),
                                    generator=torch.Generator().manual_seed(seed),
                                )

        val_loader = DataLoader(
                                    val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2,
                                    worker_init_fn=self.seed_worker(seed),
                                    generator=torch.Generator().manual_seed(seed),
                                )
        
    def seed_worker(self,seed):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)