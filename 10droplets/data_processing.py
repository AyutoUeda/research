from typing import Any
import pandas as pd
import numpy as np
import math

class DataProcessing:
    """入力されたデータを基準(target)からの距離と角度に変換するクラス
    input_data: 入力データ(numpy.array)
    n_nearest_neighbors: 近傍点の数(int)
    
    return:
        labels: 各時刻におけるtargetのラベル(numpy.array), 30度ごとにラベルを振る, 0~11, 12は停止
            [基準からの角度1, 基準からの角度2, ...]
        data_d_and_angle: 各時刻におけるtargetとの距離と角度(numpy.array) 
            [[基準からの距離1, 基準からの角度1, neighboor1の速度ベクトルx, neighboor1の速度ベクトルy, 
            基準からの距離2, 基準からの角度2, neighboor2の速度ベクトルx, neighboor2の速度ベクトルy,...], ...]
    """
    def __init__(self, input_data, n_nearest_neighbors): 
        self.input_data = input_data[1:] # ベクトルの計算のために1つずらす
        self.vectors = np.diff(input_data, axis=0)
        self.n_nearest_neighbors = n_nearest_neighbors # 近傍点の数
        
    def __call__(self):
        self.labels, self.data_d_and_angle = self.data_create()
        
        return self.labels, self.data_d_and_angle
        
    def data_create(self):
        """
        入力データを基準からの距離と角度に変換する関数
        また、基準から最も近い点をn_nearest_neighbors個取得する
        """
        
        labels = []
        data_d_and_angle = []

        for i in range(len(self.input_data) - 1):
            temp_coordinate = self.input_data[i] # i秒目における各点の座標
            x_target = temp_coordinate[0] # # i秒目における基準(target)のx座標
            y_target = temp_coordinate[1] # # i秒目における基準(target)のy座標

            vector = self.vectors[i] # <-- 次の時刻との差
            
            # =====基準が次の時刻にどの方向に進むか=====
            target_vector_x = vector[0] 
            target_vector_y = vector[1]
            
            if target_vector_x == 0 and target_vector_y == 0: # 基準が停止しているとき
                label = 12
            else:
                # 基準が次の時刻にどの方向に進むか
                if target_vector_y == 0 and target_vector_x > 0:
                    target_vector = 0
                elif target_vector_x == 0 and target_vector_y > 0:
                    target_vector = 90
                elif target_vector_y == 0 and target_vector_x < 0:
                    target_vector = 180
                elif target_vector_x == 0 and target_vector_y < 0:
                    target_vector = 270
                else:
                    target_vector = np.arctan(vector[1] / vector[0]) * 180 / np.pi

                # target_vectorがマイナスの値を取ったとき、360度以内に変換
                if target_vector < 0:
                    target_vector = target_vector - 360 * math.floor(target_vector/360)
                
                # 30度ごとにラベルを振る
                label = target_vector // 30

            labels.append(label)

            # =====各点とtargetを比較=====
            temp_list = []
            for j in range(2,temp_coordinate.shape[0], 2):
                # 基準となる座標と比べる座標の差
                x_diff = x_target - temp_coordinate[j]
                y_diff = y_target - temp_coordinate[j+1]
                
                # 基準との距離
                d = math.sqrt(x_diff ** 2 + y_diff ** 2)
                temp_list.append(d)
                
                # 基準との角度
                if x_diff == 0 and y_diff < 0: # 比べる座標が基準より下にあるとき
                    atan = 90
                elif x_diff == 0 and y_diff > 0: # 比べる座標が基準より上にあるとき
                    atan = 270
                elif y_diff == 0 and x_diff < 0: # 比べる座標が基準より右にあるとき
                    atan = 0
                elif y_diff == 0 and x_diff > 0: # 比べる座標が基準より左にあるとき
                    atan = 180
                else:
                    tan = y_diff / x_diff
                    
                    # 基準から見た角度
                    atan = np.arctan(tan) * 180 / np.pi
                
                # 基準からみた角度を追加
                temp_list.append(atan)
                
                # nearest neighboorsが次の時刻にどの方向に進むか（速度ベクトル）
                temp_list.append(vector[j])
                temp_list.append(vector[j+1])
                
            
            # 基準から近い点をn_nearest個取得
            temp_list = np.array(temp_list)
            min_indices = np.argsort(temp_list[::4])[:self.n_nearest_neighbors]
            
            temp_data = []
            for j in min_indices:
                temp_data.append(temp_list[2*j]) # 基準からの距離
                temp_data.append(temp_list[2*j+1]) # 基準からの角度
                temp_data.append(temp_list[2*j+2]) # nearest neighboorsの速度ベクトル x成分
                temp_data.append(temp_list[2*j+3]) # nearest neighboorsの速度ベクトル y成分

            data_d_and_angle.append(temp_data)
        
        return np.array(labels), np.array(data_d_and_angle)
        
    def __len__(self):
        return len(self.data_d_and_angle) # データ数
    
if __name__ == "__main__":
    print("DataProcessing")
    
    def mk_dataframe(path):
        df = pd.read_table(path, sep=" ", header=None)
        df = df.drop(columns=0, axis=1)
        df = df.drop(columns=df.shape[1], axis=1)
        return df

    data_path = "~/Documents/Research/neuralnetwork/position_data/10droplets/231030/pos-0.dat"
    df = mk_dataframe(data_path)
    
    input_data = df.values
    
    data_processing = DataProcessing(input_data, n_nearest_neighbors=3)
    labels, data_d_and_angle = data_processing()
    
    length = len(data_processing)
    
    print(length)