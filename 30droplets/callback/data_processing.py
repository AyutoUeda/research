import pandas as pd
import numpy as np
import math

class DataProcessing:
    """入力されたデータを基準(target)からの距離と角度に変換するクラス
    
    Args:
        input_data: 入力データ(```numpy.array```) \n
        n_nearest_neighbors: 近傍点の数(```int```) \n
        target_no: 基準となる点の番号(```int```) # 0から始まる \n
        split_angle: ラベルを振る角度の間隔(```int```) \n
        time_range: 何秒間隔のデータを使うか(```int```) \n
    
    Returns:
        labels: 各時刻におけるtargetのラベル(```numpy.array```), 30度ごとにラベルを振る, 0~11, 12は停止 \n
            [基準からの角度1, 基準からの角度2, ...]
        data_d_and_angle: 各時刻における基準と比べる座標との距離・角度(```numpy.array```)、比較対象がどの方向に進んでいるか \n
            [[基準からの距離1, 基準からの角度1, neighbor1の速度ベクトルx, neighbor1の速度ベクトルy, 
            基準からの距離2, 基準からの角度2, neighbor2の速度ベクトルx, neighbor2の速度ベクトルy,...], ...]
    
    Example:
        >>> input_data = np.array([[x1, y1, x2, y2, x3, y3, ...], [x1, y1, x2, y2, x3, y3, ...], ...]) \n
        >>> data_processing = DataProcessing(input_data, target_no=0, n_nearest_neighbors=9) \n
        >>> labels, data_d_and_angle = data_processing() \n
        
    Note:
        基準となる点の番号(target_no)は0から始まる
    """
    def __init__(self, input_data, n_nearest_neighbors, target_no=0, split_angle=30, time_range=1): 
        self.input_data = input_data[:-1] # ベクトルの計算のために1つずらす
        self.vectors = np.diff(input_data, axis=0) # 次の時刻との差（速度ベクトル）
        self.n_nearest_neighbors = n_nearest_neighbors # 近傍点の数
        self.target_no = target_no # 基準となる点の番号(default: 0)
        self.split_angle = split_angle # ラベルを振る角度の間隔(default: 30度)
        self.time_range = time_range # 何秒間隔のデータを使うか(default: 1s)
        
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
            x_target = temp_coordinate[self.target_no*2] # # i秒目における基準(target)のx座標
            y_target = temp_coordinate[self.target_no*2+1] # # i秒目における基準(target)のy座標

            vector = self.vectors[i] # <-- 次の時刻との差
            
            # =====基準が次の時刻にどの方向に進むか=====
            target_vector_x = vector[self.target_no*2] 
            target_vector_y = vector[self.target_no*2+1]
            
            if target_vector_x == 0 and target_vector_y == 0: # 基準が停止しているとき
                label = 360 // self.split_angle 
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
                    target_vector = np.arctan(target_vector_y / target_vector_x) * 180 / np.pi

                    # target_vectorがマイナスの値を取ったとき、360度以内に変換
                    if target_vector < 0:
                        target_vector = target_vector - 180 * math.floor(target_vector/360)
                    
                    # 基準が下方向に進んだとき
                    if target_vector_y < 0:
                        target_vector += 180
                
                # split_angle度ごとにラベルを振る
                label = target_vector // self.split_angle

            labels.append(label)

            # =====各点とtargetを比較=====
            temp_list = [] # 基準と各点との比較を格納するリスト
            for j in range(0,temp_coordinate.shape[0], 2):
                """
                temp_listには ``d, x_diff, y_diff, atan, vector_x, vector_y`` の順に格納される
                """
                if j == self.target_no * 2: # 基準は除く
                    continue
                else:
                    # 基準となる座標と比べる座標の差
                    x_diff = x_target - temp_coordinate[j]
                    y_diff = y_target - temp_coordinate[j+1]
                    
                    # 基準との距離
                    d = math.sqrt(x_diff ** 2 + y_diff ** 2)
                    
                    temp_list.append(d)
                    
                    # 基準との距離x成分
                    temp_list.append(x_diff)
                    # 基準との距離y成分
                    temp_list.append(y_diff)
                
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
                        # atan がマイナスの値を取ったとき、360度以内に変換
                        if atan < 0:
                            atan = atan - 180 * math.floor(atan/360)
                            
                        # 比べる座標が基準より下にある場合
                        if y_diff > 0:
                            atan = atan + 180
                
                    # 基準からみた角度を追加
                    temp_list.append(atan)
                    
                    # nearest neighboorsが次の時刻にどの方向に進むか（速度ベクトル）
                    temp_list.append(vector[j] / self.time_range)
                    temp_list.append(vector[j+1] / self.time_range)
                
            
            # 基準から近い点をn_nearest個取得
            temp_array = np.array(temp_list) # numpy配列に変換
            min_indices = np.argsort(temp_array[::6])[:self.n_nearest_neighbors] # 基準からの距離でソートし、n_nearest_neighbors個取得
            
            temp_data = []
            for j in min_indices:
                # temp_data.append(temp_array[6*j]) # 基準からの距離
                temp_data.append(temp_array[6*j+1]) # 基準との距離x成分
                temp_data.append(temp_array[6*j+2]) # 基準との距離y成分
                # temp_data.append(temp_array[6*j+3]) # 基準からの角度
                temp_data.append(temp_array[6*j+4]) # nearest neighboorsの速度ベクトル x成分
                temp_data.append(temp_array[6*j+5]) # nearest neighboorsの速度ベクトル y成分

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
    print(np.diff(input_data, axis=0)[0])
    data_processing = DataProcessing(input_data, target_no=0,n_nearest_neighbors=9)
    labels, data_d_and_angle = data_processing()
    
    print(data_d_and_angle[0])
    
    length = len(data_processing)
    print(labels)
    print(length)