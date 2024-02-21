## ファイルの説明
研究に使用したフォルダは ```/1droplet``` と ```/30droplets```.  
### 1droplet
#### 1droplet/RNN_for_1droplet.py
1つの油滴に対して、RNNを適応して時系列予測.  
結果をcsvファイルに出力.

#### 1droplet/tcn_for_1droplet.py
1つの油滴に対して、TCNを適応して時系列予測.  
結果をcsvファイルに出力.

#### 1droplet/callback
モデルを記述したファイルとtrainデータを作成するファイルを格納

### 30droplets
#### 30droplets/30droplets.py
運動方向予測を実行するファイル. 

#### 30droplets/callback/directon_pred_model.py
モデルを記述

#### 30droplets/callback/train_eval.py
学習・評価用関数を記述


## 油滴の運動予測

### RNN

#### ハイパーパラメータ

- epochs: early stopping (5回更新がなければ打ち切り)
- hidden dim: tuning
- kernel size: tuning
- time step(train dataの時間間隔): tuning
- batch size: 100
- learning rate: 1e-3


### TCN

- TCNは`2370724/pos-7.dat`のデータに対して実行
- `tcn_for_1droplet.py`を使用
- 画像は`pred_oscillated/`, `oscillated_loss/`に格納
- 結果は`new_model_log_oscillated.csv`に格納
  
#### ハイパーパラメータ

- epochs: early stopping (5回更新がなければ打ち切り)
- level: tuning
- hidden dim: tuning
- kernel size: tuning
- batch size: 100
- time step(train dataの時間間隔): 300
- learning rate: 1e-3

#### 結果

- kernel size は大きいほうが精度がよい
- levelを上げれば良いというわけではない

#### コード実行例

```
>>>$ python3 tcn_for_1droplet.py --epochs 5 --batch_size 100 --time_step 300
```
