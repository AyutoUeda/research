## 油滴の運動予測

### RNN

### TCN

- TCNは`2370724/pos-7.dat`のデータに対して実行
- `tcn_for_1droplet.py`を使用
- 画像は`pred_oscillated/`, `oscillated_loss/`に格納
- 結果は`new_model_log_oscillated.csv`に格納
  
#### ハイパーパラメータ

- epochs: parse argで指定
- batch size: parse argで指定
- time step(train dataの時間間隔): parse argで指定 
- level: parse argで指定
- learning rate: 1e-3
- hidden dim: 15
- kernel size: 5

#### コード実行

```
>>>$ python3 tcn_for_1droplet.py --epochs 5 --batch_size 100 --time_step 300
```


## 対象の油滴を方向によるラベリング

### やったこと1

1. ラベルを45度刻みで設定。(d, theta, v_x, v_y)を特徴量としてモデルに投入。
   ==> lossが減らない
2. thetaを360で割り、正規化
   ==> 多少lossが減ったものの、正答率20%ほど

**考え**
===> 学習データを変える(ex. 基準からの距離x成分, 基準からの距離y成分, 基準の速度xベクトル, 基準の速度yベクトル)

### やったこと2

学習データを変える(ex. 基準からの距離x成分, 基準からの距離y成分, 基準の速度xベクトル, 基準の速度yベクトル)
==> 多少はlossが減った (正答率23%)

**考え**
===> データ数を増やす、ラベルの偏りをなくす, paramsを減らす
https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=ja#%E9%81%8E%E5%AD%A6%E7%BF%92%E3%81%AE%E3%83%87%E3%83%A2


精度改善
- 時間間隔を広げる
- サンプルとなる油滴の数を増やす