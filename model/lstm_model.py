import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

def lstm_predict(df, return_rmse=False):
    """
    株価の終値をLSTMで予測（過去60日分を使用）
    
    Args:
        df (pd.DataFrame): 'Close'列を含むデータフレーム
        return_rmse (bool): Trueの場合、予測誤差（RMSE）も返す
        
    Returns:
        float: 翌日の終値予測値（小数点2桁）
        float (optional): RMSE（Root Mean Squared Error）
    """
    df = df.copy()

    if len(df) < 100:
        raise ValueError("最低でも100日分のデータが必要です（推奨: 200日以上）")

    # スケーリング
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 特徴量とラベルの生成
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 学習データとテストデータに分割（8:2）
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # LSTMモデル定義
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 学習
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # テストデータで予測し精度確認
    test_preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, test_preds))

    # 未来1日の予測
    last_60 = scaled_data[-60:]
    last_60 = np.reshape(last_60, (1, 60, 1))
    pred_scaled = model.predict(last_60)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]

    if return_rmse:
        return round(float(pred), 2), round(float(rmse), 4)
    else:
        return round(float(pred), 2)
