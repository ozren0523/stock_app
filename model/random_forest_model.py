import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def random_forest_predict(df):
    """
    終値（'Close'）の予測をランダムフォレストで行うシンプルな例。
    特徴量は過去の終値のラグ特徴量（1日前、2日前）を作成し予測。
    """

    # データ前処理：終値のラグ特徴量作成
    df = df.copy()
    df['Close_1'] = df['Close'].shift(1)
    df['Close_2'] = df['Close'].shift(2)
    df.dropna(inplace=True)

    X = df[['Close_1', 'Close_2']]
    y = df['Close']

    # 学習用・テスト用に分割（ここではシンプルに7割学習、3割テスト）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # テストデータの終値を予測
    preds = model.predict(X_test)

    # 予測値の最後の1つ（最新日の予測）を返す例
    return round(preds[-1], 2)
