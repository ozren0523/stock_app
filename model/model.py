import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import base64

def get_stock_data(ticker, period):
    """
    yfinanceで指定銘柄・期間の株価データを取得する
    """
    df = yf.download(ticker, period=period)
    return df

def predict_stock(df, model_name):
    """
    仮の予測関数：直近5日の終値平均を翌日の予測とする
    """
    if len(df) < 5:
        return None
    recent_close = df['Close'][-5:]
    prediction = recent_close.mean()
    return round(prediction, 2)

def create_plot(df, prediction):
    """
    過去30日の終値と翌日の予測終値をmatplotlibで描画し、
    base64エンコードしてHTML埋め込み用imgタグの文字列を返す
    """
    if len(df) >= 30:
        recent_df = df.tail(30)
    else:
        recent_df = df

    # Windowsの日本語フォント（メイリオ）を指定
    font_path = "C:/Windows/Fonts/meiryo.ttc"
    font_prop = fm.FontProperties(fname=font_path)

    plt.figure(figsize=(10,5))
    plt.plot(recent_df.index, recent_df['Close'], marker='o', label='過去30日終値')

    if prediction is not None:
        next_date = recent_df.index[-1] + pd.Timedelta(days=1)
        plt.scatter(next_date, prediction, color='red', label='予測終値', zorder=5)
        plt.text(next_date, prediction, '予測', color='red', fontsize=10, ha='center', va='bottom', fontproperties=font_prop)

    plt.title('過去30日間の終値と予測値', fontproperties=font_prop)
    plt.xlabel('日付', fontproperties=font_prop)
    plt.ylabel('価格（USD）', fontproperties=font_prop)
    plt.xticks(rotation=45, fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()

    return f'<img src="data:image/png;base64,{plot_data}" />'
