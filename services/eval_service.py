import random
import pandas as pd
import yfinance as yf
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.code_map import company_to_code
import numpy as np
import traceback
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima.model import ARIMA

CSV_PATH = "arima_evaluation_results.csv"

def evaluate_arima_single(name, code, test_size=7, future_days=1):
    print(f"--- ARIMA評価開始: {name} ({code}) ---")
    df = yf.download(code, period="90d", interval="1d", progress=False, auto_adjust=True)

    if df.empty or len(df) < 40:
        msg = f"{name}（{code}）: データが不足しています。"
        print(msg)
        return None, msg

    close_series = df['Close'].dropna()
    close_series.index = pd.to_datetime(close_series.index)

    train = close_series[:-test_size]
    test = close_series[-test_size:]

    try:
        order = (1, 1, 1)
        print(f"固定ARIMAオーダーを使用: {order}")

        history = [float(x) for x in train.values]
        predictions = []

        for i in range(test_size + future_days):
            print(f"Step {i + 1} / {test_size + future_days}")
            model_fit = ARIMA(history, order=order).fit()
            pred = model_fit.forecast()
            pred_scalar = float(pred[0]) 
            predictions.append(pred_scalar)
            history.append(pred_scalar)
            print(f"history 最新: {history[-3:]}")

        test_pred = predictions[:test_size]
        future_pred = predictions[test_size:]

        mse = mean_squared_error(test, test_pred)
        print(f"MSE計算完了: {mse}")

        all_dates = list(test.index) + [test.index[-1] + pd.DateOffset(days=i) for i in range(1, future_days + 1)]
        actuals = list(test.values.flatten()) + [np.nan] * future_days
        predicted = test_pred + future_pred

        result_df = pd.DataFrame({
            '企業名': [name] * len(all_dates),
            'date': [d.strftime('%Y-%m-%d') for d in all_dates],
            'actual': actuals,
            'predicted': predicted
        })

        message = f"{name}（{code}）のARIMA予測完了。MSE: {mse:.4f}、未来{future_days}日分の予測追加。"
        print(message)
        return result_df, message

    except Exception as e:
        print(f"エラー発生: {e}")
        traceback.print_exc()
        return None, f"{name}（{code}）: ARIMA予測中にエラーが発生しました: {e}"


def arima_evaluate_yf(name, code, test_size=7):
    try:
        df = yf.download(code, period="90d", interval="1d", progress=False, auto_adjust=False)
        if df.empty or len(df) < 40:
            print(f"[{code}] データ不足でスキップ")
            return None

        close_series = df['Close'].dropna()
        close_series.index = pd.to_datetime(close_series.index)

        train = close_series[:-test_size]
        test = close_series[-test_size:]

        model = auto_arima(train, seasonal=False, suppress_warnings=True, error_action='ignore')
        order = model.order

        predictions = []
        for i in range(test_size):
            history = close_series[:-(test_size - i)]

            try:
                model = auto_arima(history, seasonal=False, suppress_warnings=True, error_action='ignore')
                fitted_model = model.fit(history)
                pred = fitted_model.predict(n_periods=1)
                predictions.append(pred[0])
            except Exception as inner_e:
                print(f"[{code}] 内部予測エラー: {inner_e}")
                return None

        print(f"企業: {name}, コード: {code}, MSE計算前")
        print(f"test: {test.values}")
        print(f"pred: {predictions}")

        mse = mean_squared_error(test, predictions)
        return {'企業名': name, '銘柄コード': code, 'MSE': round(mse, 4)}

    except Exception as e:
        print(f"[{code}] エラー: {e}")
        return None

def evaluate_arima_parallel(n=2000, max_workers=20, test_size=7, future_days=1):
    selected = random.sample(list(company_to_code.items()), k=n)
    all_dfs = []
    summary_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(evaluate_arima_single, name, code, test_size, future_days): (name, code)
            for name, code in selected
        }

        for future in as_completed(future_to_name):
            name, code = future_to_name[future]
            try:
                result_df, message = future.result()
                print(message)
                if result_df is not None:
                    all_dfs.append(result_df)
                    import re
                    mse_match = re.search(r"MSE: ([\d\.]+)", message)
                    mse_val = float(mse_match.group(1)) if mse_match else None
                    summary_list.append({
                        "企業名": name,
                        "銘柄コード": code,
                        "MSE": mse_val
                    })
                else:
                    summary_list.append({
                        "企業名": name,
                        "銘柄コード": code,
                        "MSE": None
                    })
            except Exception as e:
                print(f"[{name} ({code})] 並列処理中にエラー: {e}")
                summary_list.append({
                    "企業名": name,
                    "銘柄コード": code,
                    "MSE": None
                })

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df.to_csv("arima_predictions_all.csv", index=False, encoding="utf-8-sig")
    else:
        all_df = None

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv("arima_summary_mse.csv", index=False, encoding="utf-8-sig")

    return all_df, summary_df


def get_csv_path():
    return CSV_PATH
