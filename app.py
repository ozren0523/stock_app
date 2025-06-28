from flask import Flask, render_template, request, jsonify
from model.model import get_stock_data, predict_stock, create_plot
from services.chatbot_service import get_stock_code_from_text
from flask import send_from_directory
import os
from services.eval_service import evaluate_arima_parallel, evaluate_arima_single

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    ticker_code = ''
    error = None
    prediction = None
    plot_div = None
    arima_result_table = None
    arima_message = None

    if request.method == 'POST':
        if 'company' in request.form:
            company_name = request.form['company'].strip()
            response = get_stock_code_from_text(company_name)
            return render_template('index.html', ticker_code='', error=None, chatbot_response=response)

        elif 'ticker' in request.form:
            ticker_code = request.form['ticker'].strip()
            model_name = request.form.get('model', 'random_forest')
            period = '1y'

            try:
                if model_name == 'arima':
                    company_name = None
                    from data.code_map import company_to_code
                    for name, code in company_to_code.items():
                        if code == ticker_code:
                            company_name = name
                            break
                    if not company_name:
                        company_name = ticker_code 

                    df_arima, arima_message = evaluate_arima_single(company_name, ticker_code)
                    if df_arima is not None:
                        arima_result_table = df_arima.to_html(classes='table table-striped', index=False)
                    else:
                        error = arima_message

                else:
                    df = get_stock_data(ticker_code, period)
                    if df.empty:
                        error = f"銘柄コード「{ticker_code}」のデータが取得できませんでした。"
                    else:
                        prediction = predict_stock(df, model_name)
                        plot_div = create_plot(df, prediction)

            except Exception as e:
                error = f"データ取得または予測でエラーが発生しました: {e}"

            return render_template('index.html',
                                   ticker_code=ticker_code,
                                   prediction=prediction,
                                   plot_html=plot_div,
                                   arima_result_table=arima_result_table,
                                   arima_message=arima_message,
                                   error=error,
                                   chatbot_response=None)

    return render_template('index.html',
                           ticker_code='',
                           prediction=None,
                           plot_html=None,
                           arima_result_table=None,
                           arima_message=None,
                           error=None,
                           chatbot_response=None)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat_query', methods=['POST'])
def chat_query():
    user_text = request.json.get('message', '')
    response = get_stock_code_from_text(user_text)
    return jsonify({'response': response})

@app.route('/download_csv')
def download_csv():
    static_dir = os.path.join(app.root_path, 'static')
    return send_from_directory(directory=static_dir, path='arima_eval_results.csv', as_attachment=True)

@app.route('/evaluate', methods=['GET'])
def evaluate():
    try:
        all_df, summary_df = evaluate_arima_parallel(n=2000, max_workers=20)
        static_dir = os.path.join(app.root_path, 'static')
        os.makedirs(static_dir, exist_ok=True)
        save_path = os.path.join(static_dir, 'arima_eval_results.csv')
        summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')

        message = f"ARIMA評価完了：{len(summary_df)}件の企業でMSE計算が成功しました。"
        return render_template('evaluate.html', message=message, table=summary_df.head(20).to_html(classes='table'))
    except Exception as e:
        return render_template('evaluate.html', message=f"評価中にエラー発生: {e}", table=None)

if __name__ == '__main__':
    app.run(debug=True)
