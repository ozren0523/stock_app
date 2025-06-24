from flask import Flask, render_template, request, jsonify
from model.model import get_stock_data, predict_stock, create_plot
from services.chatbot_service import get_stock_code_from_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    ticker_code = ''
    error = None
    prediction = None
    plot_div = None

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
                                   error=error,
                                   chatbot_response=None)

    return render_template('index.html',
                           ticker_code='',
                           prediction=None,
                           plot_html=None,
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

if __name__ == '__main__':
    app.run(debug=True)
