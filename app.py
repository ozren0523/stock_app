from flask import Flask, render_template, request
from model.model import get_stock_data, predict_stock, create_plot
from data.code_map import company_to_code

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
            ticker_code = company_to_code.get(company_name)
            if not ticker_code:
                error = f"企業名「{company_name}」の銘柄コードが見つかりません。"
            return render_template('index.html', ticker_code=ticker_code or '', error=error)

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
                                   error=error)

    return render_template('index.html',
                           ticker_code='',
                           prediction=None,
                           plot_html=None,
                           error=None)

if __name__ == '__main__':
    app.run(debug=True)
