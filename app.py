from flask import Flask, render_template, request
from model.model import get_stock_data, predict_stock, create_plot

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        period = request.form['period']
        model_name = request.form['model']

        df = get_stock_data(ticker, period)
        prediction = predict_stock(df, model_name)
        plot_div = create_plot(df, prediction)

        return render_template('result.html', ticker=ticker, prediction=prediction, plot_div=plot_div, model=model_name)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
