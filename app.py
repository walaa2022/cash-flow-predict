from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from prophet import Prophet

app = Flask(__name__)

def remove_outliers(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df

def plot_forecast(forecast, title):
    plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, label='Confidence Interval')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Close the plot to free up memory
    
    img_str = base64.b64encode(image_png).decode("utf-8")
    return img_str

@app.route('/')
def index():
    return render_template('index.html')  # Changed from 'plot.html' to 'index.html'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    df = remove_outliers(df, 'Cash_In')
    df = remove_outliers(df, 'Cash_Out')
    
    df_prophet_in = df[['Cash_In']].reset_index().rename(columns={'Date': 'ds', 'Cash_In': 'y'})
    df_prophet_out = df[['Cash_Out']].reset_index().rename(columns={'Date': 'ds', 'Cash_Out': 'y'})
    
    prophet_in = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    prophet_in.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    prophet_in.fit(df_prophet_in)
    
    future_in = prophet_in.make_future_dataframe(periods=12, freq='M')
    forecast_in = prophet_in.predict(future_in)
    
    prophet_out = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    prophet_out.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    prophet_out.fit(df_prophet_out)
    
    future_out = prophet_out.make_future_dataframe(periods=12, freq='M')
    forecast_out = prophet_out.predict(future_out)
    
    forecast_in['cash_out'] = forecast_out['yhat']
    forecast_in['cash_balance'] = forecast_in['yhat'] - forecast_in['cash_out']
    
    cash_in_plot = plot_forecast(forecast_in, "Cash In Forecast")
    cash_out_plot = plot_forecast(forecast_out, "Cash Out Forecast")
    
    results = {
        "forecast_in": forecast_in[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'cash_out', 'cash_balance']].tail(12).to_dict(orient='records'),
        "forecast_out": forecast_out[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_dict(orient='records'),
        "cash_in_plot": cash_in_plot,
        "cash_out_plot": cash_out_plot
    }
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)