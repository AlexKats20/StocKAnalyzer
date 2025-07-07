from flask import Flask, render_template, request
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load static CSV data (download from https://finance.yahoo.com/quote/AAPL/history/export)
df = pd.read_csv('aapl_historical.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df = df.sort_values('Date', ascending=False).reset_index(drop=True)

# Precompute indicators
df['RSI'] = ta.rsi(df['Close'], length=14) if len(df) >= 14 else pd.Series([50] * len(df), index=df.index)

@app.route('/', methods=['GET', 'POST'])
def index():
    classification = 'Neutral'
    if request.method == 'POST':
        period = request.form.get('period', '6mo')
        freq = request.form.get('freq', 'Daily')
        
        # Filter data based on period
        period_map = {'1mo': 1, '3mo': 3, '6mo': 6, '1y': 12, '2y': 24, '5y': 60, 'max': 1000}
        days = period_map.get(period.lower(), 1000)
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(months=days)
        filtered_df = df[df['Date'] >= start_date].copy()
        
        if not filtered_df.empty:
            rsi_val = filtered_df['RSI'].iloc[-1] if not filtered_df['RSI'].isna().iloc[-1] else 50
            classification = 'Bullish' if rsi_val < 35 else 'Bearish' if rsi_val > 65 else 'Neutral'
        else:
            classification = 'Neutral (No data for selected period)'

        return render_template('index.html', classification=classification, period=period, freq=freq, rsi=f"{rsi_val:.2f}" if 'rsi_val' in locals() else "N/A")

    return render_template('index.html', classification=classification, period='6mo', freq='Daily', rsi="N/A")

if __name__ == '__main__':
    app.run(debug=True)