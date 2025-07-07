from flask import Flask, render_template, request
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import finnhub
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key='YOUR_API_KEY')

@app.route('/', methods=['GET', 'POST'])
def index():
    classification = 'Neutral'
    rsi = 'N/A'
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'AAPL').upper()
        period = request.form.get('period', '6mo')
        freq = request.form.get('freq', 'Daily')

        # Fetch data from Finnhub
        try:
            # Map period to Finnhub resolution
            period_map = {'1mo': '1', '3mo': '3', '6mo': '6', '1y': '12', '2y': '24', '5y': '60', 'max': 'max'}
            months = period_map.get(period.lower(), 'max')
            resolution = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}.get(freq, 'D')

            # Get historical data
            data = finnhub_client.stock_candles(ticker, resolution, int(datetime.now().timestamp()) - (months * 30 * 24 * 60 * 60), int(datetime.now().timestamp()))
            if data['s'] != 'ok' or not data['c']:  # Check status and empty data
                raise ValueError("No data available from Finnhub.")

            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime([datetime.fromtimestamp(t) for t in data['t']]),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })
            df = df.sort_values('Date', ascending=False).reset_index(drop=True)

            # Indicators
            df['RSI'] = ta.rsi(df['Close'], length=14) if len(df) >= 14 else pd.Series([50] * len(df), index=df.index)
            rsi_val = df['RSI'].iloc[-1] if not df['RSI'].isna().iloc[-1] else 50
            classification = 'Bullish' if rsi_val < 35 else 'Bearish' if rsi_val > 65 else 'Neutral'

            # Generate plot (simplified for demo)
            plt.figure(figsize=(10, 6))
            plt.plot(df['Date'], df['Close'], label='Close Price')
            plt.title(f"{ticker} Close Price ({freq})")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=45)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img_str = base64.b64encode(img.getvalue()).decode()

            return render_template('index.html', classification=classification, ticker=ticker, period=period, freq=freq, rsi=f"{rsi_val:.2f}", plot=img_str)

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}. Using default values.")
            return render_template('index.html', classification='Neutral', ticker=ticker, period=period, freq=freq, rsi='N/A', plot=None)

    return render_template('index.html', classification='Neutral', ticker='AAPL', period='6mo', freq='Daily', rsi='N/A', plot=None)

if __name__ == '__main__':
    app.run(debug=True)
