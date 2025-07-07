# streamlit_app.py

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import openpyxl
import os
import numpy as np
from datetime import datetime
from openpyxl.drawing.image import Image as XLImage
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_ta as ta
from fpdf import FPDF
from sklearn.linear_model import LinearRegression

# === Style ===
tplt = plt.get_cmap('tab10')
PALETTE = tplt.colors
ACCENT = '#00c853'
plt.style.use('default')

# === Channel detection ===
def detect_valid_channels(df, ax1, lookback=50, stride=5, min_slope=0.005):
    if len(df) < lookback:
        return
    last_channel_end = -1
    for start in range(0, len(df)-lookback, stride):
        if start < last_channel_end:
            continue
        end = start + lookback
        x = np.arange(lookback).reshape(-1,1)
        y_high = df['High'].iloc[start:end].values.reshape(-1,1)
        y_low = df['Low'].iloc[start:end].values.reshape(-1,1)
        reg_high = LinearRegression().fit(x, y_high)
        reg_low = LinearRegression().fit(x, y_low)
        slope_high, slope_low = reg_high.coef_[0][0], reg_low.coef_[0][0]
        upper_line = reg_high.predict(x).flatten()
        lower_line = reg_low.predict(x).flatten()
        channel_width = upper_line - lower_line
        width_var = np.std(channel_width)
        close_prices = df['Close'].iloc[start:end].values
        within_channel = np.mean((close_prices >= lower_line) & (close_prices <= upper_line))
        if (abs(slope_high) > min_slope and abs(slope_low) > min_slope) and width_var < 4.0 and within_channel > 0.4:
            ax1.plot(df['Date_Num'].iloc[start:end], upper_line, '--', lw=1.2, color=PALETTE[3])
            ax1.plot(df['Date_Num'].iloc[start:end], lower_line, '--', lw=1.2, color=PALETTE[3])

# === Analysis function ===
def analyze_stock(ticker, period, freq_str):
    freq_map = {'daily':'1d', 'weekly':'1wk', 'monthly':'1mo'}
    interval = freq_map.get(freq_str.lower(), '1d')
    actual_period = 'max' if str(period).lower() == 'max' else period
    st.write(f"⏳ Downloading {ticker} ({actual_period}, {freq_str})")
    df = yf.Ticker(ticker).history(period=actual_period, interval=interval,
                                   auto_adjust=False, prepost=True)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['Date_Num'] = mdates.date2num(df.index.to_pydatetime())

    # Indicators
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA100'] = df['Close'].rolling(100, min_periods=1).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal'] = macd['MACDs_12_26_9']

    # Bullish markers
    bullish_indices = []
    if df['RSI'].iloc[-1] < 30:
        bullish_indices.append(len(df)-1)
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] and df['MA20'].iloc[-2] < df['MA50'].iloc[-2]:
        bullish_indices.append(len(df)-1)

    df_plot = df.tail(250 if interval=='1d' else 52 if interval=='1wk' else 24)

    fig, ax1 = plt.subplots(figsize=(10,6))
    ohlc = df_plot[['Date_Num','Open','High','Low','Close']].values
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup=PALETTE[0], colordown=PALETTE[1])
    ax1.plot(df_plot['Date_Num'], df_plot['MA20'], color=PALETTE[0], lw=2, label='MA20')
    ax1.plot(df_plot['Date_Num'], df_plot['MA50'], color=PALETTE[1], lw=1.5, linestyle='--', label='MA50')
    ax1.plot(df_plot['Date_Num'], df_plot['MA100'], color=PALETTE[2], lw=1.5, linestyle=':', label='MA100')
    detect_valid_channels(df_plot, ax1)

    plot_start = len(df) - len(df_plot)
    for idx in bullish_indices:
        if idx >= plot_start:
            idx_plot = idx - plot_start
            ax1.scatter(df_plot['Date_Num'].iloc[idx_plot], df_plot['Low'].iloc[idx_plot] * 0.99,
                        color='green', marker='^', s=80, zorder=5)

    ax1.grid(True, linestyle=':', lw=0.5, alpha=0.4)
    ax1.legend(loc='upper left')
    plt.tight_layout()
    chart_file = f"{ticker}_chart.png"
    plt.savefig(chart_file)
    plt.close(fig)
    return chart_file

# === Streamlit app ===
st.title("📊 Stock Pattern Analyzer")

uploaded_file = st.file_uploader("Upload your stocks.xlsx file", type="xlsx")

if uploaded_file:
    with open("stocks.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    wb = openpyxl.load_workbook("stocks.xlsx")
    ws = wb['Current']
    charts = []
    for row in ws.iter_rows(min_row=2, max_col=3):
        tkr = row[0].value
        per = row[1].value or '6mo'
        freq = row[2].value or 'daily'
        if tkr:
            chart = analyze_stock(tkr, per, freq)
            charts.append(chart)

    for chart in charts:
        st.image(chart, use_column_width=True)

    st.success("✅ Analysis complete!")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Stock Charts',0,1,'C')
    for chart in charts:
        pdf.add_page()
        pdf.image(chart,10,30,190)
    pdf_file = "stock_report.pdf"
    pdf.output(pdf_file)
    st.download_button("Download PDF Report", open(pdf_file, 'rb'), file_name=pdf_file)
