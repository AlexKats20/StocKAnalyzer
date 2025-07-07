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
import talib
from fpdf import FPDF
from sklearn.linear_model import LinearRegression

# === Style ===
tplt = plt.get_cmap('tab10')
PALETTE = tplt.colors
ACCENT = '#00c853'
plt.style.use('default')

# === Utility ===
def draw_pattern_visual(ax, df, idx, pattern_name):
    bar_width = 0.6
    if idx <= 1 or idx >= len(df):
        return
    involved = ([idx-2, idx-1, idx] if '3' in pattern_name or 'MORNINGSTAR' in pattern_name else [idx-1, idx])
    for i in involved:
        if 0 <= i < len(df):
            x0 = df['Date_Num'].iloc[i] - bar_width/2
            y0 = df['Low'].iloc[i]
            h = df['High'].iloc[i] - df['Low'].iloc[i]
            rect = Rectangle((x0, y0), bar_width, h, edgecolor=ACCENT, facecolor=ACCENT, alpha=0.4)
            ax.add_patch(rect)

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
    df = yf.Ticker(ticker).history(period=actual_period, interval=interval,
                                   auto_adjust=False, prepost=True)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['Date_Num'] = mdates.date2num(df.index.to_pydatetime())

    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0); loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean(); avg_loss = loss.rolling(14).mean()
    df['RSI'] = 100 - (100/(1 + avg_gain/avg_loss))

    matches = []
    for name in dir(talib):
        if name.startswith('CDL'):
            result = getattr(talib, name)(df['Open'], df['High'], df['Low'], df['Close'])
            idxs = np.where(result != 0)[0]
            for idx in idxs:
                if (df.index[-1] - df.index[idx]).days <= 10:
                    matches.append({'name': name.replace('CDL',''), 'index': idx})

    bullish_indices = []
    for match in matches:
        if 'HAMMER' in match['name'].upper():
            bullish_indices.append(match['index'])
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
    detect_valid_channels(df_plot, ax1)
    plot_start = len(df) - len(df_plot)
    for idx in bullish_indices:
        if idx >= plot_start:
            idx_plot = idx - plot_start
            ax1.scatter(df_plot['Date_Num'].iloc[idx_plot], df_plot['Low'].iloc[idx_plot]*0.99,
                        color='green', marker='^', s=80, zorder=5)
    plt.tight_layout()
    chart_file = f"{ticker}_chart.png"
    plt.savefig(chart_file)
    plt.close(fig)
    return chart_file

# === Streamlit App ===
st.title("📈 Stock Pattern Analyzer")

uploaded_file = st.file_uploader("Upload your stocks.xlsx", type="xlsx")

if uploaded_file:
    with open("stocks.xlsx", "wb") as f:
        f.write(uploaded_file.read())
    wb = openpyxl.load_workbook("stocks.xlsx")
    ws = wb['Current']
    charts = []
    for row in ws.iter_rows(min_row=2, max_col=3):
        tkr = row[0].value; per = row[1].value or '6mo'; freq = row[2].value or 'daily'
        if tkr:
            chart = analyze_stock(tkr, per, freq)
            charts.append(chart)
    for chart in charts:
        st.image(chart, use_column_width=True)
    st.success("✅ Analysis complete!")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Stock Charts',0,1,'C')
    for chart in charts:
        pdf.add_page()
        pdf.image(chart,10,30,190)
    pdf_file = "stock_report.pdf"
    pdf.output(pdf_file)
    st.download_button("Download PDF Report", open(pdf_file, 'rb'), file_name=pdf_file)
