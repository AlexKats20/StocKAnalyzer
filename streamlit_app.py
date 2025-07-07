# streamlit_app.py

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import openpyxl
import os
import numpy as np
from datetime import datetime
from matplotlib.gridspec import GridSpec
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

# === Stock Analysis ===
def analyze_stock(ticker, period, freq_str):
    try:
        freq_map = {'daily':'1d', 'weekly':'1wk', 'monthly':'1mo'}
        interval = freq_map.get(freq_str.lower(), '1d')
        actual_period = 'max' if str(period).lower() == 'max' else period

        df = yf.Ticker(ticker).history(period=actual_period, interval=interval,
                                       auto_adjust=False, prepost=True)
        df = df[['Open','High','Low','Close','Volume']].dropna()
        df['Date_Num'] = mdates.date2num(df.index.to_pydatetime())

        # === Indicators ===
        df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['MA100'] = df['Close'].rolling(100, min_periods=1).mean()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['Signal'] = macd['MACDs_12_26_9']

        # === Classification ===
        rsi_val = df['RSI'].iloc[-1]
        macd_val = df['MACD'].iloc[-1]
        sig_val = df['Signal'].iloc[-1]
        classification = (
            'Bullish' if rsi_val<35 or (macd_val>sig_val and macd_val>-0.5)
            else 'Bearish' if rsi_val>65 or (macd_val<sig_val and macd_val<0.5)
            else 'Neutral'
        )

        # === Bullish markers ===
        bullish_indices = []
        if rsi_val < 30:
            bullish_indices.append(len(df)-1)
        if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] and df['MA20'].iloc[-2] < df['MA50'].iloc[-2]:
            bullish_indices.append(len(df)-1)

        df_plot = df.tail(250 if interval=='1d' else 52 if interval=='1wk' else 24)

        fig = plt.figure(figsize=(12,8))
        gs = GridSpec(4,1, height_ratios=[3,1,1,1])

        ax1 = fig.add_subplot(gs[0])
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
                ax1.scatter(df_plot['Date_Num'].iloc[idx_plot],
                            df_plot['Low'].iloc[idx_plot] * 0.99,
                            color='green', marker='^', s=80, zorder=5)

        ax1.set_title(f"{ticker} | {freq_str.capitalize()} Chart")
        ax1.set_ylabel("Price")
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', lw=0.5, alpha=0.4)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(df_plot['Date_Num'], df_plot['Volume'], color='gray')

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(df_plot['Date_Num'], df_plot['RSI'], color=PALETTE[4], label='RSI')
        ax3.axhline(70, color='red', linestyle='--')
        ax3.axhline(30, color='green', linestyle='--')
        ax3.legend(loc='upper left')

        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(df_plot['Date_Num'], df_plot['MACD'], color=PALETTE[5], label='MACD')
        ax4.plot(df_plot['Date_Num'], df_plot['Signal'], color=PALETTE[6], label='Signal')
        ax4.legend(loc='upper left')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        fig.autofmt_xdate()
        plt.tight_layout()
        chart_path = f"{ticker}_chart.png"
        plt.savefig(chart_path, dpi=100)
        plt.close(fig)

        return classification, chart_path, rsi_val, macd_val, sig_val, "N/A", ""
    except Exception as e:
        st.error(f"{ticker} error: {e}")
        return "Error", None, None, None, None, None, None

# === Streamlit App ===
st.title("📊 Stock Pattern Analyzer")

uploaded_file = st.file_uploader("Upload your stocks.xlsx file", type="xlsx")
if uploaded_file:
    with open("stocks.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    wb = openpyxl.load_workbook("stocks.xlsx")
    ws_current = wb['Current']
    if 'History' not in wb.sheetnames:
        ws_history = wb.create_sheet('History')
        ws_history.append(['Date','Ticker','Period','Freq','Classification','RSI','MACD','Signal','Pattern'])
    else:
        ws_history = wb['History']
    today = datetime.now().strftime('%Y-%m-%d')

    charts = []
    for row in ws_current.iter_rows(min_row=2, max_col=3):
        tkr = row[0].value; per = row[1].value or '6mo'; freq = row[2].value or 'daily'
        if tkr:
            cl, path, r, m, s, patt, strg = analyze_stock(tkr, per, freq)
            ws_current.cell(row=row[0].row, column=4).value = cl
            ws_current.cell(row=row[0].row, column=5).value = path
            ws_current.cell(row=row[0].row, column=6).value = patt
            ws_current.cell(row=row[0].row, column=7).value = strg
            if path and os.path.exists(path):
                charts.append(path)
            ws_history.append([today,tkr,per,freq,cl,round(r or 0,2),round(m or 0,2),round(s or 0,2),patt])

    wb.save('stocks.xlsx')

    # ✅ Debug: show collected charts
    st.write("Charts collected:", charts)

    # === Build PDF ===
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Stock Pattern Summary Report',0,1,'C')
    pdf.ln(5)
    pdf.set_font('Arial',size=12)
    pdf.cell(30,10,'Date:'); pdf.cell(50,10,today,0,1)
    pdf.ln(5)

    pdf.set_font('Arial','B',11)
    pdf.cell(30,10,'Ticker',1)
    pdf.cell(25,10,'Period',1)
    pdf.cell(25,10,'Freq',1)
    pdf.cell(30,10,'Classif',1)
    pdf.cell(30,10,'Pattern',1)
    pdf.ln()
    pdf.set_font('Arial',size=10)
    for row in ws_current.iter_rows(min_row=2, max_col=7, values_only=True):
        tkr, per, freq, cl, _, patt, _ = row
        pdf.cell(30,10,str(tkr),1)
        pdf.cell(25,10,str(per),1)
        pdf.cell(25,10,str(freq),1)
        pdf.cell(30,10,str(cl),1)
        pdf.cell(30,10,str(patt),1)
        pdf.ln()

    # ✅ Embed chart pages
    for chart in charts:
        pdf.add_page()
        pdf.image(chart, 10, 30, 190)

    pdf.output("stock_report.pdf")

    st.success("✅ Analysis complete!")
    with open("stock_report.pdf","rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="stock_report.pdf")
