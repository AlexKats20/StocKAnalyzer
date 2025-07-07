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

# === Streamlit app ===
st.set_page_config(page_title="Stock Pattern Analyzer")
st.title("📊 Stock Pattern Analyzer (with PDF)")

uploaded_file = st.file_uploader("Upload your 'stocks.xlsx' file", type="xlsx")

if uploaded_file:
    with open("stocks.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    tplt = plt.get_cmap('tab10')
    PALETTE = tplt.colors
    ACCENT = '#ff6f00'
    plt.style.use('default')

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
                rect = Rectangle((x0, y0), bar_width, h,
                                 edgecolor=ACCENT, facecolor=ACCENT, alpha=0.4)
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
            y_low  = df['Low'].iloc[start:end].values.reshape(-1,1)
            reg_high = LinearRegression().fit(x, y_high)
            reg_low  = LinearRegression().fit(x, y_low)
            slope_high, slope_low = reg_high.coef_[0][0], reg_low.coef_[0][0]
            upper_line = reg_high.predict(x).flatten()
            lower_line = reg_low.predict(x).flatten()
            channel_width = upper_line - lower_line
            width_var = np.std(channel_width)
            close_prices = df['Close'].iloc[start:end].values
            within_channel = np.mean((close_prices >= lower_line) & (close_prices <= upper_line))
            is_up = slope_high > min_slope and slope_low > min_slope
            is_down = slope_high < -min_slope and slope_low < -min_slope
            if (is_up or is_down) and width_var < 4.0 and within_channel > 0.4:
                color = PALETTE[3]
                date_nums = df['Date_Num'].iloc[start:end]
                ax1.plot(date_nums, upper_line, '--', lw=1.2, color=color)
                ax1.plot(date_nums, lower_line, '--', lw=1.2, color=color)

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
        df['MA100'] = df['Close'].rolling(100, min_periods=1).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta>0,0)
        loss = -delta.where(delta<0,0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        df['RSI'] = 100 - (100/(1 + avg_gain/avg_loss))
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()

        matches = []
        for name in dir(talib):
            if name.startswith('CDL'):
                result = getattr(talib, name)(df['Open'], df['High'], df['Low'], df['Close'])
                idxs = np.where(result != 0)[0]
                for idx in idxs:
                    if (df.index[-1] - df.index[idx]).days <= 10:
                        matches.append({
                            'name': name.replace('CDL',''),
                            'index': idx,
                            'date': df.index[idx].date(),
                            'strength': int(result.iloc[idx])
                        })

        if matches:
            detected_pattern = matches[-1]['name']
            strength = f"{detected_pattern}={matches[-1]['strength']}"
        else:
            detected_pattern = 'None'
            strength = ''

        rsi_val, macd_val, sig_val = df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['Signal'].iloc[-1]
        classification = 'Bullish' if rsi_val<35 or (macd_val>sig_val and macd_val>-0.5) else 'Bearish' if rsi_val>65 or (macd_val<sig_val and macd_val<0.5) else 'Neutral'

        bullish_indices = []
        for match in matches:
            p = match['name'].upper()
            if 'HAMMER' in p or 'ENGULFING' in p or 'MORNINGSTAR' in p:
                if match['strength'] > 0:
                    bullish_indices.append(match['index'])
        if rsi_val < 30: bullish_indices.append(len(df)-1)
        if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] and df['MA20'].iloc[-2] < df['MA50'].iloc[-2]:
            bullish_indices.append(len(df)-1)

        df_plot = df.tail(250 if interval=='1d' else 52 if interval=='1wk' else 24)
        os.makedirs('charts', exist_ok=True)
        chart_path = f"charts/{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        fig = plt.figure(figsize=(12,8))
        gs = GridSpec(4,1, height_ratios=[3,1,1,1])
        ax1 = fig.add_subplot(gs[0])
        ohlc = df_plot[['Date_Num','Open','High','Low','Close']].values
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup=PALETTE[0], colordown=PALETTE[1])
        ax1.plot(df_plot['Date_Num'], df_plot['MA20'], color=PALETTE[0], lw=2)
        ax1.plot(df_plot['Date_Num'], df_plot['MA50'], color=PALETTE[1], lw=1.5, linestyle='--')
        ax1.plot(df_plot['Date_Num'], df_plot['MA100'], color=PALETTE[2], lw=1.5, linestyle=':')
        ax1.set_title(f"{ticker} | {freq_str.capitalize()}")
        detect_valid_channels(df_plot, ax1)

        if detected_pattern != 'None':
            raw_idx = matches[-1]['index'] if matches else len(df)-1
            plot_start = len(df) - len(df_plot)
            if raw_idx >= plot_start:
                idx_plot = raw_idx - plot_start
                x0 = df_plot['Date_Num'].iloc[idx_plot]
                y0 = df_plot['High'].iloc[idx_plot]
                ax1.annotate(detected_pattern, xy=(x0, y0), xytext=(x0, y0*1.05),
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black'),
                             arrowprops=dict(arrowstyle='->'))
                draw_pattern_visual(ax1, df_plot, idx_plot, detected_pattern)

        for idx in bullish_indices:
            if idx >= len(df)-len(df_plot):
                x0 = df_plot['Date_Num'].iloc[idx-(len(df)-len(df_plot))]
                y0 = df_plot['Low'].iloc[idx-(len(df)-len(df_plot))] * 0.99
                ax1.scatter(x0, y0, color='green', marker='^', s=80)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(chart_path, dpi=100)
        plt.close(fig)
        return classification, chart_path, rsi_val, macd_val, sig_val, detected_pattern

    wb = openpyxl.load_workbook("stocks.xlsx")
    ws = wb["Current"]
    if "History" not in wb.sheetnames:
        ws_history = wb.create_sheet("History")
    else:
        ws_history = wb["History"]
    today = datetime.now().strftime("%Y-%m-%d")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"Stock Pattern Summary Report",0,1,"C")
    pdf.ln(5)

    for row in ws.iter_rows(min_row=2, max_col=3):
        tkr = row[0].value
        per = row[1].value or "6mo"
        freq = row[2].value or "daily"
        if not tkr: continue
        cl, path, r, m, s, patt = analyze_stock(tkr, per, freq)
        ws.cell(row=row[0].row, column=4).value = cl
        ws.cell(row=row[0].row, column=5).value = path
        ws.cell(row=row[0].row, column=6).value = patt
        ws_history.append([today, tkr, per, freq, cl, round(r,2), round(m,2), round(s,2), patt])
        pdf.add_page()
        pdf.image(path,10,30,190)

    wb.save("stocks.xlsx")
    pdf.output("stock_report.pdf")
    st.success("✅ Done! Download your PDF below:")
    with open("stock_report.pdf", "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="stock_report.pdf")
