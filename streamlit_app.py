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

# === Colors & Patterns ===
PALETTE = plt.get_cmap('tab10').colors
ACCENT = '#00c853'
candlestick_patterns = [
    "hammer", "hangingman", "engulfing", "morningstar", "eveningstar",
    "doji", "spinningtop", "shootingstar", "invertedhammer",
    "3blackcrows", "3whitesoldiers", "harami", "piercing", "darkcloudcover",
    "abandonedbaby", "kicking", "marubozu", "counterattack", "belt_hold"
    # Add more as needed (keep practical)
]

# === Detect channels ===
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
        within = np.mean((close_prices >= lower_line) & (close_prices <= upper_line))
        if (abs(slope_high) > min_slope and abs(slope_low) > min_slope) and width_var < 4.0 and within > 0.4:
            ax1.plot(df['Date_Num'].iloc[start:end], upper_line, '--', lw=1.2, color=PALETTE[3])
            ax1.plot(df['Date_Num'].iloc[start:end], lower_line, '--', lw=1.2, color=PALETTE[3])

# === Main stock analyzer ===
def analyze_stock(ticker, period, freq_str):
    freq_map = {'daily':'1d', 'weekly':'1wk', 'monthly':'1mo'}
    interval = freq_map.get(freq_str.lower(), '1d')
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['Date_Num'] = mdates.date2num(df.index.to_pydatetime())
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA100'] = df['Close'].rolling(100).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal'] = macd['MACDs_12_26_9']

    # === Find MOST RECENT pattern ===
    detected_pattern = 'None'
    latest_idx = -1
    for name in candlestick_patterns:
        try:
            res = df.ta.cdl_pattern(pattern=name)
            if isinstance(res, pd.DataFrame): res = res.iloc[:,0]
            idxs = np.where(res != 0)[0]
            if len(idxs) > 0 and idxs[-1] > latest_idx:
                detected_pattern = name.upper()
                latest_idx = idxs[-1]
        except: continue

    rsi_val, macd_val, sig_val = df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['Signal'].iloc[-1]
    classification = 'Bullish' if rsi_val<35 or (macd_val>sig_val and macd_val>-0.5) else 'Bearish' if rsi_val>65 or (macd_val<sig_val and macd_val<0.5) else 'Neutral'

    df_plot = df.tail(250 if interval=='1d' else 52 if interval=='1wk' else 24)
    fig = plt.figure(figsize=(12,8))
    gs = GridSpec(4,1,[3,1,1,1])
    ax1 = fig.add_subplot(gs[0])
    ohlc = df_plot[['Date_Num','Open','High','Low','Close']].values
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup=PALETTE[0], colordown=PALETTE[1])
    ax1.plot(df_plot['Date_Num'], df_plot['MA20'], color=PALETTE[0], lw=2)
    ax1.plot(df_plot['Date_Num'], df_plot['MA50'], color=PALETTE[1], lw=1.5, ls='--')
    ax1.plot(df_plot['Date_Num'], df_plot['MA100'], color=PALETTE[2], lw=1.5, ls=':')
    detect_valid_channels(df_plot, ax1)
    if detected_pattern != 'None' and latest_idx >= len(df)-len(df_plot):
        idx_plot = latest_idx - (len(df)-len(df_plot))
        x0, y0 = df_plot['Date_Num'].iloc[idx_plot], df_plot['High'].iloc[idx_plot]
        ax1.annotate(detected_pattern, xy=(x0,y0), xytext=(x0,y0*1.05),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black'),
            arrowprops=dict(arrowstyle='->'))
    ax1.set_title(f"{ticker} | {freq_str}")
    ax1.grid(True, linestyle=':', lw=0.5, alpha=0.4)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(df_plot['Date_Num'], df_plot['Volume'], color='gray')
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_plot['Date_Num'], df_plot['RSI'], color=PALETTE[4])
    ax3.axhline(70, color='red', ls='--'); ax3.axhline(30, color='green', ls='--')
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df_plot['Date_Num'], df_plot['MACD'], color=PALETTE[5])
    ax4.plot(df_plot['Date_Num'], df_plot['Signal'], color=PALETTE[6])
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(); plt.tight_layout()
    os.makedirs('charts', exist_ok=True)
    path = f"charts/{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(path, dpi=100); plt.close(fig)

    return classification, path, rsi_val, macd_val, sig_val, detected_pattern

# === Streamlit & PDF ===
st.title("📊 Pattern Analyzer")
file = st.file_uploader("Upload your stocks.xlsx", type="xlsx")
if file:
    with open("stocks.xlsx","wb") as f: f.write(file.read())
    wb = openpyxl.load_workbook("stocks.xlsx")
    ws = wb['Current']
    if 'History' not in wb.sheetnames:
        ws_h = wb.create_sheet('History')
        ws_h.append(['Date','Ticker','Period','Freq','Classification','RSI','MACD','Signal','Pattern'])
    else: ws_h = wb['History']
    today = datetime.now().strftime('%Y-%m-%d')
    charts = []
    for row in ws.iter_rows(min_row=2, max_col=3):
        tkr, per, freq = row[0].value, row[1].value or '6mo', row[2].value or 'daily'
        if not tkr: continue
        cl, path, r, m, s, patt = analyze_stock(tkr, per, freq)
        ws.cell(row=row[0].row, column=4).value = cl
        ws.cell(row=row[0].row, column=5).value = path
        ws.cell(row=row[0].row, column=6).value = patt
        charts.append((tkr, path, patt, per, freq))
        ws_h.append([today,tkr,per,freq,cl,round(r,2),round(m,2),round(s,2),patt])
    wb.save("stocks.xlsx")

    pdf = FPDF(); pdf.add_page()
    pdf.set_font('Arial','B',16); pdf.cell(0,10,'Stock Pattern Summary',0,1,'C')
    pdf.set_font('Arial',size=12); pdf.cell(30,10,'Date:'); pdf.cell(50,10,today,0,1); pdf.ln(5)
    pdf.set_font('Arial','B',11); pdf.cell(30,10,'Ticker',1); pdf.cell(25,10,'Period',1); pdf.cell(25,10,'Freq',1); pdf.cell(30,10,'Classif',1); pdf.cell(30,10,'Pattern',1); pdf.ln()
    pdf.set_font('Arial',size=10)
    for tkr, _, patt, per, freq in charts: pdf.cell(30,10,tkr,1); pdf.cell(25,10,per,1); pdf.cell(25,10,freq,1); pdf.cell(30,10,cl,1); pdf.cell(30,10,patt,1); pdf.ln()
    for tkr, path, patt, per, freq in charts:
        pdf.add_page(); pdf.cell(0,10,f"{tkr} | Pattern: {patt}",0,1,'C'); pdf.image(path,10,30,190)
        df = yf.Ticker(tkr).history(period=per, interval={'daily':'1d','weekly':'1wk','monthly':'1mo'}.get(freq,'1d'))
        df = df[['Open','High','Low','Close']].dropna()
        occ = df.ta.cdl_pattern(pattern=patt.lower()) if patt != 'None' else None
        occ_idx = []
        if occ is not None:
            if isinstance(occ, pd.DataFrame): occ = occ.iloc[:,0]
            occ_idx = np.where(occ != 0)[0]
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(df.index, df['Close'])
        if occ_idx: ax2.scatter(df.index[occ_idx], df['Close'].iloc[occ_idx], color=ACCENT)
        tmp = f"{tkr}_occ.png"; fig2.savefig(tmp, dpi=100); plt.close(fig2)
        pdf.add_page(); pdf.image(tmp,10,30,190); os.remove(tmp)
        counts, returns = {}, {}
        for name in candlestick_patterns:
            try:
                r = df.ta.cdl_pattern(pattern=name)
                if isinstance(r, pd.DataFrame): r = r.iloc[:,0]
                idxs = np.where(r != 0)[0]
                f1y,f2y = [],[]
                for i in idxs:
                    if i+252<len(df): f1y.append((df['Close'].iloc[i+252]-df['Close'].iloc[i])/df['Close'].iloc[i])
                    if i+504<len(df): f2y.append((df['Close'].iloc[i+504]-df['Close'].iloc[i])/df['Close'].iloc[i])
                if idxs.size>0:
                    counts[name.upper()]=len(idxs)
                    returns[name.upper()]={'1Y':np.mean(f1y)*100 if f1y else None, '2Y':np.mean(f2y)*100 if f2y else None}
            except: continue
        top = sorted(counts.items(), key=lambda x:x[1], reverse=True)[:20]
        pdf.add_page(); pdf.cell(0,10,f"{tkr} | Top 20 Patterns",0,1,'C'); pdf.ln(4)
        pdf.set_font('Arial','B',10); pdf.cell(60,8,'Pattern',1); pdf.cell(30,8,'Occur.',1); pdf.cell(30,8,'1Y Avg %',1); pdf.cell(30,8,'2Y Avg %',1); pdf.ln()
        pdf.set_font('Arial','',10)
        for pname,c in top:
            r1,r2=returns[pname]['1Y'],returns[pname]['2Y']
            pdf.cell(60,8,pname,1); pdf.cell(30,8,str(c),1)
            pdf.cell(30,8,f"{r1:.2f}%" if r1 else '-',1); pdf.cell(30,8,f"{r2:.2f}%" if r2 else '-',1); pdf.ln()
    pdf.output("stock_report.pdf")
    st.success("✅ PDF done! Download:")
    with open("stock_report.pdf","rb") as f:
        st.download_button("📄 Download PDF", f, file_name="stock_report.pdf")
