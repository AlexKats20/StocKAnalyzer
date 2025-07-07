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

# === Styles ===
tplt = plt.get_cmap('tab10')
PALETTE = tplt.colors
ACCENT = '#ff6f00'
plt.style.use('default')

# === File uploader ===
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

    def detect_valid_channels(df, ax1, lookback=50, stride=5, min_slope=0.005):
        # Your channel logic (keep as is)
        ...

    def analyze_stock(ticker, period, freq_str):
        # Your full GridSpec version, with pandas-ta not talib
        # Use RSI, MACD, MAs, channel detection, bullish markers
        # Return chart_path, classification, pattern name if any
        ...

    for row in ws_current.iter_rows(min_row=2, max_col=3):
        tkr = row[0].value; per = row[1].value or '6mo'; freq = row[2].value or 'daily'
        if not tkr: continue
        try:
            cl, path, r, m, s, patt, strg = analyze_stock(tkr, per, freq)
            ws_current.cell(row=row[0].row, column=4).value = cl
            ws_current.cell(row=row[0].row, column=5).value = path
            ws_current.cell(row=row[0].row, column=6).value = patt
            ws_current.cell(row=row[0].row, column=7).value = strg
            if path and os.path.exists(path):
                img = XLImage(path); img.width=180; img.height=120
                ws_current.add_image(img, f'H{row[0].row}')
            ws_history.append([today,tkr,per,freq,cl,round(r,2),round(m,2),round(s,2),patt])
        except Exception as e:
            ws_current.cell(row=row[0].row, column=4).value = f'Error: {e}'
            print(f'⚠️ {tkr} error: {e}')

    wb.save('stocks.xlsx')

    # === Build PDF ===
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
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
    for vals in ws_current.iter_rows(min_row=2, max_col=7, values_only=True):
        tkr, period, freq, cl, chart_path, patt, strength = vals
        if tkr:
            pdf.cell(30,10,str(tkr),1)
            pdf.cell(25,10,str(period),1)
            pdf.cell(25,10,str(freq),1)
            pdf.cell(30,10,str(cl),1)
            pdf.cell(30,10,str(patt),1)
            pdf.ln()

    pdf.output('stock_report.pdf')

    # ✅ Show download link in Streamlit
    st.success("✅ Analysis done!")
    with open("stock_report.pdf", "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="stock_report.pdf")
