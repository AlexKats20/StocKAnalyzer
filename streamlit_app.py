import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas_ta as ta
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.linear_model import LinearRegression

# Custom Style & Palette
tplt = plt.get_cmap('tab10')
PALETTE = tplt.colors
ACCENT = '#ff6f00'
plt.style.use('default')

# Pattern Visual Highlight
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

# Channel Detection
def detect_valid_channels(df, ax1, lookback=50, stride=5, min_slope=0.005):
    if len(df) < lookback:
        return
    last_channel_end = -1
    up_plotted = False
    down_plotted = False
    for start in range(0, len(df)-lookback, stride):
        if start < last_channel_end:
            continue
        end = start + lookback
        x = np.arange(lookback).reshape(-1,1)
        y_high = df['High'].iloc[start:end].values.reshape(-1,1)
        y_low = df['Low'].iloc[start:end].values.reshape(-1,1)
        close_prices = df['Close'].iloc[start:end].values

        reg_high = LinearRegression().fit(x, y_high)
        reg_low = LinearRegression().fit(x, y_low)
        slope_high, slope_low = reg_high.coef_[0][0], reg_low.coef_[0][0]

        upper_line = reg_high.predict(x).flatten()
        lower_line = reg_low.predict(x).flatten()
        channel_width = upper_line - lower_line
        width_var = np.std(channel_width)
        within_channel = np.mean((close_prices >= lower_line) & (close_prices <= upper_line))

        is_up = slope_high > min_slope and slope_low > min_slope
        is_down = slope_high < -min_slope and slope_low < -min_slope

        if (is_up or is_down) and width_var < 4.0 and within_channel > 0.4:
            color = PALETTE[3]
            hatch = '///'
            label = None
            if is_up and not up_plotted:
                label = 'Upward Channel'
                up_plotted = True
            if is_down and not down_plotted:
                label = 'Downward Channel'
                down_plotted = True

            date_nums = df['Date_Num'].iloc[start:end]
            ax1.plot(date_nums, upper_line, '--', lw=1.2, color=color, label=label)
            ax1.plot(date_nums, lower_line, '--', lw=1.2, color=color)
            ax1.fill_between(date_nums, lower_line, upper_line, color=color, alpha=0.2, hatch=hatch)
            last_channel_end = end

# Stock Analysis
def analyze_stock(ticker, period, freq_str):
    freq_map = {'Daily': '1d', 'Weekly': '1wk', 'Monthly': '1mo'}
    interval = freq_map.get(freq_str, '1d')
    actual_period = 'max' if period.lower() == 'max' else period

    st.write(f"⏳ Downloading data for {ticker} ({actual_period}, {freq_str})")
    df = yf.Ticker(ticker).history(period=actual_period, interval=interval, auto_adjust=False, prepost=True)
    if df.empty:
        raise ValueError("No data returned for the specified ticker, period, or frequency.")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if len(df) < 14:  # Minimum for RSI (14) and partial MACD
        st.warning(f"Insufficient data ({len(df)} rows) for {ticker}. Try a longer period (e.g., 3mo or 6mo) or Daily/Weekly frequency.")
        raise ValueError("Insufficient data returned for the specified period and frequency.")
    
    df['Date_Num'] = mdates.date2num(df.index)

    # Indicators
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA100'] = df['Close'].rolling(100, min_periods=1).mean()

    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Calculate MACD with dynamic parameters based on data length
    min_periods = min(len(df), 26)  # Use available data, up to slow=26
    macd_df = ta.macd(df['Close'], fast=12, slow=min_periods, signal=9)
    if macd_df is None or macd_df.empty or len(df) < min_periods:
        st.warning(f"MACD calculation requires {min_periods} periods but got {len(df)}. Using default values.")
        df['MACD'] = 0
        df['Signal'] = 0
    else:
        df['MACD'] = macd_df['MACD_12_{}_9'.format(min_periods)]
        df['Signal'] = macd_df['MACDs_12_{}_9'.format(min_periods)]

    # Pattern detection using pandas_ta
    matches = []
    candles = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name='all')
    for col in candles.columns:
        idxs = np.where(candles[col] != 0)[0]
        for idx in idxs:
            if (df.index[-1] - df.index[idx]).days <= 10:
                matches.append({
                    'name': col.upper(),
                    'index': idx,
                    'date': df.index[idx].date(),
                    'strength': int(candles[col].iloc[idx])
                })

    # Pattern fallback
    if matches:
        detected_pattern = matches[-1]['name']
        strength = f"{detected_pattern}={matches[-1]['strength']}"
    else:
        window = min(60, len(df))
        xw = np.arange(window).reshape(-1, 1)
        yh = df['High'].iloc[-window:].values.reshape(-1, 1)
        yl = df['Low'].iloc[-window:].values.reshape(-1, 1)
        if window < 2:
            detected_pattern = 'None'; strength = ''
        else:
            rh = LinearRegression().fit(xw, yh)
            rl = LinearRegression().fit(xw, yl)
            sh, sl = rh.coef_[0][0], rl.coef_[0][0]
            if sh > 0.01 and sl > 0.01:
                detected_pattern = 'UpwardChannel'
            elif sh < -0.01 and sl < -0.01:
                detected_pattern = 'DownwardChannel'
            else:
                detected_pattern = 'None'
            strength = ''

    # Final classification
    rsi_val = df['RSI'].iloc[-1] if not df['RSI'].isna().iloc[-1] else 50
    macd_val = df['MACD'].iloc[-1] if not df['MACD'].isna().iloc[-1] else 0
    sig_val = df['Signal'].iloc[-1] if not df['Signal'].isna().iloc[-1] else 0
    classification = (
        'Bullish' if rsi_val < 35 or (macd_val > sig_val and macd_val > -0.5)
        else 'Bearish' if rsi_val > 65 or (macd_val < sig_val and macd_val < 0.5)
        else 'Neutral'
    )

    # Bullish markers
    bullish_indices = []
    for match in matches:
        pattern = match['name'].upper()
        idx = match['index']
        if 'HAMMER' in pattern or 'ENGULFING' in pattern or 'MORNINGSTAR' in pattern:
            if match['strength'] > 0:
                bullish_indices.append(idx)
    if rsi_val < 30:
        bullish_indices.append(len(df) - 1)
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] and df['MA20'].iloc[-2] < df['MA50'].iloc[-2]:
        bullish_indices.append(len(df) - 1)

    # Plot window
    df_plot = df.tail(250 if interval == '1d' else 52 if interval == '1wk' else 24) if actual_period != 'max' else df

    # Main Chart
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ohlc = df_plot[['Date_Num', 'Open', 'High', 'Low', 'Close']].values
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup=PALETTE[0], colordown=PALETTE[1])

    ax1.plot(df_plot['Date_Num'], df_plot['MA20'], color=PALETTE[0], lw=2, label='MA20')
    ax1.plot(df_plot['Date_Num'], df_plot['MA50'], color=PALETTE[1], lw=1.5, linestyle='--', label='MA50')
    ax1.plot(df_plot['Date_Num'], df_plot['MA100'], color=PALETTE[2], lw=1.5, linestyle=':', label='MA100')
    ax1.set_title(f"{ticker} | {freq_str} Chart")
    ax1.set_ylabel('Price')

    detect_valid_channels(df_plot, ax1)

    if detected_pattern and detected_pattern != 'None':
        raw_idx = matches[-1]['index'] if matches else len(df) - 1
        plot_start = len(df) - len(df_plot)
        if raw_idx >= plot_start:
            idx_plot = raw_idx - plot_start
            x0 = df_plot['Date_Num'].iloc[idx_plot]
            y0 = df_plot['High'].iloc[idx_plot]
            ax1.annotate(detected_pattern, xy=(x0, y0), xytext=(x0, y0 * 1.05),
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black'),
                         arrowprops=dict(arrowstyle='->'))
            draw_pattern_visual(ax1, df_plot, idx_plot, detected_pattern)

    # Plot bullish markers
    plot_start = len(df) - len(df_plot)
    for idx in bullish_indices:
        if idx >= plot_start and idx < len(df):
            idx_plot = idx - plot_start
            x0 = df_plot['Date_Num'].iloc[idx_plot]
            y0 = df_plot['Low'].iloc[idx_plot] * 0.99
            ax1.scatter(x0, y0, color='green', marker='^', s=80, zorder=5)

    ax1.grid(True, linestyle=':', lw=0.5, alpha=0.4)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(df_plot['Date_Num'], df_plot['Volume'], color='gray')

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_plot['Date_Num'], df_plot['RSI'], color=PALETTE[4], label='RSI')
    ax3.axhline(70, color='red', linestyle='--')
    ax3.axhline(30, color='green', linestyle='--')
    ax3.legend(loc='upper left')

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df_plot['Date_Num'], df_plot['MACD'], color=PALETTE[5], label='MACD')
    ax4.plot(df_plot['Date_Num'], df['Signal'], color=PALETTE[6], label='Signal')
    ax4.legend(loc='upper left')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate()
    plt.tight_layout()

    # Pattern Occurrence Chart
    df_occ = df.tail(250 if interval == '1d' else 52 if interval == '1wk' else 24) if actual_period != 'max' else df
    candles = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name=detected_pattern.lower())
    occ_idx = np.where(candles.iloc[:, 0] != 0)[0] if not candles.empty else []
    occ_idx_plot = [i for i in occ_idx if i >= len(df) - len(df_occ)]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df_occ.index, df_occ['Close'], label='Close')
    if occ_idx_plot:
        occ_dates = df_occ.index[[i - (len(df) - len(df_occ)) for i in occ_idx_plot]]
        occ_prices = df_occ['Close'].iloc[[i - (len(df) - len(df_occ)) for i in occ_idx_plot]]
        ax2.scatter(occ_dates, occ_prices, color=ACCENT, label=detected_pattern)
    ax2.set_title(f"{ticker} Occurrences of {detected_pattern}")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.legend()
    fig2.autofmt_xdate()

    return classification, fig, fig2, rsi_val, macd_val, sig_val, detected_pattern, strength, len(occ_idx)

# Streamlit App
st.title("Stock Pattern Analysis")
st.write("Enter a stock ticker and select analysis parameters to view technical analysis results.")
st.info("For short periods (e.g., 1mo) with Monthly frequency, select a longer period (e.g., 3mo or 6mo) or Daily/Weekly frequency to ensure sufficient data. This app uses free yfinance data.")

ticker = st.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
freq = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)

if st.button("Analyze"):
    try:
        classification, fig, fig2, rsi, macd, signal, pattern, strength, occ_count = analyze_stock(ticker, period, freq)
        if fig and fig2:
            st.subheader("Analysis Results")
            st.write(f"**Classification**: {classification}")
            st.write(f"**RSI**: {rsi:.2f}")
            st.write(f"**MACD**: {macd:.2f}")
            st.write(f"**Signal**: {signal:.2f}")
            st.write(f"**Detected Pattern**: {pattern} {strength}")
            st.subheader("Technical Chart")
            st.pyplot(fig)
            st.subheader(f"Pattern Occurrences: {occ_count}")
            st.pyplot(fig2)
            plt.close(fig)
            plt.close(fig2)
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
