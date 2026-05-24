import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Financial Risk & Performance Analysis", layout="wide")

# Sidebar Control Panel
with st.sidebar:
    st.header("⚙️ Control Panel")
    tickers = st.text_input("Enter ticker symbols (comma separated):", "AAPL,MSFT,GOOG").split(",")
    benchmark = st.text_input("Benchmark index (e.g. ^GSPC for S&P500):", "^GSPC")
    start_date = st.date_input("Start date", datetime.date(2020,1,1))
    end_date = st.date_input("End date", datetime.date.today())
    freq = st.selectbox("Frequency", ["1d","1wk","1mo"])
    price_type = st.selectbox("Price type", ["Adj Close","Close"])
    risk_free_rate = st.number_input("Risk-free rate", value=0.0457)

# ---------------- DATA ----------------
def load_data(ticker):
    return yf.download(ticker, start=start_date, end=end_date, interval=freq)

data_dict = {t.strip(): load_data(t.strip()) for t in tickers}
benchmark_data = load_data(benchmark)

# ---------------- TABS ----------------
tab_overview, tab_benchmark, tab_corr, tab_cov, tab_download = st.tabs(
    ["Overview","Benchmark","Correlation","Covariance","Download"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.subheader("📈 Price History & Indicators")
    for t in tickers:
        df = data_dict[t.strip()]
        if df.empty: 
            st.warning(f"No data for {t}")
            continue

        # Heikin Ashi
        ha_df = df.copy()
        ha_df['HA_Close'] = (df['Open']+df['High']+df['Low']+df['Close'])/4
        ha_df['HA_Open'] = (df['Open'].shift(1)+df['Close'].shift(1))/2
        ha_df['HA_High'] = df[['High','Open','Close']].max(axis=1)
        ha_df['HA_Low'] = df[['Low','Open','Close']].min(axis=1)

        fig = go.Figure(data=[go.Candlestick(
            x=ha_df.index,
            open=ha_df['HA_Open'],
            high=ha_df['HA_High'],
            low=ha_df['HA_Low'],
            close=ha_df['HA_Close'],
            name="Heikin Ashi"
        )])

        # EMA
        for ema in [7,30,50,200]:
            ha_df[f"EMA{ema}"] = ha_df['HA_Close'].ewm(span=ema).mean()
            fig.add_trace(go.Scatter(x=ha_df.index, y=ha_df[f"EMA{ema}"], mode="lines", name=f"EMA {ema}"))

        # Bollinger Bands
        ha_df['MA20'] = ha_df['HA_Close'].rolling(20).mean()
        ha_df['BB_up'] = ha_df['MA20'] + 2*ha_df['HA_Close'].rolling(20).std()
        ha_df['BB_down'] = ha_df['MA20'] - 2*ha_df['HA_Close'].rolling(20).std()
        fig.add_trace(go.Scatter(x=ha_df.index, y=ha_df['BB_up'], line=dict(color='gray'), name="BB Upper"))
        fig.add_trace(go.Scatter(x=ha_df.index, y=ha_df['BB_down'], line=dict(color='gray'), name="BB Lower"))

        st.plotly_chart(fig, use_container_width=True)

        # Forecast (ARIMA)
        try:
            model = ARIMA(ha_df['HA_Close'].dropna(), order=(5,1,0))
            fit = model.fit()
            forecast = fit.forecast(steps=60)  # ~3 months
            st.line_chart(forecast)
        except Exception as e:
            st.warning(f"Forecast model failed for {t}: {e}")

        # Buy/Sell/Hold Recommendation
        last_price = ha_df['HA_Close'].iloc[-1]
        ema200 = ha_df['EMA200'].iloc[-1]
        if last_price > ema200:
            st.success(f"Recommendation for {t}: BUY (Price above EMA200)")
        elif last_price < ema200*0.95:
            st.error(f"Recommendation for {t}: SELL (Price far below EMA200)")
        else:
            st.info(f"Recommendation for {t}: HOLD (Neutral zone)")

# ---------------- BENCHMARK ----------------
with tab_benchmark:
    st.subheader("📊 Benchmark Comparison")
    for t in tickers:
        df = data_dict[t.strip()]
        returns = df[price_type].pct_change().dropna()
        bench_returns = benchmark_data[price_type].pct_change().dropna()
        if not returns.empty and not bench_returns.empty:
            beta = np.cov(returns, bench_returns)[0,1]/np.var(bench_returns)
            st.write(f"{t} Beta vs {benchmark}: {beta:.2f}")

# ---------------- CORRELATION ----------------
with tab_corr:
    st.subheader("🔗 Correlation Heatmap")
    returns_df = pd.DataFrame({t.strip(): data_dict[t.strip()][price_type].pct_change() for t in tickers}).dropna()
    corr = returns_df.corr()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

# ---------------- COVARIANCE ----------------
with tab_cov:
    st.subheader("📐 Covariance Heatmap")
    cov = returns_df.cov()
    st.dataframe(cov.style.background_gradient(cmap="viridis"))

# ---------------- DOWNLOAD ----------------
with tab_download:
    st.subheader("📥 Download Results")
    output = pd.ExcelWriter("analysis_results.xlsx", engine="openpyxl")
    for t in tickers:
        data_dict[t.strip()].to_excel(output, sheet_name=t.strip())
    output.close()
    with open("analysis_results.xlsx","rb") as f:
        st.download_button("Download Excel", f, file_name="analysis_results.xlsx")
