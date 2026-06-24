import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
 
warnings.filterwarnings("ignore")
 
st.set_page_config(page_title="Stock Price Forecast (ARIMA)", layout="wide")
 
st.title("📈 Stock Price Forecast using ARIMA")
st.caption("Pulls 5 years of historical data from Yahoo Finance, plots it, "
           "and forecasts the price up to a target date using ARIMA.")
 
# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
 
ticker = st.sidebar.text_input(
    "Ticker symbol (Yahoo Finance format)",
    value="AAPL",
    help="Examples: AAPL, TSLA, MSFT, RELIANCE.NS, ^NSEI, BTC-USD",
)
 
years_back = st.sidebar.slider("Years of historical data", 1, 10, 5)
 
target_date = st.sidebar.date_input(
    "Forecast target date",
    value=datetime(2027, 6, 30),
    min_value=datetime.today() + timedelta(days=1),
)
 
p = st.sidebar.number_input("ARIMA p (AR order)", min_value=0, max_value=10, value=5)
d = st.sidebar.number_input("ARIMA d (differencing)", min_value=0, max_value=3, value=1)
q = st.sidebar.number_input("ARIMA q (MA order)", min_value=0, max_value=10, value=0)
 
run_button = st.sidebar.button("Run Forecast", type="primary")
 
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ ARIMA is a purely statistical model. It extrapolates from past price "
    "patterns only — it has no knowledge of news, earnings, or future events. "
    "Long-horizon forecasts (months/years ahead) carry very wide uncertainty. "
    "This tool is for educational purposes, **not** investment advice."
)
 
 
@st.cache_data(ttl=3600)
def load_data(tkr: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
    return df
 
 
def run_arima_forecast(close_series: pd.Series, order: tuple, target: datetime):
    """Fit ARIMA on closing price series and forecast to target date."""
    last_date = close_series.index[-1]
    # business days between last available date and target date
    horizon = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=target)
    steps = len(horizon)
    if steps < 1:
        steps = 1
        horizon = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)
 
    model = ARIMA(close_series, order=order)
    fitted = model.fit()
    forecast_result = fitted.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
 
    forecast_df = pd.DataFrame({
        "Forecast": forecast_mean.values,
        "Lower_95": conf_int.iloc[:, 0].values,
        "Upper_95": conf_int.iloc[:, 1].values,
    }, index=horizon)
 
    return fitted, forecast_df
 
 
if run_button:
    if not ticker.strip():
        st.error("Please enter a valid ticker symbol.")
        st.stop()
 
    with st.spinner(f"Downloading {ticker.upper()} data from Yahoo Finance..."):
        data = load_data(ticker.upper(), years_back)
 
    if data.empty:
        st.error(
            f"No data found for ticker '{ticker}'. Check the symbol "
            f"(e.g. use 'RELIANCE.NS' for NSE stocks, 'BTC-USD' for crypto)."
        )
        st.stop()
 
    # Handle possible MultiIndex columns (yfinance sometimes returns these)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
 
    close = data["Close"].dropna()
    close.index = pd.to_datetime(close.index)
 
    st.subheader(f"{ticker.upper()} — Historical Closing Price ({years_back}Y)")
 
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(close.index, close.values, label="Historical Close", color="#1f77b4")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker.upper()} Closing Price — Last {years_back} Years")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
 
    st.markdown(f"**Latest available close:** {close.iloc[-1]:.2f} on {close.index[-1].date()}")
 
    with st.spinner("Fitting ARIMA model and forecasting..."):
        try:
            fitted_model, forecast_df = run_arima_forecast(
                close, (int(p), int(d), int(q)), pd.Timestamp(target_date)
            )
        except Exception as e:
            st.error(f"ARIMA fitting failed: {e}")
            st.stop()
 
    st.subheader(f"Forecast through {target_date}")
 
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(close.index, close.values, label="Historical Close", color="#1f77b4")
    ax2.plot(forecast_df.index, forecast_df["Forecast"], label="ARIMA Forecast",
              color="#ff7f0e", linestyle="--")
    ax2.fill_between(
        forecast_df.index,
        forecast_df["Lower_95"],
        forecast_df["Upper_95"],
        color="#ff7f0e",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.set_title(f"{ticker.upper()} — Historical + ARIMA{(p, d, q)} Forecast")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
 
    final_row = forecast_df.iloc[-1]
    st.success(
        f"**Forecasted price on {forecast_df.index[-1].date()}:** "
        f"{final_row['Forecast']:.2f}  "
        f"(95% CI: {final_row['Lower_95']:.2f} – {final_row['Upper_95']:.2f})"
    )
 
    with st.expander("Show forecast data table"):
        st.dataframe(forecast_df.style.format("{:.2f}"))
 
    with st.expander("Show ARIMA model summary"):
        st.text(str(fitted_model.summary()))
 
    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        "Download forecast as CSV",
        data=csv,
        file_name=f"{ticker.upper()}_arima_forecast.csv",
        mime="text/csv",
    )
 
else:
    st.info("Set your options in the sidebar and click **Run Forecast** to begin.")
  
