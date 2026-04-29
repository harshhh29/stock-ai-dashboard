import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
nse_df = pd.read_csv(nse_url)

nse_tickers = [symbol + ".NS" for symbol in nse_df['SYMBOL'].tolist()]

st.set_page_config(page_title="Stock Intelligence", layout="wide")

st.title("🧠 Stock Intelligence Dashboard")

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Controls")

stock = st.sidebar.text_input(
    "Enter Stock Symbol (e.g. AAPL, TSLA, RELIANCE.NS, BTC-USD)",
    "AAPL"
)

compare_input = st.sidebar.text_input(
    "Compare Stocks (comma separated)",
    "AAPL,TSLA"
)

compare_stocks = [s.strip().upper() for s in compare_input.split(",")]

investment = st.sidebar.number_input("Investment Amount ($)", value=1000)

st.sidebar.markdown("💡 Examples: AAPL, TSLA, RELIANCE.NS, BTC-USD")

# ===================== DATA =====================
@st.cache_data
def load_data(symbol):
    return yf.download(symbol, start="2020-01-01")

data = load_data(stock)

if data.empty:
    st.error("❌ Invalid stock symbol")
    st.stop()

# ===================== INDICATORS =====================
data['MA50'] = data['Close'].rolling(50).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data = data.dropna()

# ===================== ML MODEL =====================
X = np.array(range(len(data))).reshape(-1,1)
y = data['Close'].values

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_score = model.score(X_test, y_test)

# Future prediction
last_x = X[-1][0]
future_X = np.array(range(last_x+1, last_x+6)).reshape(-1,1)
predictions = model.predict(future_X)

# ===================== SEQUENCE MODEL =====================
st.subheader("🧠 Sequence-Based Prediction")

prices = data['Close'].values

# Normalize manually
normalized = (prices - prices.min()) / (prices.max() - prices.min())

window = 10
weights = np.linspace(0.1, 1, window)

last_window = normalized[-window:]
last_window = last_window.flatten()

prediction_norm = np.dot(last_window, weights) / weights.sum()

pred_price = prediction_norm * (prices.max() - prices.min()) + prices.min()

st.write(f"Next Predicted Price (Sequence Model): ${pred_price:.2f}")

# ===================== PRICE CHART =====================
st.subheader(f"📊 {stock.upper()} Price Chart")

fig, ax = plt.subplots()
ax.plot(data['Close'], label="Price")
ax.plot(data['MA50'], label="MA50")
ax.legend()
st.pyplot(fig)

# ===================== RSI =====================
st.subheader("📉 RSI Indicator")

fig2, ax2 = plt.subplots()
ax2.plot(data['RSI'], color='orange')
ax2.axhline(70, linestyle='--')
ax2.axhline(30, linestyle='--')
st.pyplot(fig2)

# ===================== LINEAR PREDICTIONS =====================
st.subheader("🔮 Linear Regression Predictions")

for i, price in enumerate(predictions):
    st.write(f"Day {i+1}: ${price.item():.2f}")

# ===================== PERFORMANCE =====================
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2_score:.4f}")
col2.metric("RMSE", f"{rmse:.2f}")

# ===================== VALIDATION =====================
st.subheader("📉 Prediction vs Actual")

fig4, ax4 = plt.subplots()
ax4.plot(y_test, label="Actual")
ax4.plot(y_pred, label="Predicted")
ax4.legend()
st.pyplot(fig4)

# ===================== TREND =====================
st.subheader("📈 30-Day Trend")

close_prices = data['Close'].values.flatten()

if len(close_prices) < 30:
    st.warning("Not enough data for trend calculation")
else:
    change_pct = (close_prices[-1] - close_prices[-30]) / close_prices[-30] * 100

    if change_pct > 0:
        st.success(f"Uptrend 📈 (+{change_pct:.2f}%)")
    else:
        st.error(f"Downtrend 📉 ({change_pct:.2f}%)")

# ===================== PORTFOLIO =====================
st.subheader("💼 Portfolio Simulation")

close_prices = data['Close'].values.flatten()

if len(close_prices) < 30:
    st.warning("Not enough data for portfolio simulation")
else:
    buy_price = float(close_prices[-30])
    current_price = float(close_prices[-1])

    shares = investment / buy_price
    current_value = shares * current_price
    profit = current_value - investment

    st.write(f"Buy Price: ${buy_price:.2f}")
    st.write(f"Current Price: ${current_price:.2f}")
    st.write(f"Current Value: ${current_value:.2f}")

    if profit > 0:
        st.success(f"Profit: +${profit:.2f}")
    else:
        st.error(f"Loss: ${profit:.2f}")

# ===================== COMPARISON =====================
st.subheader("⚖️ Stock Comparison")

compare_df = pd.DataFrame()

for s in compare_stocks:
    temp = load_data(s)['Close']
    compare_df[s] = temp

compare_df = compare_df.dropna()
compare_df = compare_df / compare_df.iloc[0]

fig3, ax3 = plt.subplots()
for col in compare_df.columns:
    ax3.plot(compare_df[col], label=col)

ax3.legend()
st.pyplot(fig3)

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Machine Learning-based stock analysis system with regression, sequence modeling, and portfolio simulation")