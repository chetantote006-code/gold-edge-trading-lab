import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="GOLD EDGE TRADING LAB",
    page_icon="📈",
    layout="wide"
)

st.title("📈 GOLD EDGE TRADING LAB")
st.caption("Gold Strategy Backtesting Dashboard")

st.warning(
    "Research only. Results are simulations, not guaranteed profits. "
    "Historical futures prices may differ from your broker's XAUUSD prices."
)

st.sidebar.header("⚙️ Strategy Settings")

risk = st.sidebar.number_input(
    "Risk per trade ($)", min_value=1.0, value=25.0, step=5.0
)

rr = st.sidebar.selectbox(
    "Risk : Reward", [2, 3, 4], index=1
)

avg_period = st.sidebar.number_input(
    "Average range candles", min_value=5, value=20
)

multiplier = st.sidebar.number_input(
    "Signal range multiplier", min_value=1.0, value=1.5, step=0.1
)

entry_wait = st.sidebar.number_input(
    "Entry valid candles", min_value=1, value=20
)

max_hold = st.sidebar.number_input(
    "Maximum holding candles", min_value=1, value=100
)

buffer_pct = st.sidebar.number_input(
    "Stop buffer (%)", min_value=0.0, value=1.0, step=0.5
)

cost = st.sidebar.number_input(
    "Cost per trade ($)", min_value=0.0, value=0.0, step=0.5
)

uploaded_file = st.file_uploader(
    "📂 Upload your gold_data.csv",
    type=["csv"]
)


def load_data(file):
    raw = pd.read_csv(file)

    # Normalize column names
    columns = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in raw.columns
    }

    def find_column(*names):
        for name in names:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in columns:
                return columns[key]
        return None

    open_col = find_column("Open", "O")
    high_col = find_column("High", "H")
    low_col = find_column("Low", "L")
    close_col = find_column("Close", "C", "Adj Close")
    time_col = find_column("Datetime", "Timestamp", "Date", "Time")

    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError(
            "CSV mein Open, High, Low aur Close columns hone chahiye."
        )

    data = pd.DataFrame({
        "Open": pd.to_numeric(raw[open_col], errors="coerce"),
        "High": pd.to_numeric(raw[high_col], errors="coerce"),
        "Low": pd.to_numeric(raw[low_col], errors="coerce"),
        "Close": pd.to_numeric(raw[close_col], errors="coerce"),
    })

    if time_col:
        data["Time"] = raw[time_col].astype(str)
    else:
        data["Time"] = np.arange(len(raw))

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    ).reset_index(drop=True)

    valid = (
        (data["High"] >= data["Low"]) &
        (data["High"] >= data["Open"]) &
        (data["High"] >= data["Close"]) &
        (data["Low"] <= data["Open"]) &
        (data["Low"] <= data["Close"])
    )

    data = data.loc[valid].reset_index(drop=True)

    if len(data) < avg_period + 5:
        raise ValueError("Backtest ke liye candles bahut kam hain.")

    return data


def backtest(data):
    data = data.copy()
    data["Range"] = data["High"] - data["Low"]

    data["AverageRange"] = (
        data["Range"].rolling(avg_period).mean().shift(1)
    )

    trades = []
    i = avg_period

    while i < len(data) - 1:
        candle = data.iloc[i]
        average = candle["AverageRange"]
        candle_range = candle["Range"]

        if (
            pd.isna(average)
            or average <= 0
            or candle_range <= multiplier * average
        ):
            i += 1
            continue

        signal_high = float(candle["High"])
        signal_low = float(candle["Low"])

        buffer = candle_range * buffer_pct / 100
        entry_index = None
        side = None
        entry = None

        last_entry_bar = min(
            len(data) - 1,
            i + int(entry_wait)
        )

        for j in range(i + 1, last_entry_bar + 1):
            bar = data.iloc[j]

            broke_up = bar["High"] >= signal_high
            broke_down = bar["Low"] <= signal_low

            # OHLC data cannot tell which side broke first
            if broke_up and broke_down:
                entry_index = j
                side = "Ambiguous"
                break

            if broke_up:
                side = "Long"
                entry = signal_high
                entry_index = j
                break

            if broke_down:
                side = "Short"
                entry = signal_low
                entry_index = j
                break

        if entry_index is None:
            i += 1
            continue

        if side == "Ambiguous":
            i = entry_index + 1
            continue

        if side == "Long":
            sl = signal_low - buffer
            tp = entry + rr * (entry - sl)
        else:
            sl = signal_high + buffer
            tp = entry - rr * (sl - entry)

        stop_distance = abs(entry - sl)

        if stop_distance <= 0:
            i = entry_index + 1
            continue

        last_exit_index = min(
            len(data) - 1,
            entry_index + int(max_hold)
        )

        exit_price = float(data.iloc[last_exit_index]["Close"])
        exit_index = last_exit_index
        reason = "Time exit"

        for k in range(entry_index, last_exit_index + 1):
            bar = data.iloc[k]

            if side == "Long":
                stop_hit = bar["Low"] <= sl
                target_hit = bar["High"] >= tp
            else:
                stop_hit = bar["High"] >= sl
                target_hit = bar["Low"] <= tp

            # Conservative assumption if both levels touch
            if stop_hit:
                exit_price = sl
                exit_index = k
                reason = "SL"
                break

            if target_hit:
                exit_price = tp
                exit_index = k
                reason = "TP"
                break

        if side == "Long":
            price_change = exit_price - entry
        else:
            price_change = entry - exit_price

        r_multiple = price_change / stop_distance
        gross_pnl = r_multiple * risk
        net_pnl = gross_pnl - cost

        trades.append({
            "Signal Time": candle["Time"],
            "Entry Time": data.iloc[entry_index]["Time"],
            "Exit Time": data.iloc[exit_index]["Time"],
            "Side": side,
            "Entry": entry,
            "Stop Loss": sl,
            "Take Profit": tp,
            "Exit Price": exit_price,
            "Exit Reason": reason,
            "R Multiple": r_multiple,
            "Gross P/L ($)": gross_pnl,
            "Cost ($)": cost,
            "Net P/L ($)": net_pnl,
            "Bars Held": exit_index - entry_index,
        })

        i = exit_index + 1

    return pd.DataFrame(trades)


if uploaded_file is None:
    st.info(
        "Sidebar mein settings select karo aur upar "
        "gold_data.csv upload karo."
    )
    st.stop()

try:
    candles = load_data(uploaded_file)
except Exception as error:
    st.error(str(error))
    st.stop()

st.subheader("📊 Historical Market Data")

a, b, c = st.columns(3)
a.metric("Valid candles", f"{len(candles):,}")
b.metric("First candle", str(candles["Time"].iloc[0]))
c.metric("Last candle", str(candles["Time"].iloc[-1]))

with st.expander("Preview candles"):
    st.dataframe(candles.head(20), use_container_width=True)

if st.button("🚀 RUN BACKTEST", type="primary"):
    with st.spinner("Testing your strategy..."):
        st.session_state["trades"] = backtest(candles)
        st.session_state["settings_used"] = {
            "risk": risk,
            "rr": rr,
            "average": avg_period,
            "multiplier": multiplier,
            "entry_wait": entry_wait,
            "max_hold": max_hold,
            "buffer_pct": buffer_pct,
            "cost": cost,
        }

trades = st.session_state.get("trades")

if trades is None:
    st.info("Click RUN BACKTEST to calculate results.")
    st.stop()

st.divider()
st.subheader("📈 Backtest Performance")

if trades.empty:
    st.warning(
        "Is strategy ke liye koi trade nahi mila. "
        "Signal multiplier ya entry wait badal kar test karo."
    )
    st.stop()

pnl = trades["Net P/L ($)"]
wins = pnl > 0

gross_profit = pnl[pnl > 0].sum()
gross_loss = abs(pnl[pnl < 0].sum())

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else float("inf")
)

equity = pnl.cumsum()
peak = equity.cummax().clip(lower=0)
drawdown = equity - peak
max_drawdown = abs(drawdown.min())

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Total trades", len(trades))
m2.metric("Net P/L", f"${pnl.sum():,.2f}")
m3.metric("Win rate", f"{wins.mean() * 100:.2f}%")
m4.metric(
    "Profit factor",
    "∞" if np.isinf(profit_factor) else f"{profit_factor:.2f}"
)
m5.metric("Max drawdown", f"${max_drawdown:,.2f}")

left, right = st.columns(2)

with left:
    st.markdown("#### Equity Curve")
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(equity) + 1), equity)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Net P/L ($)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("#### Trade Outcomes")
    counts = trades["Exit Reason"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of trades")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("📋 Trade History")
st.dataframe(trades, use_container_width=True, height=400)

st.download_button(
    "📥 Download Trade History CSV",
    data=trades.to_csv(index=False).encode("utf-8"),
    file_name="gold_edge_trade_history.csv",
    mime="text/csv"
)

equity_data = pd.DataFrame({
    "Trade": np.arange(1, len(equity) + 1),
    "Net P/L ($)": pnl.values,
    "Equity ($)": equity.values,
})

st.download_button(
    "📥 Download Equity Curve CSV",
    data=equity_data.to_csv(index=False).encode("utf-8"),
    file_name="gold_edge_equity_curve.csv",
    mime="text/csv"
)

with st.expander("⚠️ Methodology & limitations"):
    st.markdown("""
    - Signal candle range is compared with the average range of prior candles.
    - Entries use signal-candle high/low breakouts.
    - Stop loss is beyond the opposite edge of the signal candle.
    - Take profit uses the selected risk-to-reward multiple.
    - Only one trade is held at a time.
    - If SL and TP touch in the same OHLC candle, SL is assumed first.
    - Risk is normalized using R-multiples; this does not calculate broker-specific lot sizes.
    - Costs are a flat deduction per trade, not a precise spread/slippage model.
    - Yahoo GC=F futures prices can differ from your broker's XAUUSD CFD.
    - This tool does not connect to MT5, place orders, or provide live signals.
    - Historical results do not guarantee future performance.
    """)