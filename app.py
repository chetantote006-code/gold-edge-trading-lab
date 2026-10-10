import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from supabase import create_client

st.set_page_config(
    page_title="GOLD EDGE TRADING LAB",
    page_icon="📈",
    layout="wide"
)

# ---------------- SUPABASE CONNECTION ----------------

@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = get_supabase()
except Exception:
    st.error(
        "Supabase connection configure nahi hai. "
        "Streamlit Secrets mein SUPABASE_URL aur SUPABASE_KEY check karo."
    )
    st.stop()


# ---------------- AUTHENTICATION ----------------

st.title("📈 GOLD EDGE TRADING LAB")
st.caption("Gold Strategy Backtesting Dashboard")

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.subheader("🔐 Login / Create Account")

    tab_login, tab_signup = st.tabs(["Login", "Create Account"])

    with tab_login:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", type="primary"):
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": login_email.strip(),
                    "password": login_password
                })

                st.session_state.user = response.user
                st.session_state.access_token = response.session.access_token
                st.session_state.refresh_token = response.session.refresh_token

                st.rerun()

            except Exception:
                st.error(
                    "Login nahi hua. Email/password check karo. "
                    "Agar email confirmation enabled hai, pehle email verify karo."
                )

    with tab_signup:
        signup_email = st.text_input("Email address", key="signup_email")
        signup_password = st.text_input(
            "Password (minimum 8 characters)",
            type="password",
            key="signup_password"
        )

        if st.button("Create Account"):
            if len(signup_password) < 8:
                st.error("Password kam se kam 8 characters ka hona chahiye.")
            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": signup_email.strip(),
                        "password": signup_password
                    })

                    if response.session:
                        st.session_state.user = response.user
                        st.session_state.access_token = (
                            response.session.access_token
                        )
                        st.session_state.refresh_token = (
                            response.session.refresh_token
                        )
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.success(
                            "Account request received. Email confirmation "
                            "agar enabled hai to inbox check karke verify karo, "
                            "phir Login karo."
                        )

                except Exception:
                    st.error(
                        "Account create nahi hua. Email check karo; "
                        "shayad account pehle se registered hai."
                    )

    st.info("Login karne ke baad tum apna private backtest data use kar sakte ho.")
    st.stop()


# Use the signed-in user's access token so Supabase RLS policies
# can identify the current user. Never use a service-role key here.
try:
    supabase.postgrest.auth(st.session_state.access_token)
except Exception:
    pass

user_id = st.session_state.user.id


if st.sidebar.button("🚪 Logout"):
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    for key in [
        "user",
        "access_token",
        "refresh_token",
        "trades",
        "settings_used",
        "saved_backtest_id",
    ]:
        st.session_state.pop(key, None)

    st.rerun()


st.sidebar.success("Logged in")
st.sidebar.caption(st.session_state.user.email or "Your account")


st.warning(
    "Research only. Results are simulations, not guaranteed profits. "
    "Historical futures prices may differ from your broker's XAUUSD prices."
)


# ---------------- STRATEGY SETTINGS ----------------

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


# ---------------- CSV LOADING ----------------

uploaded_file = st.file_uploader(
    "📂 Upload your gold_data.csv",
    type=["csv"]
)


def load_data(file):
    raw = pd.read_csv(file)

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


# ---------------- BACKTEST STRATEGY ----------------

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


# ---------------- BACKTEST HISTORY HELPERS ----------------

def current_settings():
    return {
        "risk": float(risk),
        "rr": int(rr),
        "average": int(avg_period),
        "multiplier": float(multiplier),
        "entry_wait": int(entry_wait),
        "max_hold": int(max_hold),
        "buffer_pct": float(buffer_pct),
        "cost": float(cost),
    }


def save_backtest(dataset_name, trades, settings):
    pnl = trades["Net P/L ($)"]
    wins = pnl > 0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))

    metrics = {
        "total_trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "win_rate": float(wins.mean() * 100),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0 else None
        ),
        "max_drawdown": float(
            abs((pnl.cumsum() - pnl.cumsum().cummax().clip(lower=0)).min())
        ),
    }

    equity = pnl.cumsum().tolist()

    response = supabase.table("backtests").insert({
        "user_id": user_id,
        "dataset_name": dataset_name,
        "settings": settings,
        "metrics": metrics,
        "equity_curve": equity,
    }).execute()

    if not response.data:
        raise RuntimeError("Backtest database mein save nahi hua.")

    backtest_id = response.data[0]["id"]

    rows = []
    for number, (_, row) in enumerate(trades.iterrows(), start=1):
        rows.append({
            "user_id": user_id,
            "backtest_id": backtest_id,
            "trade_number": number,
            "trade_data": {
                key: (
                    value.item() if isinstance(value, np.generic)
                    else value
                )
                for key, value in row.to_dict().items()
            },
        })

    # Insert batches to avoid an oversized request.
    for start in range(0, len(rows), 100):
        supabase.table("backtest_trades").insert(
            rows[start:start + 100]
        ).execute()

    return backtest_id


def load_backtest_history():
    return (
        supabase.table("backtests")
        .select("id, created_at, dataset_name, settings, metrics")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
        .data
    )


# ---------------- USER'S PRIVATE SAVED HISTORY ----------------

st.divider()
st.subheader("🗂️ Your Saved Backtests")

try:
    history = load_backtest_history()

    if history:
        history_rows = []
        for item in history:
            metric = item.get("metrics") or {}
            history_rows.append({
                "ID": item["id"],
                "Date": item["created_at"],
                "Dataset": item["dataset_name"],
                "Trades": metric.get("total_trades", 0),
                "Net P/L ($)": metric.get("net_pnl", 0),
                "Win Rate (%)": metric.get("win_rate", 0),
            })

        st.dataframe(
            pd.DataFrame(history_rows),
            use_container_width=True
        )

        history_ids = [item["id"] for item in history]
        selected_id = st.selectbox(
            "Load a saved trade history",
            history_ids
        )

        if st.button("📖 Load Saved Trades"):
            saved = (
                supabase.table("backtest_trades")
                .select("trade_number, trade_data")
                .eq("user_id", user_id)
                .eq("backtest_id", selected_id)
                .order("trade_number")
                .execute()
                .data
            )

            if saved:
                st.session_state["trades"] = pd.DataFrame(
                    [item["trade_data"] for item in saved]
                )
                st.session_state["saved_backtest_id"] = selected_id
                st.success("Saved trades loaded.")
                st.rerun()
            else:
                st.info("Is saved backtest mein trades nahi hain.")
    else:
        st.info("Abhi koi saved backtest nahi hai.")

except Exception:
    st.error(
        "Saved history load nahi hui. Supabase tables, login session "
        "aur Row Level Security policies check karo."
    )


# ---------------- UPLOAD AND RUN ----------------

if uploaded_file is None:
    st.info("Upar apni gold_data.csv upload karo, phir backtest run karo.")
    st.stop()

try:
    candles = load_data(uploaded_file)
except Exception as error:
    st.error(str(error))
    st.stop()

dataset_name = uploaded_file.name

st.subheader("📊 Historical Market Data")

a, b, c = st.columns(3)
a.metric("Valid candles", f"{len(candles):,}")
b.metric("First candle", str(candles["Time"].iloc[0]))
c.metric("Last candle", str(candles["Time"].iloc[-1]))

with st.expander("Preview candles"):
    st.dataframe(candles.head(20), use_container_width=True)

if st.button("🚀 RUN BACKTEST", type="primary"):
    with st.spinner("Testing your strategy..."):
        result = backtest(candles)
        st.session_state["trades"] = result
        st.session_state["settings_used"] = current_settings()
        st.session_state.pop("saved_backtest_id", None)

trades = st.session_state.get("trades")

if trades is None:
    st.info("Click RUN BACKTEST to calculate results.")
    st.stop()

# Save only when explicitly requested.
if st.button("💾 SAVE THIS BACKTEST TO MY ACCOUNT"):
    if trades.empty:
        st.warning("No trades available to save.")
    else:
        try:
            backtest_id = save_backtest(
                dataset_name,
                trades,
                st.session_state.get("settings_used", current_settings())
            )
            st.session_state["saved_backtest_id"] = backtest_id
            st.success(f"Backtest saved! ID: {backtest_id}")
            st.rerun()
        except Exception:
            st.error(
                "Save failed. Check Supabase authentication, RLS policies, "
                "and table columns. Agar partial save hua ho to history check karo."
            )


# ---------------- PERFORMANCE ----------------

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
    gross_profit / gross_loss if gross_loss > 0 else float("inf")
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
    - Risk is normalized using R-multiples; broker-specific lot sizes are not calculated.
    - Costs are a flat deduction per trade, not a precise spread/slippage model.
    - Yahoo GC=F futures prices can differ from your broker's XAUUSD CFD.
    - This tool does not connect to MT5, place orders, or provide live signals.
    - Historical results do not guarantee future performance.
    """)
