import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from agent import download_hourly, simulate_hourly, get_company_info_and_performance

TICKER_LIST = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms Inc.",
    "CBA.AX": "Commonwealth Bank of Australia",
    "BHP.AX": "BHP Group Ltd",
    "WES.AX": "Wesfarmers Ltd",
    "CSL.AX": "CSL Ltd",
    "FMG.AX": "Fortescue Metals Group",
}

st.set_page_config(
    page_title="🧬 Cyberpunk AI Trading Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Cyberpunk Styling Helpers
# -----------------------------
CYBER_CSS = """
<style>
body {
    background-color: #050608;
    color: #e0ffe0;
}
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top, #101820 0, #050608 60%);
    border-right: 1px solid #22ff88;
}
h1, h2, h3, h4 {
    color: #22ff88;
    text-shadow: 0 0 10px #22ff88;
}
div.stButton > button {
    background: linear-gradient(90deg, #22ff88, #7b2cff);
    color: #050608;
    border-radius: 999px;
    border: 1px solid #22ff88;
    box-shadow: 0 0 10px #22ff88;
}
div.stButton > button:hover {
    box-shadow: 0 0 20px #7b2cff;
}
.css-1d391kg, .stSelectbox, .stSlider, .stTextInput {
    color: #e0ffe0 !important;
}
</style>
"""

st.markdown(CYBER_CSS, unsafe_allow_html=True)

# -----------------------------
# Sidebar — Simulation Settings
# -----------------------------
st.sidebar.title("⚙️ Cyberpunk Settings Panel")

st.sidebar.subheader("Ticker Selection")

ticker_options = [f"{code} — {name}" for code, name in TICKER_LIST.items()]
selected_ticker = st.sidebar.selectbox(
    "Choose a stock",
    ticker_options,
    index=0,
    help="Select a stock by its ticker code and full company name. Data is pulled from a dystopian market grid."
)
ticker = selected_ticker.split(" — ")[0]

manual_ticker = st.sidebar.text_input(
    "Or enter a custom ticker",
    help="Override the dropdown with any valid ticker symbol. Use with caution in the neon wasteland."
)
if manual_ticker.strip():
    ticker = manual_ticker.strip().upper()

st.sidebar.subheader("Basic Settings")

data_period = st.sidebar.selectbox(
    "Data Period",
    ["7d", "30d", "60d", "90d", "180d"],
    index=2,
    help="How much historical data the AI loads from the grid. Longer periods = more training context."
)

initial_equity = st.sidebar.number_input(
    "Initial Equity ($)",
    value=10000.0,
    step=500.0,
    help="Starting capital for the simulation. Purely virtual credits in this cyberpunk sandbox."
)

st.sidebar.subheader("Risk Settings")

stop_pct = st.sidebar.slider(
    "Stop-loss %",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    help="Percentage drop from entry price where the AI auto-exits to contain damage. Your digital parachute."
) / 100.0

tp_pct = st.sidebar.slider(
    "Take-profit %",
    min_value=1.0,
    max_value=20.0,
    value=4.0,
    help="Percentage gain from entry price where the AI locks in profit. Your neon cash-out threshold."
) / 100.0

st.sidebar.subheader("Model Sensitivity")

entry_threshold = st.sidebar.slider(
    "Entry Probability Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.01,
    help="How confident the AI must be before entering a BUY trade. Higher = fewer, more selective signals."
)

with st.sidebar.expander("Advanced Settings", expanded=False):
    interval = st.selectbox(
        "Data Interval",
        ["1h", "30m", "15m"],
        index=0,
        help="Timeframe of each candle. Shorter intervals mean more trades, more noise, more chaos."
    )

    daily_loss_limit = st.slider(
        "Daily Loss Limit (%)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        help="Max percentage of equity the AI is allowed to lose in a single day before shutting down operations."
    ) / 100.0

run_button = st.sidebar.button("⚡ Run Neon Simulation")

# -----------------------------
# Main Title
# -----------------------------
st.title("🧬 Cyberpunk AI Hourly Trading Simulator")
st.markdown(
    "Welcome to the **neon grid**. This is an educational, simulated AI trader — "
    "no real orders, no real money, just pure cyberpunk sandbox."
)

# -----------------------------
# Run Simulation
# -----------------------------
if run_button:
    info, perf = get_company_info_and_performance(ticker)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("🏢 Entity Profile")
        st.write(f"**Name:** {info['name']}")
        st.write(f"**Exchange:** {info['exchange']}")
        st.write(f"**Sector:** {info['sector']}")
        st.write(f"**Country:** {info['country']}")

    with col2:
        st.subheader("📈 Recent Performance Scan")

        def fmt(v):
            return "N/A" if v is None else f"{v:.2f}%"

        st.write(f"**1 Day:** {fmt(perf['1D'])}")
        st.write(f"**1 Week:** {fmt(perf['1W'])}")
        st.write(f"**1 Month:** {fmt(perf['1M'])}")
        st.write(f"**3 Months:** {fmt(perf['3M'])}")

    st.info(f"Pulling {interval} data for **{ticker}** over {data_period} from the neon market grid...")
    df = download_hourly(ticker, period=data_period, interval=interval)

    if df.empty:
        st.warning("No data available for this ticker, period, and interval. Try a different combination in the grid.")
    else:
        st.info("Spinning up the rogue AI trader...")
        equity_curve, final_equity, trade_log = simulate_hourly(
            df,
            initial_equity=initial_equity,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            entry_threshold=entry_threshold,
            daily_loss_pct=daily_loss_limit,
        )

        if equity_curve.empty:
            st.warning("Not enough data to safely train the model. Extend the data period or use a coarser interval.")
        else:
            # -----------------------------
            # Neon Equity Curve with Markers
            # -----------------------------
            st.subheader("📊 Neon Equity Grid with AI Trades")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.set_facecolor("#050608")
            ax.plot(
                equity_curve["timestamp"],
                equity_curve["equity"],
                color="#22ff88",
                label="Equity (Neon Line)",
            )

            if trade_log:
                trades_df = pd.DataFrame(trade_log)
                eq_indexed = equity_curve.set_index("timestamp")

                buys = trades_df[trades_df["action"] == "BUY"]
                sells = trades_df[trades_df["action"] == "SELL"]

                if not buys.empty:
                    ax.scatter(
                        buys["timestamp"],
                        [eq_indexed.loc[t, "equity"] for t in buys["timestamp"]],
                        marker="^",
                        color="#39ff14",
                        edgecolors="#ffffff",
                        label="BUY (Signal)",
                        zorder=5,
                    )
                if not sells.empty:
                    ax.scatter(
                        sells["timestamp"],
                        [eq_indexed.loc[t, "equity"] for t in sells["timestamp"]],
                        marker="v",
                        color="#ff2efc",
                        edgecolors="#ffffff",
                        label="SELL (Exit)",
                        zorder=5,
                    )

            ax.set_xlabel("Time")
            ax.set_ylabel("Equity (credits)")
            ax.grid(True, alpha=0.2, color="#7b2cff")
            ax.legend()
            st.pyplot(fig)

            # -----------------------------
            # Performance Summary
            # -----------------------------
            st.subheader("📦 Simulation Performance Summary")

            perf_box = st.selectbox(
                "Choose a performance metric to view:",
                ["Overall Return", "Final Equity", "Initial Equity", "Return %"],
                help="Select which performance metric to highlight in the neon HUD."
            )

            if perf_box == "Overall Return":
                st.metric(
                    label="Overall Return",
                    value=f"{final_equity - initial_equity:,.2f}",
                    delta=f"{(final_equity / initial_equity - 1) * 100:.2f}%"
                )
            elif perf_box == "Final Equity":
                st.metric(
                    label="Final Equity",
                    value=f"{final_equity:,.2f}",
                    delta=f"{(final_equity / initial_equity - 1) * 100:.2f}%"
                )
            elif perf_box == "Initial Equity":
                st.metric(
                    label="Initial Equity",
                    value=f"{initial_equity:,.2f}"
                )
            elif perf_box == "Return %":
                st.metric(
                    label="Return %",
                    value=f"{(final_equity / initial_equity - 1) * 100:.2f}%"
                )

            # -----------------------------
            # Cyberpunk Trade Cards
            # -----------------------------
            st.subheader("🎴 Neon Trade Holo-Cards")

            if len(trade_log) == 0:
                st.info("No trades were triggered by the AI in this run.")
            else:
                trades_df = pd.DataFrame(trade_log)
                for _, trade in trades_df.iterrows():
                    card = st.container()
                    with card:
                        c1, c2, c3 = st.columns([2, 2, 3])
                        with c1:
                            st.markdown(
                                f"<div style='border:1px solid #39ff14;"
                                f"padding:8px;border-radius:8px;"
                                f"box-shadow:0 0 10px #39ff14;'>"
                                f"<b>{trade['action']}</b><br>"
                                f"<small>{trade['timestamp']}</small></div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Price:** {trade['price']:.2f}")
                            st.markdown(f"**Qty:** {trade['qty']}")
                        with c2:
                            if trade["action"] == "BUY":
                                st.markdown(f"**Prob up:** {trade.get('prob_up', 0):.2f}")
                                st.markdown(f"**SL:** {trade.get('stop_loss', 0):.2f}")
                                st.markdown(f"**TP:** {trade.get('take_profit', 0):.2f}")
                            else:
                                st.markdown(f"**PnL:** {trade.get('pnl', 0):.2f}")
                                st.markdown(f"**Exit:** {trade.get('exit_reason', 'N/A')}")
                        with c3:
                            st.markdown(
                                f"<div style='font-size:0.9em;color:#e0ffe0;"
                                f"text-shadow:0 0 6px #7b2cff;'>"
                                f"{trade.get('comment', 'No comment')}</div>",
                                unsafe_allow_html=True,
                            )

            # -----------------------------
            # AI Decision Feed (Glitch Log)
            # -----------------------------
            st.subheader("🧠 Rogue AI Decision Feed")

            with st.expander("Show glitch timeline"):
                if len(trade_log) == 0:
                    st.write("No decisions logged yet in this run.")
                else:
                    for trade in trade_log:
                        ts = trade["timestamp"]
                        if trade["action"] == "BUY":
                            st.markdown(
                                f"<span style='color:#39ff14;'>"
                                f"[{ts}] SIGNAL: BUY — Prob {trade.get('prob_up', 0):.2f}, "
                                f"SL {trade.get('stop_loss', 0):.2f}, "
                                f"TP {trade.get('take_profit', 0):.2f}"
                                f"</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<span style='color:#ff2efc;'>"
                                f"[{ts}] EXIT: SELL — PnL {trade.get('pnl', 0):.2f}, "
                                f"Reason: {trade.get('exit_reason', 'N/A')}"
                                f"</span>",
                                unsafe_allow_html=True,
                            )

            # -----------------------------
            # Detailed Stats
            # -----------------------------
            with st.expander("📘 Detailed Statistics (System Report)"):
                st.write(f"**Initial equity:** {initial_equity:,.2f}")
                st.write(f"**Final equity:** {final_equity:,.2f}")
                st.write(f"**Return:** {(final_equity / initial_equity - 1) * 100:.2f}%")
                st.write("System note: All activity is simulated. No real trades are executed.")