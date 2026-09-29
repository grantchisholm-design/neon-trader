import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# -----------------------------
# Download + Clean Data
# -----------------------------
def download_hourly(ticker, period="60d", interval="1h"):
    df = yf.download(ticker, interval=interval, period=period)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.dropna()

    if len(df) < 100:
        for extra_period in ["90d", "180d"]:
            df2 = yf.download(ticker, interval=interval, period=extra_period)
            if isinstance(df2.columns, pd.MultiIndex):
                df2.columns = [col[0] for col in df2.columns]
            df2 = df2.dropna()
            if len(df2) >= 100:
                return df2
    return df

# -----------------------------
# Company Info + Performance
# -----------------------------
def get_company_info_and_performance(ticker):
    ticker_obj = yf.Ticker(ticker)
    info_raw = ticker_obj.info

    info = {
        "name": info_raw.get("longName") or info_raw.get("shortName") or ticker,
        "exchange": info_raw.get("exchange") or "Unknown",
        "sector": info_raw.get("sector") or "Unknown",
        "country": info_raw.get("country") or "Unknown",
    }

    hist = ticker_obj.history(period="6mo", interval="1d")
    if hist.empty:
        return info, {"1D": None, "1W": None, "1M": None, "3M": None}

    hist = hist.dropna()
    latest_close = hist["Close"].iloc[-1]

    def pct_change_days(days):
        if len(hist) <= days:
            return None
        past_close = hist["Close"].iloc[-days]
        return (latest_close / past_close - 1) * 100.0

    perf = {
        "1D": pct_change_days(1),
        "1W": pct_change_days(5),
        "1M": pct_change_days(21),
        "3M": pct_change_days(63),
    }

    return info, perf

# -----------------------------
# Feature Engineering
# -----------------------------
def build_features(df):
    df["ret_1"] = df["Close"].pct_change()

    n = len(df)
    w20 = max(5, min(20, n // 5))
    w50 = max(10, min(50, n // 3))

    df["ma_20"] = df["Close"].rolling(w20).mean()
    df["ma_50"] = df["Close"].rolling(w50).mean()
    df["vol_20"] = df["ret_1"].rolling(w20).std()
    df["trend_up"] = (df["ma_20"] > df["ma_50"]).astype(int)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df

# -----------------------------
# Train Model
# -----------------------------
def train_model(df):
    features = ["ret_1", "ma_20", "ma_50", "vol_20", "trend_up"]
    X = df[features]
    y = df["target"]

    if len(X) < 50:
        return None, features

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, features

# -----------------------------
# Risk Manager
# -----------------------------
class RiskManager:
    def __init__(self, risk_pct=0.01, daily_loss_pct=0.03):
        self.risk_pct = risk_pct
        self.daily_loss_pct = daily_loss_pct
        self.daily_pnl = 0.0

    def reset_day(self):
        self.daily_pnl = 0.0

    def can_trade(self, equity):
        return self.daily_pnl > -self.daily_loss_pct * equity

    def position_size(self, equity, stop_distance_price):
        risk_dollars = equity * self.risk_pct
        if stop_distance_price <= 0:
            return 0
        return int(risk_dollars / stop_distance_price)

    def update_pnl(self, pnl):
        self.daily_pnl += pnl

# -----------------------------
# Simulation Engine
# -----------------------------
def simulate_hourly(
    df,
    initial_equity=10000.0,
    stop_pct=0.02,
    tp_pct=0.04,
    entry_threshold=0.60,
    daily_loss_pct=0.03,
):
    df = build_features(df)

    if len(df) < 50:
        return pd.DataFrame(), initial_equity, []

    model, features = train_model(df)
    if model is None:
        return pd.DataFrame(), initial_equity, []

    risk = RiskManager(daily_loss_pct=daily_loss_pct)
    equity = initial_equity
    position = 0
    entry_price = None
    equity_curve = []
    trade_log = []

    current_day = None

    for idx, row in df.iterrows():
        price = float(row["Close"])
        day = idx.date()

        if day != current_day:
            current_day = day
            risk.reset_day()

        # Manage open position
        if position != 0 and entry_price is not None:
            stop_price = entry_price * (1 - stop_pct)
            tp_price = entry_price * (1 + tp_pct)

            if price <= stop_price or price >= tp_price:
                pnl = (price - entry_price) * position
                equity += pnl
                risk.update_pnl(pnl)

                trade_log.append({
                    "timestamp": idx,
                    "action": "SELL",
                    "price": price,
                    "qty": position,
                    "pnl": pnl,
                    "exit_reason": "TP" if price >= tp_price else "SL",
                    "comment": "SYSTEM: Exit executed — "
                               + ("profit secured" if price >= tp_price else "loss contained"),
                })

                position = 0
                entry_price = None

        if not risk.can_trade(equity):
            equity_curve.append(equity)
            continue

        X_row = row[features].values.reshape(1, -1)
        prob_up = model.predict_proba(X_row)[0, 1]

        if prob_up > entry_threshold and position == 0:
            stop_distance = price * stop_pct
            qty = risk.position_size(equity, stop_distance)

            if qty > 0:
                position = qty
                entry_price = price

                trade_log.append({
                    "timestamp": idx,
                    "action": "BUY",
                    "price": price,
                    "qty": qty,
                    "prob_up": prob_up,
                    "stop_loss": price * (1 - stop_pct),
                    "take_profit": price * (1 + tp_pct),
                    "comment": f"SIGNAL: Confidence {prob_up:.2f} > {entry_threshold:.2f} — long bias engaged.",
                })

        equity_curve.append(equity)

    result = pd.DataFrame({
        "timestamp": df.index[:len(equity_curve)],
        "equity": equity_curve
    })

    return result, equity, trade_log