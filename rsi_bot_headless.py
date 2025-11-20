# rsi_bot_headless.py — Bot ligero (paper) + log CSV tabular
# Dependencias: pandas, numpy, yfinance
# Guarda/añade operaciones en: ~/trades_log.csv (con cabeceras)

import os
from datetime import datetime, timezone as tz

import numpy as np
import pandas as pd
import yfinance as yf

# ===================== Parámetros (con overrides por variables de entorno) =====================
SYMBOL = os.environ.get("BOT_SYMBOL", "BTC-USD")
INTERVAL = os.environ.get("BOT_INTERVAL", "60m")   # 60m = 1h
PERIOD = os.environ.get("BOT_PERIOD", "180d")      # histórico descargado
LOOKBACK = int(os.environ.get("BOT_LOOKBACK", "800"))

# Estrategia y gestión de riesgo
EMA_FAST = int(os.environ.get("BOT_EMA_FAST", "20"))
EMA_SLOW = int(os.environ.get("BOT_EMA_SLOW", "100"))
RSI_BUY = float(os.environ.get("BOT_RSI_BUY", "30"))
RSI_SELL = float(os.environ.get("BOT_RSI_SELL", "70"))
EMA_SPREAD = float(os.environ.get("BOT_EMA_SPREAD", "0.1"))     # % separación mínima EMAs
ATR_PERIOD = int(os.environ.get("BOT_ATR_PERIOD", "14"))
ATR_STOP = float(os.environ.get("BOT_ATR_STOP", "1.2"))         # SL = 1.2 * ATR
ATR_TAKE = float(os.environ.get("BOT_ATR_TAKE", "2.4"))         # TP = 2.4 * ATR
RISK_PCT = float(os.environ.get("BOT_RISK_PCT", "0.02"))        # 2% del capital en riesgo
FEE_BPS = float(os.environ.get("BOT_FEE_BPS", "5.0"))           # 0.05% por lado
INIT_CAP = float(os.environ.get("BOT_INIT_CAP", "100.0"))       # Capital inicial por ejecución

LOG_PATH = os.path.expanduser("~/trades_log.csv")
TIMEFRAME = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}.get(INTERVAL, INTERVAL)

# =============================== Utilidades de indicadores ===============================

def ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=int(w), adjust=False).mean()

def rsi(c: pd.Series, period: int = 14) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0.0)
    l = (-d).clip(lower=0.0)
    ag = g.rolling(period).mean()
    al = l.rolling(period).mean().replace(0, np.nan)
    rs = ag / al
    return 100 - 100 / (1 + rs)

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev).abs(), (l - prev).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def safe_float(x, default=np.nan) -> float:
    """Convierte a float sin romper si viene algo raro."""
    try:
        return float(x)
    except Exception:
        return default

def to_bool(val) -> bool:
    """Convierte a bool evitando el error de Series ambiguo."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Series, pd.Array)):
        # Si por lo que sea llega un Series, usamos .iloc[0] o any()
        try:
            return bool(val.iloc[0])
        except Exception:
            return bool(val.any())
    if pd.isna(val):
        return False
    return bool(val)

# ================================== Datos e indicadores ==================================

def fetch_ohlc() -> pd.DataFrame:
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        }
    )
    df.index.name = "ts"
    df = df.dropna().tail(LOOKBACK)
    # Aseguramos float
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    df["ema_fast"] = ema(c, EMA_FAST)
    df["ema_slow"] = ema(c, EMA_SLOW)
    df["rsi"] = rsi(c, 14)
    df["atr"] = atr(h, l, c, ATR_PERIOD)

    # Señales base de RSI (cruces)
    base_buy = (df["rsi"].shift(1) <= RSI_BUY) & (df["rsi"] > RSI_BUY)
    base_sell = (df["rsi"].shift(1) >= RSI_SELL) & (df["rsi"] < RSI_SELL)

    # Tendencia
    trend_long = df["ema_fast"] > df["ema_slow"]
    trend_short = df["ema_fast"] < df["ema_slow"]

    # Spread mínimo entre EMAs
    spread_ok = (
        (df["ema_fast"] - df["ema_slow"]).abs()
        / df["ema_slow"].abs()
        * 100
        > EMA_SPREAD
    )

    df["buy_sig"] = base_buy & trend_long & spread_ok
    df["sell_sig"] = base_sell & trend_short & spread_ok
    return df

# ================================ Backtest (paper trading) ================================

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
    ts, symbol, timeframe, side, price, qty, pnl, capital,
    rsi, ema_fast, ema_slow, atr, reason
    """
    if df.empty or "buy_sig" not in df.columns:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "timeframe",
                "side",
                "price",
                "qty",
                "pnl",
                "capital",
                "rsi",
                "ema_fast",
                "ema_slow",
                "atr",
                "reason",
            ]
        )

    fee = FEE_BPS / 1e4
    capital = INIT_CAP
    pos_qty = 0.0
    entry = None
    stop = None
    take = None

    rows = []

    for ts, row in df.iterrows():
        price = safe_float(row.get("close"))
        high = safe_float(row.get("high"))
        low = safe_float(row.get("low"))

        if np.isnan(price) or np.isnan(high) or np.isnan(low):
            continue

        rsi_v = safe_float(row.get("rsi"))
        atr_v = safe_float(row.get("atr"))
        ef = safe_float(row.get("ema_fast"))
        es = safe_float(row.get("ema_slow"))

        buy_flag = to_bool(row.get("buy_sig", False))
        sell_flag = to_bool(row.get("sell_sig", False))

        # ---- Salidas (TP / SL / señal de venta) ----
        if pos_qty > 0.0:
            exit_price = None
            reason = None

            # Take profit
            if (take is not None) and (high >= take):
                exit_price = take
                reason = "TP"

            # Stop loss
            elif (stop is not None) and (low <= stop):
                exit_price = stop
                reason = "SL"

            # Señal contraria
            elif sell_flag:
                exit_price = price
                reason = "SELL_SIG"

            if exit_price is not None:
                proceeds = pos_qty * exit_price * (1 - fee)
                cost_in = pos_qty * entry * (1 + fee)
                pnl = proceeds - cost_in
                capital += proceeds

                rows.append(
                    {
                        "ts": pd.to_datetime(ts).tz_localize(None),
                        "symbol": SYMBOL,
                        "timeframe": TIMEFRAME,
                        "side": "SELL",
                        "price": round(exit_price, 2),
                        "qty": round(pos_qty, 8),
                        "pnl": round(pnl, 2),
                        "capital": round(capital, 2),
                        "rsi": round(rsi_v, 2) if not np.isnan(rsi_v) else np.nan,
                        "ema_fast": round(ef, 2) if not np.isnan(ef) else np.nan,
                        "ema_slow": round(es, 2) if not np.isnan(es) else np.nan,
                        "atr": round(atr_v, 2) if not np.isnan(atr_v) else np.nan,
                        "reason": reason,
                    }
                )

                pos_qty = 0.0
                entry = None
                stop = None
                take = None

        # ---- Entradas ----
        if (pos_qty == 0.0) and buy_flag and (atr_v is not None) and (atr_v > 0):
            risk_cash = capital * RISK_PCT
            stop_dist = ATR_STOP * atr_v

            if stop_dist <= 0:
                continue

            qty = risk_cash / stop_dist

            # No arriesgar más capital del que tenemos
            if qty * price > capital:
                qty = capital / price

            if qty <= 0:
                continue

            cost = qty * price * (1 + fee)
            if cost > capital:
                continue

            capital -= cost
            pos_qty = qty
            entry = price
            stop = price - stop_dist
            take = price + ATR_TAKE * atr_v

            rows.append(
                {
                    "ts": pd.to_datetime(ts).tz_localize(None),
                    "symbol": SYMBOL,
                    "timeframe": TIMEFRAME,
                    "side": "BUY",
                    "price": round(entry, 2),
                    "qty": round(pos_qty, 8),
                    "pnl": 0.0,
                    "capital": round(capital, 2),
                    "rsi": round(rsi_v, 2) if not np.isnan(rsi_v) else np.nan,
                    "ema_fast": round(ef, 2) if not np.isnan(ef) else np.nan,
                    "ema_slow": round(es, 2) if not np.isnan(es) else np.nan,
                    "atr": round(atr_v, 2) if not np.isnan(atr_v) else np.nan,
                    "reason": "ENTRY",
                }
            )

    trades = pd.DataFrame(
        rows,
        columns=[
            "ts",
            "symbol",
            "timeframe",
            "side",
            "price",
            "qty",
            "pnl",
            "capital",
            "rsi",
            "ema_fast",
            "ema_slow",
            "atr",
            "reason",
        ],
    )
    return trades

# ===================================== Logging CSV ======================================

def append_log_csv(trades: pd.DataFrame, path: str):
    """Anexa con cabecera si el archivo no existe o está vacío."""
    if trades is None or trades.empty:
        return
    path = os.path.expanduser(path)
    header_needed = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    trades.to_csv(path, mode="a", header=header_needed, index=False)

# ======================================== Main ==========================================

def main():
    now = datetime.now(tz.utc).isoformat()
    print(f"[{now}] Bot {SYMBOL} {TIMEFRAME} iniciado…")

    df = fetch_ohlc()
    if df.empty:
        print("Sin datos OHLC — fin.")
        return

    df = compute_signals(df)
    trades = backtest(df)
    append_log_csv(trades, LOG_PATH)

    print(f"Filas nuevas registradas: {len(trades)}  | Log: {LOG_PATH}")

if __name__ == "__main__":
    main()
