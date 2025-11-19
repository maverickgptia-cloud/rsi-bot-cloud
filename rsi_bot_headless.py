# rsi_bot_headless.py — Bot ligero (paper) + log CSV tabular
# Dependencias: pandas, numpy, yfinance
# Guarda/añade operaciones en: ~/trades_log.csv (con cabeceras)

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone as tz

# ===================== Parámetros (con overrides por variables de entorno) =====================
SYMBOL      = os.environ.get("BOT_SYMBOL", "BTC-USD")
INTERVAL    = os.environ.get("BOT_INTERVAL", "60m")   # 60m = 1h
PERIOD      = os.environ.get("BOT_PERIOD", "180d")    # histórico descargado
LOOKBACK    = int(os.environ.get("BOT_LOOKBACK", "800"))

# Estrategia y gestión de riesgo
EMA_FAST    = int(os.environ.get("BOT_EMA_FAST", "20"))
EMA_SLOW    = int(os.environ.get("BOT_EMA_SLOW", "100"))
RSI_BUY     = float(os.environ.get("BOT_RSI_BUY", "30"))
RSI_SELL    = float(os.environ.get("BOT_RSI_SELL", "70"))
EMA_SPREAD  = float(os.environ.get("BOT_EMA_SPREAD", "0.1"))   # % separación mínima EMAs
ATR_PERIOD  = int(os.environ.get("BOT_ATR_PERIOD", "14"))
ATR_STOP    = float(os.environ.get("BOT_ATR_STOP", "1.2"))     # SL = 1.2 * ATR
ATR_TAKE    = float(os.environ.get("BOT_ATR_TAKE", "2.4"))     # TP = 2.4 * ATR
RISK_PCT    = float(os.environ.get("BOT_RISK_PCT", "0.02"))    # 2% del capital en riesgo
FEE_BPS     = float(os.environ.get("BOT_FEE_BPS", "5.0"))      # 0.05% por lado
INIT_CAP    = float(os.environ.get("BOT_INIT_CAP", "100.0"))   # Capital inicial por ejecución

LOG_PATH    = os.path.expanduser("~/trades_log.csv")
TIMEFRAME   = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "1h", "1h": "1h", "4h": "4h", "1d": "1d"
}.get(INTERVAL, INTERVAL)

# =============================== Utilidades de indicadores ===============================
def ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=int(w), adjust=False).mean()

def rsi(c: pd.Series, period: int = 14) -> pd.Series:
    d  = c.diff()
    g  = d.clip(lower=0.0)
    l  = (-d).clip(lower=0.0)
    ag = g.rolling(period).mean()
    al = l.rolling(period).mean().replace(0, np.nan)
    rs = ag / al
    return 100 - 100 / (1 + rs)

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def safe_float(x):
    """Convierte a float o devuelve NaN si no es convertible."""
    try:
        return float(x)
    except Exception:
        return np.nan

# ================================== Datos e indicadores ==================================
def fetch_ohlc() -> pd.DataFrame:
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "close", "Volume": "volume"
    })
    df.index.name = "ts"
    df = df.dropna().tail(LOOKBACK).astype(float)
    return df

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    df["ema_fast"] = ema(c, EMA_FAST)
    df["ema_slow"] = ema(c, EMA_SLOW)
    df["rsi"]      = rsi(c, 14)
    df["atr"]      = atr(h, l, c, ATR_PERIOD)

    base_buy  = (df["rsi"].shift(1) <= RSI_BUY)  & (df["rsi"] > RSI_BUY)
    base_sell = (df["rsi"].shift(1) >= RSI_SELL) & (df["rsi"] < RSI_SELL)

    trend_long  = df["ema_fast"] > df["ema_slow"]
    trend_short = df["ema_fast"] < df["ema_slow"]

    spread_ok = (df["ema_fast"] - df["ema_slow"]).abs() / df["ema_slow"].abs() * 100 > EMA_SPREAD

    df["buy_sig"]  = base_buy  & trend_long  & spread_ok
    df["sell_sig"] = base_sell & trend_short & spread_ok
    return df

# ================================ Backtest (paper trading) ================================
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
    ts, symbol, timeframe, side, price, qty, pnl, capital, rsi, ema_fast, ema_slow, atr, reason
    """
    fee = FEE_BPS / 1e4
    capital = INIT_CAP
    pos_qty = 0.0
    entry = stop = take = None

    rows = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        ts   = df.index[i]
        c    = safe_float(row["close"])
        h    = safe_float(row["high"])
        l    = safe_float(row["low"])
        rsi_v = safe_float(row["rsi"])
        atr_v = safe_float(row["atr"])
        ef    = safe_float(row["ema_fast"])
        es    = safe_float(row["ema_slow"])

        # ---- Salidas (TP / SL / señal de venta) ----
        if pos_qty > 0.0:
            px = None
            reason = None
            if take is not None and h >= take:
                px, reason = take, "TP"
            elif stop is not None and l <= stop:
                px, reason = stop, "SL"
            elif bool(row["sell_sig"]):
                px, reason = c, "SELL_SIG"

            if px is not None:
                proceeds = pos_qty * px * (1 - fee)
                cost_in  = pos_qty * entry * (1 + fee)
                pnl      = proceeds - cost_in
                capital += proceeds
                rows.append({
                    "ts": pd.to_datetime(ts).tz_localize(None),
                    "symbol": SYMBOL,
                    "timeframe": TIMEFRAME,
                    "side": "SELL",
                    "price": round(px, 2),
                    "qty": round(pos_qty, 8),
                    "pnl": round(pnl, 2),
                    "capital": round(capital, 2),
                    "rsi": round(rsi_v, 2) if np.isfinite(rsi_v) else np.nan,
                    "ema_fast": round(ef, 2) if np.isfinite(ef) else np.nan,
                    "ema_slow": round(es, 2) if np.isfinite(es) else np.nan,
                    "atr": round(atr_v, 2) if np.isfinite(atr_v) else np.nan,
                    "reason": reason,
                })
                pos_qty = 0.0
                entry = stop = take = None

        # ---- Entradas ----
        if (
            pos_qty == 0.0
            and bool(row["buy_sig"])
            and np.isfinite(atr_v)
            and atr_v > 0
            and np.isfinite(c)
        ):
            risk_cash = capital * RISK_PCT
            stop_dist = ATR_STOP * atr_v
            qty = risk_cash / max(stop_dist, 1e-9)
            if qty * c > capital:
                qty = capital / c
            if qty > 0:
                cost = qty * c * (1 + fee)
                if cost <= capital:
                    capital -= cost
                    pos_qty = qty
                    entry   = c
                    stop    = c - stop_dist
                    take    = c + ATR_TAKE * atr_v
                    rows.append({
                        "ts": pd.to_datetime(ts).tz_localize(None),
                        "symbol": SYMBOL,
                        "timeframe": TIMEFRAME,
                        "side": "BUY",
                        "price": round(entry, 2),
                        "qty": round(pos_qty, 8),
                        "pnl": 0.0,
                        "capital": round(capital, 2),
                        "rsi": round(rsi_v, 2) if np.isfinite(rsi_v) else np.nan,
                        "ema_fast": round(ef, 2) if np.isfinite(ef) else np.nan,
                        "ema_slow": round(es, 2) if np.isfinite(es) else np.nan,
                        "atr": round(atr_v, 2) if np.isfinite(atr_v) else np.nan,
                        "reason": "ENTRY",
                    })

    return pd.DataFrame(
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

# ===================================== Logging CSV ======================================
def append_log_csv(trades: pd.DataFrame, path: str):
    """Anexa con cabecera si el archivo no existe o está vacío."""
    if trades.empty:
        return
    path = os.path.expanduser(path)
    header_needed = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    trades.to_csv(path, mode="a", header=header_needed, index=False)

# ======================================== Main ==========================================
def main():
    print(f"[{datetime.now(tz.utc).isoformat()}] Bot {SYMBOL} {TIMEFRAME} iniciado…")
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

