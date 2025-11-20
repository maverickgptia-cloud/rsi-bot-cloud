# rsi_bot_headless.py — Bot ligero (paper) + log CSV tabular
# Dependencias: pandas, numpy, yfinance
# Guarda/añade operaciones en: ~/trades_log.csv (con cabeceras)

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone as tz

# ===================== Helpers para leer variables de entorno =====================

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    # Limpieza básica de espacios
    return str(v).strip()

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, str(default))
    v = str(v).strip()
    # Reemplazamos coma por punto por si acaso
    v = v.replace(",", ".")
    try:
        return int(float(v))
    except ValueError:
        return int(default)

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, str(default))
    v = str(v).strip()
    v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return float(default)

# ===================== Parámetros (con overrides por variables de entorno) =====================

SYMBOL      = _env_str("BOT_SYMBOL",   "BTC-USD")
INTERVAL    = _env_str("BOT_INTERVAL", "60m")    # 60m = 1h
PERIOD      = _env_str("BOT_PERIOD",   "180d")   # histórico descargado
LOOKBACK    = _env_int("BOT_LOOKBACK", 800)

# Estrategia y gestión de riesgo
EMA_FAST    = _env_int("BOT_EMA_FAST",   20)
EMA_SLOW    = _env_int("BOT_EMA_SLOW",   100)
RSI_BUY     = _env_float("BOT_RSI_BUY",  30.0)
RSI_SELL    = _env_float("BOT_RSI_SELL", 70.0)
EMA_SPREAD  = _env_float("BOT_EMA_SPREAD", 0.10)   # % separación mínima EMAs
ATR_PERIOD  = _env_int("BOT_ATR_PERIOD",  14)
ATR_STOP    = _env_float("BOT_ATR_STOP",  1.2)     # SL = 1.2 * ATR
ATR_TAKE    = _env_float("BOT_ATR_TAKE",  2.4)     # TP = 2.4 * ATR
RISK_PCT    = _env_float("BOT_RISK_PCT",  0.02)    # 2% del capital en riesgo
FEE_BPS     = _env_float("BOT_FEE_BPS",   5.0)     # 0.05% por lado
INIT_CAP    = _env_float("BOT_INIT_CAP",  100.0)   # Capital inicial por ejecución

LOG_PATH    = os.path.expanduser("~/trades_log.csv")

TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "1h", "1h": "1h", "4h": "4h", "1d": "1d"
}
TIMEFRAME   = TIMEFRAME_MAP.get(INTERVAL, INTERVAL)

# =============================== Utilidades de indicadores ===============================

def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA simple sobre una serie (no la usamos directamente para evitar errores de asignación)."""
    return series.ewm(span=int(period), adjust=False).mean()

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

# ================================== Datos e indicadores ==================================

def fetch_ohlc() -> pd.DataFrame:
    """Descarga OHLC de yfinance y devuelve DataFrame con columnas:
    ts, open, high, low, close, volume (float)
    """
    print(f"Descargando datos de {SYMBOL} | PERIOD={PERIOD} | INTERVAL={INTERVAL}")
    df = yf.download(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        print("⚠ Sin datos desde yfinance.")
        return pd.DataFrame()

    # Si viene con MultiIndex (ticker en columnas), nos quedamos con el primer nivel (Open, High, Low, Close, Volume)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            # En caso raro, cogemos la primera columna de cada par
            df = df.copy()
            df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"⚠ Faltan columnas OHLC en datos: {missing}")
        return pd.DataFrame()

    df = df[cols].astype(float)
    df.index.name = "ts"
    df = df.dropna().tail(LOOKBACK)

    print(f"Filas OHLC descargadas: {len(df)}")
    return df

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    # EMAs (FORMA SEGURA: 1 sola columna cada una)
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # RSI y ATR
    df["rsi"] = rsi(c, 14)
    df["atr"] = atr(h, l, c, ATR_PERIOD)

    # Señales base RSI
    base_buy  = (df["rsi"].shift(1) <= RSI_BUY)  & (df["rsi"] > RSI_BUY)
    base_sell = (df["rsi"].shift(1) >= RSI_SELL) & (df["rsi"] < RSI_SELL)

    # Tendencia por EMAs
    trend_long  = df["ema_fast"] > df["ema_slow"]
    trend_short = df["ema_fast"] < df["ema_slow"]

    # Separación mínima entre EMAs (en %)
    spread = (df["ema_fast"] - df["ema_slow"]).abs() / df["ema_slow"].abs() * 100.0
    spread_ok = spread > EMA_SPREAD

    df["buy_sig"]  = base_buy  & trend_long  & spread_ok
    df["sell_sig"] = base_sell & trend_short & spread_ok

    return df

# ================================ Backtest (paper trading) ================================

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
    ts, symbol, timeframe, side, price, qty, pnl, capital, rsi, ema_fast, ema_slow, atr, reason
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ts", "symbol", "timeframe", "side", "price", "qty",
                "pnl", "capital", "rsi", "ema_fast", "ema_slow", "atr", "reason"
            ]
        )

    fee = FEE_BPS / 1e4
    capital = INIT_CAP
    pos_qty = 0.0
    entry = stop = take = None

    rows = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts  = df.index[i]

        c = float(row["close"])
        h = float(row["high"])
        l = float(row["low"])

        rsi_v = float(row["rsi"]) if np.isfinite(row["rsi"]) else np.nan
        atr_v = float(row["atr"]) if np.isfinite(row["atr"]) else None
        ef    = float(row["ema_fast"]) if np.isfinite(row["ema_fast"]) else np.nan
        es    = float(row["ema_slow"]) if np.isfinite(row["ema_slow"]) else np.nan

        # --------------------- Salidas (TP / SL / señal de venta) ---------------------
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
                    "rsi": round(rsi_v, 2),
                    "ema_fast": round(ef, 2),
                    "ema_slow": round(es, 2),
                    "atr": round(atr_v, 2) if atr_v else np.nan,
                    "reason": reason,
                })

                pos_qty = 0.0
                entry = stop = take = None

        # --------------------- Entradas ---------------------
        if pos_qty == 0.0 and bool(row["buy_sig"]) and atr_v and atr_v > 0:
            risk_cash = capital * RISK_PCT
            stop_dist = ATR_STOP * atr_v
            if stop_dist <= 0:
                continue

            qty = risk_cash / stop_dist
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
                        "rsi": round(rsi_v, 2),
                        "ema_fast": round(ef, 2),
                        "ema_slow": round(es, 2),
                        "atr": round(atr_v, 2) if atr_v else np.nan,
                        "reason": "ENTRY",
                    })

    return pd.DataFrame(
        rows,
        columns=[
            "ts", "symbol", "timeframe", "side", "price", "qty",
            "pnl", "capital", "rsi", "ema_fast", "ema_slow", "atr", "reason"
        ],
    )

# ===================================== Logging CSV ======================================

def append_log_csv(trades: pd.DataFrame, path: str):
    """Anexa con cabecera si el archivo no existe o está vacío."""
    if trades is None or trades.empty:
        print("Sin nuevas operaciones para registrar.")
        return

    path = os.path.expanduser(path)
    header_needed = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    trades.to_csv(path, mode="a", header=header_needed, index=False)
    print(f"Guardadas {len(trades)} filas nuevas en {path}")

# ======================================== Main ==========================================

def main():
    print(f"[{datetime.now(tz.utc).isoformat()}] Bot {SYMBOL} {TIMEFRAME} iniciado…")

    df = fetch_ohlc()
    if df.empty:
        print("Sin datos OHLC — fin de ejecución.")
        return

    df = compute_signals(df)
    trades = backtest(df)
    append_log_csv(trades, LOG_PATH)

    print(f"Filas nuevas registradas: {len(trades)}  | Log: {LOG_PATH}")
    if not df.empty:
        last = df.iloc[-1]
        print(
            f"Última vela: close={last['close']:.2f}, RSI={last['rsi']:.2f}, "
            f"EMA_FAST={last['ema_fast']:.2f}, EMA_SLOW={last['ema_slow']:.2f}"
        )

if __name__ == "__main__":
    main()
