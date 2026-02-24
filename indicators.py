"""
indicators.py — computes technical indicators from OHLCV DataFrames.
All indicators are implemented from scratch using pandas/numpy so we
don't depend on the broken 'ta' package.
"""

import pandas as pd
import numpy as np


# ── Trend ──────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def macd(close: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def ema_crossover(close: pd.Series,
                  short: int = 9, long: int = 21
                  ) -> tuple[pd.Series, pd.Series]:
    """Returns (short_ema, long_ema). Cross above → bullish, cross below → bearish."""
    return ema(close, short), ema(close, long)


# ── Momentum ───────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3
               ) -> tuple[pd.Series, pd.Series]:
    """Returns (%K, %D)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


# ── Volatility ─────────────────────────────────────────────────────────────

def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0
                    ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    middle = sma(close, period)
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ── Volume ─────────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period).mean()


# ── Signal generator ───────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> dict:
    """
    Given an OHLCV DataFrame, compute all indicators and return a dict
    with the latest values + plain-English interpretations.
    """
    if df.empty or len(df) < 30:
        return {}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # Compute
    rsi_vals                    = rsi(close)
    macd_line, sig_line, hist   = macd(close)
    short_ema, long_ema         = ema_crossover(close)
    bb_upper, bb_mid, bb_lower  = bollinger_bands(close)
    stoch_k, stoch_d            = stochastic(high, low, close)
    atr_vals                    = atr(high, low, close)
    obv_vals                    = obv(close, volume)
    vol_avg                     = volume_sma(volume)

    # Latest values
    latest = {
        "close":       round(float(close.iloc[-1]), 4),
        "rsi":         round(float(rsi_vals.iloc[-1]), 2),
        "macd":        round(float(macd_line.iloc[-1]), 4),
        "macd_signal": round(float(sig_line.iloc[-1]), 4),
        "macd_hist":   round(float(hist.iloc[-1]), 4),
        "ema_short":   round(float(short_ema.iloc[-1]), 4),
        "ema_long":    round(float(long_ema.iloc[-1]), 4),
        "bb_upper":    round(float(bb_upper.iloc[-1]), 4),
        "bb_mid":      round(float(bb_mid.iloc[-1]), 4),
        "bb_lower":    round(float(bb_lower.iloc[-1]), 4),
        "stoch_k":     round(float(stoch_k.iloc[-1]), 2),
        "stoch_d":     round(float(stoch_d.iloc[-1]), 2),
        "atr":         round(float(atr_vals.iloc[-1]), 4),
        "volume":      int(volume.iloc[-1]),
        "volume_avg":  int(vol_avg.iloc[-1]),
    }

    # Interpretations
    interp = {}

    # RSI
    r = latest["rsi"]
    if r < 30:
        interp["rsi"] = f"Oversold ({r:.1f}) — potential bounce"
    elif r > 70:
        interp["rsi"] = f"Overbought ({r:.1f}) — potential pullback"
    else:
        interp["rsi"] = f"Neutral ({r:.1f})"

    # MACD
    if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
        interp["macd"] = "Bullish — MACD above signal, histogram positive"
    elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
        interp["macd"] = "Bearish — MACD below signal, histogram negative"
    else:
        interp["macd"] = "Crossing — potential trend change imminent"

    # EMA crossover
    if latest["ema_short"] > latest["ema_long"]:
        interp["ema_cross"] = "Bullish — short EMA (9) above long EMA (21)"
    else:
        interp["ema_cross"] = "Bearish — short EMA (9) below long EMA (21)"

    # Bollinger Bands
    c = latest["close"]
    if c > latest["bb_upper"]:
        interp["bb"] = "Price above upper band — overbought / breakout"
    elif c < latest["bb_lower"]:
        interp["bb"] = "Price below lower band — oversold / breakdown"
    else:
        pct = (c - latest["bb_lower"]) / (latest["bb_upper"] - latest["bb_lower"]) * 100
        interp["bb"] = f"Price within bands ({pct:.0f}% from bottom)"

    # Stochastic
    sk, sd = latest["stoch_k"], latest["stoch_d"]
    if sk < 20 and sd < 20:
        interp["stoch"] = f"Oversold (%K={sk:.1f}, %D={sd:.1f})"
    elif sk > 80 and sd > 80:
        interp["stoch"] = f"Overbought (%K={sk:.1f}, %D={sd:.1f})"
    else:
        interp["stoch"] = f"Neutral (%K={sk:.1f}, %D={sd:.1f})"

    # Volume
    vol_ratio = latest["volume"] / latest["volume_avg"] if latest["volume_avg"] else 1
    if vol_ratio > 1.5:
        interp["volume"] = f"High volume ({vol_ratio:.1f}x avg) — strong conviction"
    elif vol_ratio < 0.5:
        interp["volume"] = f"Low volume ({vol_ratio:.1f}x avg) — weak conviction"
    else:
        interp["volume"] = f"Normal volume ({vol_ratio:.1f}x avg)"

    # Simple composite score: -3 to +3
    score = 0
    if r < 35:   score += 1
    elif r > 65: score -= 1
    if latest["macd"] > latest["macd_signal"]: score += 1
    else: score -= 1
    if latest["ema_short"] > latest["ema_long"]: score += 1
    else: score -= 1

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "values":  latest,
        "interp":  interp,
        "score":   score,
        "signal":  signal,
        # Raw series for charting
        "series": {
            "rsi":       rsi_vals,
            "macd_line": macd_line,
            "sig_line":  sig_line,
            "hist":      hist,
            "bb_upper":  bb_upper,
            "bb_mid":    bb_mid,
            "bb_lower":  bb_lower,
            "ema_short": short_ema,
            "ema_long":  long_ema,
            "obv":       obv_vals,
        }
    }
