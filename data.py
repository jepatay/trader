"""
data.py — fetches OHLCV price data and basic news headlines via yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_ohlcv(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for a ticker.
    period: '1mo', '3mo', '6mo', '1y'
    interval: '1d', '1h', '15m'
    Returns empty DataFrame on failure.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        print(f"[data] fetch_ohlcv failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_info(ticker: str) -> dict:
    """Return basic ticker metadata (name, currency, exchange, sector)."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "name":       info.get("longName") or info.get("shortName", ticker),
            "currency":   info.get("currency", "—"),
            "exchange":   info.get("exchange", "—"),
            "sector":     info.get("sector", "—"),
            "market_cap": info.get("marketCap"),
            "price":      info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        print(f"[data] fetch_info failed for {ticker}: {e}")
        return {"name": ticker, "currency": "—", "exchange": "—", "sector": "—",
                "market_cap": None, "price": None}


def fetch_news(ticker: str, max_items: int = 8) -> list[dict]:
    """Return recent news headlines for a ticker."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        results = []
        for item in news[:max_items]:
            results.append({
                "title":     item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link":      item.get("link", ""),
                "published": datetime.fromtimestamp(item.get("providerPublishTime", 0))
                             .strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "—",
            })
        return results
    except Exception as e:
        print(f"[data] fetch_news failed for {ticker}: {e}")
        return []
