"""
watchlist.py — persists the user's ticker watchlist to a local JSON file.
"""

import json
import os

_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

_DEFAULTS = [
    "NOVO-B.CO",   # Novo Nordisk (Denmark)
    "CARL-B.CO",   # Carlsberg (Denmark)
    "ORSTED.CO",   # Ørsted (Denmark)
    "AAPL",        # Apple (US)
    "NVDA",        # Nvidia (US)
]


def load() -> list[str]:
    if os.path.exists(_FILE):
        try:
            with open(_FILE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return list(_DEFAULTS)


def save(tickers: list[str]) -> None:
    with open(_FILE, "w") as f:
        json.dump(tickers, f, indent=2)


def add(ticker: str, tickers: list[str]) -> list[str]:
    t = ticker.strip().upper()
    if t and t not in tickers:
        tickers = tickers + [t]
        save(tickers)
    return tickers


def remove(ticker: str, tickers: list[str]) -> list[str]:
    tickers = [t for t in tickers if t != ticker]
    save(tickers)
    return tickers
