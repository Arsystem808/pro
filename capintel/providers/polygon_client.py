# capintel/providers/polygon_client.py
from __future__ import annotations
import os
import pandas as pd

API_KEY = os.getenv("POLYGON_API_KEY", "")

def daily_bars(asset_class: str, ticker: str, days: int = 520) -> pd.DataFrame:
    try:
        from polygon import RESTClient
    except Exception:
        return pd.DataFrame()
    if not API_KEY:
        return pd.DataFrame()
    try:
        client = RESTClient(API_KEY)
        res = client.get_aggs(
            ticker=ticker.upper(),
            multiplier=1, timespan="day",
            limit=days, adjusted=True, sort="asc"
        )
        rows = []
        for a in res:
            rows.append({
                "t": pd.to_datetime(a.timestamp, unit="ms", utc=True),
                "o": a.open, "h": a.high, "l": a.low, "c": a.close, "v": a.volume
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("t").sort_index()
        return df
    except Exception:
        return pd.DataFrame()

def intraday_bars(asset_class: str, ticker: str, timespan: str = "5m", lookback_days: int = 5) -> pd.DataFrame:
    try:
        from polygon import RESTClient
    except Exception:
        return pd.DataFrame()
    if not API_KEY:
        return pd.DataFrame()
    try:
        client = RESTClient(API_KEY)
        # 5m → multiplier=5, timespan="minute"
        ts = timespan.lower().strip()
        if ts.endswith("m"):
            multiplier, span = int(ts[:-1]), "minute"
        elif ts.endswith("h"):
            multiplier, span = int(ts[:-1]), "hour"
        else:
            multiplier, span = 5, "minute"
        limit = max(200, lookback_days * (390 // max(1, multiplier)))  # грубая оценка
        res = client.get_aggs(
            ticker=ticker.upper(),
            multiplier=multiplier, timespan=span,
            limit=limit, adjusted=True, sort="asc"
        )
        rows = []
        for a in res:
            rows.append({
                "t": pd.to_datetime(a.timestamp, unit="ms", utc=True),
                "o": a.open, "h": a.high, "l": a.low, "c": a.close, "v": a.volume
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("t").sort_index()
        return df
    except Exception:
        return pd.DataFrame()
