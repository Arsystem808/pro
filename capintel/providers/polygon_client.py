from __future__ import annotations
import os
import time
import math
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY","")

_BASE = "https://api.polygon.io"

def _get(url: str, params: dict) -> dict:
    if not POLYGON_API_KEY:
        raise ValueError("Не задан POLYGON_API_KEY")
    params = {**params, "apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _df_from_aggs(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["o","h","l","c","t"])
    df = pd.DataFrame(rows)
    # polygon поля: o,h,l,c,v,t (t — epoch ms)
    if "t" in df.columns:
        idx = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.assign(t=idx).set_index("t").sort_index()
    df = df.rename(columns={"o":"o","h":"h","l":"l","c":"c"})
    return df[["o","h","l","c"]].astype(float)

def daily_bars(asset_class: str, ticker: str, lookback: int = 520) -> pd.DataFrame:
    """Последние ~lookback дневных баров (equity/crypto)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback*1.8))  # с запасом
    url = f"{_BASE}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)}"
    js = _get(url, {"adjusted": "true", "sort":"asc", "limit": 50000})
    df = _df_from_aggs(js.get("results", []))
    return df

def intraday_bars(asset_class: str, ticker: str, minutes: int = 390, mult: int = 5) -> pd.DataFrame:
    """Минутки/5-минутки последней ТОРГОВОЙ сессии."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)  # берём неделю и выделим последнюю сессию
    url = f"{_BASE}/v2/aggs/ticker/{ticker.upper()}/range/{mult}/minute/{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)}"
    js = _get(url, {"adjusted": "true", "sort":"asc", "limit": 50000})
    df = _df_from_aggs(js.get("results", []))
    if df.empty:
        return df

    # определим последнюю «дату» в часовом поясе Нью-Йорка (для акций)
    tz = "America/New_York"
    try:
        di = df.tz_convert(tz)
    except Exception:
        di = df.tz_localize("UTC").tz_convert(tz)

    last_date = di.index.date.max()
    session = di[di.index.date == last_date]
    if session.empty and asset_class.lower()=="equity":
        # запасной путь: просто берём последние minutes баров
        return df.tail(max(1, minutes//mult))
    return session.tail(max(1, minutes//mult))

def last_trade_price(asset_class: str, ticker: str) -> float | None:
    # сперва попробуем интрадей
    try:
        idf = intraday_bars(asset_class, ticker, minutes=60)
        if not idf.empty:
            return float(idf["c"].iloc[-1])
    except Exception:
        pass
    # иначе из дневных
    ddf = daily_bars(asset_class, ticker, lookback=10)
    if ddf.empty:
        return None
    return float(ddf["c"].iloc[-1])
