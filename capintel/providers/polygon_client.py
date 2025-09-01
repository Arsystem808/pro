# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math, datetime as dt
from typing import Literal
import httpx
import pandas as pd

API_KEY = os.getenv("POLYGON_API_KEY", "")

def _http() -> httpx.Client:
    return httpx.Client(timeout=20.0)

def _dt_ms(x: dt.datetime) -> int:
    return int(x.timestamp() * 1000)

def _to_df(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])
    df = pd.DataFrame(results)
    cols = {k:k for k in ["t","o","h","l","c","v"]}
    # polygon отдает уже в этих ключах
    return df[list(cols.keys())]

def daily_bars(asset_class: str, ticker: str, limit: int = 520) -> pd.DataFrame:
    """
    Дневные бары (источник: Polygon v2/aggs). Поддерживает параметр limit.
    Возвращает DataFrame с колонками t,o,h,l,c,v (t в мс).
    """
    if not API_KEY:
        # без ключа вернём болванку, чтобы интерфейс не падал
        return pd.DataFrame(columns=["t","o","h","l","c","v"])

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=max(365, math.ceil(limit*1.5)))
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = dict(adjusted="true", sort="asc", limit=limit, apiKey=API_KEY)
    with _http() as cli:
        r = cli.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    df = _to_df(data.get("results") or [])
    return df.tail(limit)

def intraday_bars(asset_class: str, ticker: str, interval: Literal["1m","5m","15m","30m"]="5m", limit: int = 520) -> pd.DataFrame:
    """
    Внутридневные бары. interval: 1m/5m/15m/30m. Возвращает t,o,h,l,c,v (t в мс).
    """
    if not API_KEY:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])

    # подберем период «с запасом» под лимит
    minutes = {"1m":1, "5m":5, "15m":15, "30m":30}[interval]
    total_min = minutes * max(100, limit)
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(minutes=total_min)
    mult, timespan = int(interval[:-1]), "minute"

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{timespan}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = dict(adjusted="true", sort="asc", limit=limit, apiKey=API_KEY)
    with _http() as cli:
        r = cli.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    df = _to_df(data.get("results") or [])
    return df.tail(limit)
