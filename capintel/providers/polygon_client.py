# -*- coding: utf-8 -*-
"""
Простой и надёжный клиент Polygon.io под CapIntel.
Даёт:
    - daily_bars(asset_class, ticker, n)
    - intraday_bars(asset_class, ticker, minutes)
    - latest_price(asset_class, ticker)
Возвращает pandas.DataFrame с колонками O/H/L/C и DatetimeIndex (UTC).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional

import httpx
import pandas as pd


POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
BASE = "https://api.polygon.io"


# ---------------------------- helpers ---------------------------- #

def _require_key():
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY не задан — добавь секрет в Streamlit (Secrets).")


def _poly_ticker(asset_class: str, ticker: str) -> str:
    """Нормализуем тикер для аггрегатов Polygon."""
    asset_class = (asset_class or "").lower().strip()
    t = ticker.upper().strip()
    if asset_class == "crypto":
        # Polygon ждёт формат X:BTCUSD (по умолчанию к USD)
        if not t.startswith("X:"):
            if not t.endswith("USD"):
                t = f"{t}USD"
            t = f"X:{t}"
    # equities как есть, например AAPL
    return t


def _aggs(ticker: str, mult: int, timespan: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    """Обёртка над /v2/aggs/ticker/... возвращает список баров (raw)."""
    _require_key()
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{date_from}/{date_to}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }
    with httpx.Client(timeout=20) as r:
        resp = r.get(url, params=params)
        resp.raise_for_status()
        j = resp.json()
    return j.get("results", []) or []


def _to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Превращаем аггрегаты Polygon в DataFrame O/H/L/C с DatetimeIndex (UTC)."""
    if not results:
        return pd.DataFrame(columns=["O", "H", "L", "C"])
    rows = []
    for r in results:
        ts = pd.to_datetime(r.get("t"), unit="ms", utc=True)
        rows.append({
            "t": ts,
            "O": float(r.get("o", float("nan"))),
            "H": float(r.get("h", float("nan"))),
            "L": float(r.get("l", float("nan"))),
            "C": float(r.get("c", float("nan"))),
        })
    df = pd.DataFrame(rows).set_index("t").sort_index()
    return df.dropna(how="any")


# ---------------------------- public API ---------------------------- #

def daily_bars(asset_class: str, ticker: str, n: int = 520) -> pd.DataFrame:
    """
    Дневные бары (до ~2 лет).
    """
    tt = _poly_ticker(asset_class, ticker)
    now = datetime.now(timezone.utc).date()
    date_to = now.strftime("%Y-%m-%d")
    date_from = (now - timedelta(days=max(5, int(n) + 10))).strftime("%Y-%m-%d")
    res = _aggs(tt, 1, "day", date_from, date_to)
    df = _to_df(res)
    # берём только хвост нужной длины
    if not df.empty and n:
        df = df.tail(int(n))
    return df


def intraday_bars(asset_class: str, ticker: str, minutes: int = 390) -> pd.DataFrame:
    """
    Минутные бары. Берём за последние 2 календарных дня, чтобы уложиться в лимиты.
    """
    tt = _poly_ticker(asset_class, ticker)
    now = datetime.now(timezone.utc)
    date_to = now.strftime("%Y-%m-%d")
    date_from = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    # минутные аггрегаты (1 минута)
    res = _aggs(tt, 1, "minute", date_from, date_to)
    df = _to_df(res)
    if not df.empty and minutes:
        df = df.tail(int(minutes))
    return df


def latest_price(asset_class: str, ticker: str) -> Optional[float]:
    """
    Текущая/последняя цена. Берём последнюю из минуток; если пусто — из дневных баров.
    """
    try:
        dfi = intraday_bars(asset_class, ticker, minutes=60)
        if not dfi.empty:
            return float(dfi["C"].iloc[-1])
    except Exception:
        pass
    try:
        dfd = daily_bars(asset_class, ticker, n=5)
        if not dfd.empty:
            return float(dfd["C"].iloc[-1])
    except Exception:
        pass
    return None

 
