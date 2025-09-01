from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Any, Tuple
import httpx
import pandas as pd


POLYGON_API_URL = "https://api.polygon.io"
# Ключ читаем из окружения (как ты и делал)
def _api_key() -> str:
    key = (
        os.getenv("POLYGON_API_KEY")
        or os.getenv("POLYGON_KEY")
        or os.getenv("POLYGONIO_API_KEY")
    )
    if not key:
        raise RuntimeError("POLYGON_API_KEY не задан в окружении")
    return key


def _poly_symbol(asset_class: str, ticker: str) -> str:
    t = ticker.upper().strip()
    if asset_class.lower() == "crypto":
        # Для Polygon крипта — X:BTCUSD
        return t if ":" in t else f"X:{t}"
    return t


def _parse_timespan(s: str) -> Tuple[int, str]:
    s = s.lower().strip()
    if s.endswith("m"):
        return int(s[:-1]), "minute"
    if s.endswith("h"):
        return int(s[:-1]), "hour"
    if s.endswith("d"):
        return int(s[:-1]), "day"
    # по умолчанию 5 минут
    return 5, "minute"


def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = {**params, "apiKey": _api_key()}
    with httpx.Client(timeout=20.0) as client:
        r = client.get(f"{POLYGON_API_URL}{path}", params=params)
        r.raise_for_status()
        return r.json()


def _results_to_df(js: Dict[str, Any]) -> pd.DataFrame:
    res = js.get("results") or []
    df = pd.DataFrame(res)
    if df.empty:
        # Пустой правильный фрейм
        idx = pd.DatetimeIndex([], name="t")
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"], index=idx)

    # гарантируем нужные колонки
    for col in ["t", "o", "h", "l", "c", "v"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["t", "o", "h", "l", "c", "v"]]
    # timestamp в мс -> UTC DatetimeIndex
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").sort_index()

    # типы числовых колонок
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------- Публичные функции провайдера ----------

def daily_bars(
    asset_class: str,
    ticker: str,
    limit: int = 520,
    lookback_days: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Дневные бары. Принимает и limit, и days/lookback_days (alias).
    Возвращает DataFrame с индексом DatetimeIndex(UTC) и колонками o,h,l,c,v
    """
    # alias: days -> lookback_days
    if lookback_days is None and "days" in kwargs and kwargs["days"] is not None:
        lookback_days = int(kwargs["days"])

    now = dt.datetime.utcnow().date()
    if lookback_days is None:
        # если не задано, оценим по limit (с запасом)
        lookback_days = int(max(2 * limit, 520))
    start = now - dt.timedelta(days=lookback_days)

    symbol = _poly_symbol(asset_class, ticker)
    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start:%Y-%m-%d}/{now:%Y-%m-%d}"
    js = _get(path, {"adjusted": "true", "sort": "asc", "limit": limit})
    df = _results_to_df(js)
    return df.tail(limit)


def intraday_bars(
    asset_class: str,
    ticker: str,
    timespan: str = "5m",
    lookback_days: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Интрадей бары. timespan: '1m','5m','15m','1h'...; lookback_days — глубина окна.
    Возвращает DataFrame с индексом DatetimeIndex(UTC) и колонками o,h,l,c,v
    """
    # alias: days -> lookback_days
    if "days" in kwargs and kwargs["days"] is not None:
        lookback_days = int(kwargs["days"])

    mult, unit = _parse_timespan(timespan)
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=int(lookback_days))

    symbol = _poly_symbol(asset_class, ticker)
    path = (
        f"/v2/aggs/ticker/{symbol}/range/{mult}/{unit}/"
        f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    )
    js = _get(path, {"adjusted": "true", "sort": "asc"})
    df = _results_to_df(js)
    return df
def daily_bars(asset_class: str,
               ticker: str,
               limit: int = 520,
               lookback_days: int | None = None,
               **kwargs) -> pd.DataFrame:
    # алиас: days -> lookback_days -> limit
    if lookback_days is None and 'days' in kwargs and kwargs['days'] is not None:
        lookback_days = int(kwargs['days'])
    if lookback_days is not None:
        limit = int(lookback_days)

    # === ТВОЙ текущий запрос к Polygon остаётся здесь ===
    # df = ... (полученный DataFrame от API)
    # df должен содержать хотя бы ['t','o','h','l','c','v'] или уже ['o','h','l','c','v'] + DatetimeIndex

    df = _std_ohlc(df)
    return df.tail(limit)  # на всякий случай ограничим
def intraday_bars(asset_class: str,
                  ticker: str,
                  timespan: str = '5m',
                  lookback_days: int = 5,
                  **kwargs) -> pd.DataFrame:
    # алиасы
    if 'interval' in kwargs and kwargs['interval'] and not timespan:
        timespan = kwargs['interval']
    if 'days' in kwargs and kwargs['days'] is not None:
        lookback_days = int(kwargs['days'])

    # === ТВОЙ текущий запрос к Polygon остаётся здесь ===
    # при формировании запроса используй timespan (например '5m','15m','1h')
    # и lookback_days
    # df = ... (DataFrame от API)

    df = _std_ohlc(df)
    return df
import pandas as pd

def _std_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим бары к виду:
      - DatetimeIndex
      - колонки: ['o','h','l','c','v'] с float/int типами
    Допустимо, что источник возвратил 't' — переносим в индекс.
    """
    if 't' in df.columns:
        df = df.copy()
        # polygon отдаёт ms-таймстемпы или ISO — обе ветки ок
        if pd.api.types.is_numeric_dtype(df['t']):
            df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        else:
            df['t'] = pd.to_datetime(df['t'], utc=True, errors='coerce')
        df = df.set_index('t')

    # гарантируем нужные имена колонок
    rename_map = {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}
    df = df.rename(columns=rename_map)

    need = ['o', 'h', 'l', 'c', 'v']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Provider returned unexpected columns, missing: {missing}")

    df = df[need].astype({'o': float, 'h': float, 'l': float, 'c': float, 'v': float})
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Provider returned bars without DatetimeIndex")
    return df.sort_index()
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
