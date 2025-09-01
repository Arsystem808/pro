# capintel/signal_engine.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import uuid
import math
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

# ---- УСТОЙЧИВЫЙ ИМПОРТ ПРОВАЙДЕРА POLYGON ----
# Поддерживаем разные имена функций в polygon_client.py
daily_bars = intraday_bars = None
polygon_last_price = None
try:
    from capintel.providers.polygon_client import daily_bars as _db, intraday_bars as _ib, last_price as _lp
    daily_bars, intraday_bars, polygon_last_price = _db, _ib, _lp
except Exception:
    try:
        from capintel.providers.polygon_client import get_daily_bars as _db, get_intraday_bars as _ib, get_last_price as _lp
        daily_bars, intraday_bars, polygon_last_price = _db, _ib, _lp
    except Exception:
        # провайдер не найден — дальше будет синтетика/фолбэк
        pass

# ---- Стратегия (rules + ML-надстройка) ----
from capintel.strategy.my_strategy import generate_signal_core as rb_generate
from capintel.strategy.my_strategy_ml import generate_signal_core as ml_generate, prob_success
from capintel.ml.featurizer import make_feature_row

# ---- Наратив трейдерского тона ----
from capintel.narrator import trader_tone_narrative_ru


# -----------------------------
# НАСТРОЙКИ
# -----------------------------

_TTL = {
    "intraday": timedelta(hours=8),
    "swing": timedelta(days=3),
    "position": timedelta(days=7),
}

_DEF_POS_SIZE = {
    "intraday": 0.6,   # % NAV
    "swing":    0.9,
    "position": 1.2,
}

_ALLOWED_ASSET = {"equity", "crypto"}
_ALLOWED_HOR   = {"intraday", "swing", "position"}


# -----------------------------
# УТИЛИТЫ
# -----------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _fmt(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return round(xf, 4)
    except Exception:
        return None

def _unique_id(ticker: str, horizon: str) -> str:
    ts = _utcnow().strftime("%Y%m%d%H%M%S")
    return f"{ticker.lower()}-{ts}-{horizon}"

def _load_meta_model() -> Optional[Any]:
    # /<root>/models/meta.pkl (root = родитель папки capintel)
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "models", "meta.pkl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def _safe_last_price(ticker: str, asset_class: str, user_price: Optional[float]) -> Optional[float]:
    # приоритет ручного ввода
    if user_price is not None:
        try:
            return float(user_price)
        except Exception:
            pass
    # провайдер
    if callable(polygon_last_price):
        try:
            px = polygon_last_price(asset_class, ticker)
            return float(px) if px is not None else None
        except Exception:
            return None
    return None

def _synthetic_bars(last_px: float, periods: int = 240, freq: str = "T") -> pd.DataFrame:
    """
    Синтетические OHLC вокруг last_px (квалифицированный фолбэк, чтобы UI не падал).
    """
    idx = pd.date_range(end=_utcnow(), periods=periods, freq=freq)
    noise = np.random.normal(scale=max(1e-6, last_px * 0.0006), size=periods)
    c = np.clip(last_px + noise.cumsum(), a_min=1e-6, a_max=None)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + np.abs(noise) * 0.25
    l = np.minimum(o, c) - np.abs(noise) * 0.25
    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=pd.to_datetime(idx))
    return df

def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    """
    Стягиваем бары от провайдера. Если не вышло — синтетика на основе текущей цены.
    Обязательно возвращаем DatetimeIndex и колонки o,h,l,c.
    """
    df: Optional[pd.DataFrame] = None

    # 1) попытка реальных данных
    try:
        if horizon == "intraday" and callable(intraday_bars):
            df = intraday_bars(asset_class, ticker, days=3)
        elif callable(daily_bars):
            df = daily_bars(asset_class, ticker, lookback=520)
    except Exception:
        df = None

    # 2) нормализация датаиндекса/колонок
    if df is not None and len(df) > 0:
        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            if "t" in df.columns:
                df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
                df = df.set_index("t").sort_index()
            else:
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df.sort_index()
        # привести названия к ohlc
        upper = {c.upper(): c for c in df.columns}
        for need in ("o","h","l","c"):
            if need not in df.columns:
                if need.upper() in upper:
                    df[need] = df[upper[need.upper()]]
        keep = [c for c in ("o","h","l","c") if c in df.columns]
        df = df[keep].dropna()

    # 3) фолбэк: синтетика
    if df is None or len(df) == 0:
        lp = _safe_last_price(ticker, asset_class, None) or 1.0
        periods = 300 if horizon == "intraday" else 120
        freq = "T" if horizon == "intraday" else "D"
        df = _synthetic_bars(lp, periods=periods, freq=freq)

    return df


def _position_size(conf: float, horizon: str) -> float:
    base = _DEF_POS_SIZE.get(horizon, 0.8)
    k = 0.6 + 1.2 * max(0.0, min(1.0, (conf - 0.4) / 0.4))  # 40–80% → ~0.6x–1.2x
    return round(base * k, 2)

def _dedupe_alternative(base: Dict[str, Any], alt: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """
    Убираем бессмысленную альтернативу (полностью совпадает с базовым сценарием).
    """
    if not alt:
        return []
    same_action = alt.get("action") == base.get("action")
    same_entry  = _fmt(alt.get("entry")) == _fmt(base.get("entry"))
    same_stop   = _fmt(alt.get("stop"))  == _fmt(base.get("stop"))
    same_tp     = list(map(_fmt, alt.get("take_profit", []))) == list(map(_fmt, base.get("take_profit", [])))
    return [] if (same_action and same_entry and same_stop and same_tp) else [alt]


# -----------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# -----------------------------

def build_signal(ticker: str, asset_class: str, horizon: str, price: Optional[float]) -> Dict[str, Any]:
    asset_class = (asset_class or "").lower().strip()
    horizon = (horizon or "").lower().strip()
    ticker = (ticker or "").upper().strip()

    if asset_class not in _ALLOWED_ASSET:
        asset_class = "equity"
    if horizon not in _ALLOWED_HOR:
        horizon = "intraday"

    bars = _fetch_bars(asset_class, ticker, horizon)
    last_px = _safe_last_price(ticker, asset_class, price) or float(bars["c"].iloc[-1])

    # Rule-based (ядро)
    try:
        rb = rb_generate(ticker, asset_class, horizon, last_px, bars)
    except Exception:
        rb = {"action": "WAIT", "entry": None, "take_profit": [], "stop": None, "narrative_ru": "", "alternatives": []}

    action = str(rb.get("action", "WAIT")).upper()
    entry  = _fmt(rb.get("entry"))
    tps    = [x for x in map(_fmt, rb.get("take_profit", [])) if x is not None]
    stop   = _fmt(rb.get("stop"))
    alt    = None
    if isinstance(rb.get("alternatives"), list) and rb["alternatives"]:
        alt = rb["alternatives"][0]
    elif isinstance(rb.get("alternatives"), dict):
        alt = rb["alternatives"]

    # WAIT — без уровней
    if action == "WAIT":
        entry, tps, stop = None, [], None

    # ML
    ml = {"on": False}
    model = _load_meta_model()
    p_succ = None
    if model is not None:
        try:
            x = make_feature_row(bars, last_px, horizon)
            p_succ = float(prob_success(model, x))
            ml = {"on": True, "p_succ": round(p_succ, 3)}
        except Exception:
            ml = {"on": False}

    # confidence
    if p_succ is not None:
        confidence = max(0.4, min(0.9, 0.5 + (p_succ - 0.5) * 0.8))
    else:
        confidence = 0.6 if action in {"BUY","SHORT"} else 0.54

    # sanity-правки уровней
    if action == "BUY" and entry is not None:
        if stop is not None and stop >= entry:
            stop = _fmt(entry * 0.991)
        tps = [tp for tp in tps if tp is None or tp > entry]
    if action == "SHORT" and entry is not None:
        if stop is not None and stop <= entry:
            stop = _fmt(entry * 1.009)
        tps = [tp for tp in tps if tp is None or tp < entry]

    # Наратив
    try:
        narrative = trader_tone_narrative_ru(
            ticker=ticker,
            asset_class=asset_class,
            horizon=horizon,
            action=action,
            entry=entry,
            take_profit=tps,
            stop=stop,
            confidence=confidence,
            last_price=last_px,
            pivots=rb.get("pivots"),
            context=rb.get("context"),
            ml=ml,
        )
    except Exception:
        narrative = ("Сигнал неочевиден — ждём подтверждения от уровня и стабилизации импульса."
                     if action == "WAIT" else
                     "Действуем аккуратно и бережём капитал.")

    now = _utcnow()
    out = {
        "id": _unique_id(ticker, horizon),
        "ticker": ticker,
        "asset_class": asset_class,
        "horizon": horizon,
        "action": action,
        "entry": entry,
        "take_profit": tps[:2],
        "stop": stop,
        "confidence": round(confidence, 2),
        "position_size_pct_nav": _position_size(confidence, horizon),
        "created_at": now.isoformat(),
        "expires_at": (now + _TTL[horizon]).isoformat(),
        "narrative_ru": narrative,
        "alternatives": _dedupe_alternative({"action":action,"entry":entry,"take_profit":tps,"stop":stop}, alt),
        "ml": ml,
        "disclaimer": "Не инвестиционный совет. Торговля сопряжена с риском.",
    }
    return out
