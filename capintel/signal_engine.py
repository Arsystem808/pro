# capintel/signal_engine.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import uuid
import math
import time
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Источник цен (Polygon)
from capintel.providers.polygon_client import daily_bars, intraday_bars, last_price as polygon_last_price

# Правила и ML-надстройка
from capintel.strategy.my_strategy import generate_signal_core as rb_generate  # rule-based
from capintel.strategy.my_strategy_ml import generate_signal_core as ml_generate, prob_success  # ML
from capintel.ml.featurizer import make_feature_row

# Тональность комментариев
from capintel.narrator import trader_tone_narrative_ru


# -----------------------------
# ВСПОМОГАТЕЛЬНЫЕ НАСТРОЙКИ
# -----------------------------

_TTL = {
    "intraday": timedelta(hours=8),
    "swing": timedelta(days=3),
    "position": timedelta(days=7),
}

_DEF_POS_SIZE = {
    "intraday": 0.6,   # % NAV (база) до масштабирования уверенностью
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
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), 4)

def _unique_id(ticker: str, horizon: str) -> str:
    ts = _utcnow().strftime("%Y%m%d%H%M%S")
    return f"{ticker.lower()}-{ts}-{horizon}"

def _load_meta_model() -> Optional[Any]:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "meta.pkl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def _safe_last_price(ticker: str, asset_class: str, user_price: Optional[float]) -> Optional[float]:
    if user_price is not None:
        try:
            return float(user_price)
        except Exception:
            pass
    try:
        px = polygon_last_price(asset_class, ticker)
        return float(px) if px is not None else None
    except Exception:
        return None

def _synthetic_bars(last_px: float, periods: int = 120, freq: str = "T") -> pd.DataFrame:
    """
    Создаёт синтетические бары OHLC вокруг last_px, чтобы не падать при пустых данных.
    """
    idx = pd.date_range(end=_utcnow(), periods=periods, freq=freq)
    # маленький шум вокруг last_px
    noise = np.random.normal(scale=last_px * 0.0005, size=periods)
    close = np.clip(last_px + noise.cumsum(), a_min=0.0001, a_max=None)
    o = np.r_[close[0], close[:-1]]
    h = np.maximum(o, close) + abs(noise) * 0.2
    l = np.minimum(o, close) - abs(noise) * 0.2
    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": close}, index=idx)
    df.index = pd.to_datetime(df.index)
    return df

def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    """
    Пробуем получить реальные бары, иначе возвращаем синтетику (с DatetimeIndex).
    """
    df: Optional[pd.DataFrame] = None
    try:
        if horizon == "intraday":
            df = intraday_bars(asset_class, ticker, days=3)  # 3 дня минуток (или как реализовано в провайдере)
        else:
            df = daily_bars(asset_class, ticker, lookback=520)  # ~2 года дневок
    except Exception:
        df = None

    if df is None or len(df) == 0:
        lp = _safe_last_price(ticker, asset_class, None)
        if lp is None:
            return pd.DataFrame(columns=["o", "h", "l", "c"])
        return _synthetic_bars(lp, periods=240 if horizon == "intraday" else 120, freq="T" if horizon == "intraday" else "D")

    # гарантируем DatetimeIndex
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # ожидаем колонку времени
        if "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
            df = df.set_index("t").sort_index()
        else:
            # худший случай — преобразуем индекс
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.sort_index()

    # нормализуем названия колонок
    cols = {c.lower(): c for c in df.columns}
    for need in ["o", "h", "l", "c"]:
        if need not in df.columns and need.upper() in df.columns:
            df[need] = df[need.upper()]
    # оставляем только ohlc
    keep = [c for c in ["o", "h", "l", "c"] if c in df.columns]
    df = df[keep].dropna()
    return df


def _position_size(conf: float, horizon: str) -> float:
    base = _DEF_POS_SIZE.get(horizon, 0.8)
    # линейное масштабирование уверенностью (40–80% → ~0.6x–1.2x)
    k = 0.6 + 1.2 * max(0.0, min(1.0, (conf - 0.4) / 0.4))
    return round(base * k, 2)


def _dedupe_alternative(base: Dict[str, Any], alt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Убираем бессмысленную альтернативу (полностью совпадает с базовым действием).
    """
    if not alt:
        return None
    same_action = alt.get("action") == base.get("action")
    same_entry  = _fmt(alt.get("entry")) == _fmt(base.get("entry"))
    same_stop   = _fmt(alt.get("stop"))  == _fmt(base.get("stop"))
    same_tp     = list(map(_fmt, alt.get("take_profit", []))) == list(map(_fmt, base.get("take_profit", [])))
    if same_action and same_entry and same_stop and same_tp:
        return None
    return alt


# -----------------------------
# ОСНОВНАЯ ТОЧКА ВХОДА
# -----------------------------

def build_signal(ticker: str, asset_class: str, horizon: str, price: Optional[float]) -> Dict[str, Any]:
    """
    Главная функция: собирает спецификацию сигнала и приводит к единому формату для UI.
    """
    asset_class = (asset_class or "").lower().strip()
    horizon = (horizon or "").lower().strip()
    ticker = (ticker or "").upper().strip()

    if asset_class not in _ALLOWED_ASSET:
        asset_class = "equity"
    if horizon not in _ALLOWED_HOR:
        horizon = "intraday"

    # Данные
    bars = _fetch_bars(asset_class, ticker, horizon)
    last_px = _safe_last_price(ticker, asset_class, price)
    if last_px is None:
        # последнее средство: попытка взять из баров
        try:
            last_px = float(bars["c"].iloc[-1])
        except Exception:
            last_px = None

    # Если данных совсем нет — возвращаем WAIT без уровней
    if last_px is None or len(bars) == 0:
        now = _utcnow()
        return {
            "id": _unique_id(ticker, horizon),
            "ticker": ticker,
            "asset_class": asset_class,
            "horizon": horizon,
            "action": "WAIT",
            "entry": None,
            "take_profit": [],
            "stop": None,
            "confidence": 0.5,
            "position_size_pct_nav": _position_size(0.5, horizon),
            "created_at": now.isoformat(),
            "expires_at": (now + _TTL[horizon]).isoformat(),
            "narrative_ru": "Данных от провайдера нет. Сохраняем капитал и ждём подтверждений.",
            "alternatives": [],
            "ml": {"on": False},
            "disclaimer": "Не инвестиционный совет. Торговля сопряжена с риском.",
        }

    # Выбор: есть ли модель?
    model = _load_meta_model()
    use_ml = model is not None

    # Rule-based ядро (используется и для ML, как «первичный» сценарий)
    try:
        rb_spec = rb_generate(ticker, asset_class, horizon, last_px, bars)
    except Exception:
        rb_spec = {
            "action": "WAIT",
            "entry": None,
            "take_profit": [],
            "stop": None,
            "narrative_ru": "Техническая картина неоднозначна. Ждём реакции на ключевых уровнях.",
            "alternatives": [],
        }

    # Базовые поля из rule-based
    action = str(rb_spec.get("action", "WAIT")).upper()
    entry  = _fmt(rb_spec.get("entry"))
    tps    = [x for x in map(_fmt, rb_spec.get("take_profit", [])) if x is not None]
    stop   = _fmt(rb_spec.get("stop"))
    alt    = rb_spec.get("alternatives", [])
    alt = alt[0] if isinstance(alt, list) and alt else (alt if isinstance(alt, dict) else None)
    alt = _dedupe_alternative(rb_spec, alt)

    # Для WAIT — не показываем уровни
    if action == "WAIT":
        entry, tps, stop = None, [], None

    # ML: оценка вероятности успеха
    ml_info: Dict[str, Any] = {"on": False}
    p_succ: Optional[float] = None

    if use_ml:
        try:
            x = make_feature_row(bars, last_px, horizon)  # shape (1, n_features)
            p_succ = float(prob_success(model, x))  # 0..1
            ml_info = {"on": True, "p_succ": round(p_succ, 3)}
        except Exception:
            ml_info = {"on": False}
            p_succ = None

    # Уверенность: если есть p_succ — используем, иначе — эвристика от действия
    if p_succ is not None:
        confidence = max(0.4, min(0.9, 0.5 + (p_succ - 0.5) * 0.8))
    else:
        confidence = 0.6 if action in {"BUY", "SHORT"} else 0.54

    # sanity-check уровней
    if action == "BUY":
        # стоп ниже входа, цели выше
        if stop is not None and entry is not None and stop >= entry:
            stop = _fmt(entry * (1 - 0.009))
        tps = [tp for tp in tps if entry is None or tp is None or tp > entry]
    elif action == "SHORT":
        # стоп выше входа, цели ниже
        if stop is not None and entry is not None and stop <= entry:
            stop = _fmt(entry * (1 + 0.009))
        tps = [tp for tp in tps if entry is None or tp is None or tp < entry]

    # Наратив (тон «трейдера»)
    try:
        note_ru = trader_tone_narrative_ru(
            ticker=ticker,
            asset_class=asset_class,
            horizon=horizon,
            action=action,
            entry=entry,
            take_profit=tps,
            stop=stop,
            confidence=confidence,
            last_price=last_px,
            pivots=rb_spec.get("pivots", None),
            context=rb_spec.get("context", None),
            ml=ml_info,
        )
    except Exception:
        # запасной текст
        if action == "WAIT":
            note_ru = "Сигнал неочевиден — ждём подтверждения от уровня и стабилизации импульса."
        elif action == "BUY":
            note_ru = "Покупка у опорной зоны. Работаем аккуратно и бережём капитал."
        else:
            note_ru = "Короткая позиция от «крыши»/слабости. Действуем аккуратно."

    now = _utcnow()
    out: Dict[str, Any] = {
        "id": _unique_id(ticker, horizon),
        "ticker": ticker,
        "asset_class": asset_class,
        "horizon": horizon,
        "action": action,
        "entry": entry,
        "take_profit": tps[:2],  # максимум две цели для UI
        "stop": stop,
        "confidence": round(confidence, 2),
        "position_size_pct_nav": _position_size(confidence, horizon),
        "created_at": now.isoformat(),
        "expires_at": (now + _TTL[horizon]).isoformat(),
        "narrative_ru": note_ru,
        "alternatives": ([] if alt is None else [alt]),
        "ml": ml_info,
        "disclaimer": "Не инвестиционный совет. Торговля сопряжена с риском.",
    }
    return out
