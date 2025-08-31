# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import importlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Callable, List

import pandas as pd

from capintel.narrator import make_narrative


# ----------------------------------------------------------------------
# Загрузка функции стратегии и данных (ядро)
# ----------------------------------------------------------------------

def _import_callable(path: str) -> Callable[..., Dict[str, Any]]:
    """
    Загружает функцию по строке вида 'package.module:function'.
    По умолчанию используем ML-обёртку.
    """
    path = path or "capintel.strategy.my_strategy_ml:generate_signal_core"
    if ":" not in path:
        raise ValueError("STRATEGY_PATH должен быть вида 'package.module:function'")
    mod_name, func_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"{path} не является функцией")
    return fn


def _load_rb_module():
    """Базовое rule-based ядро (пивоты, индикаторы, источник баров)."""
    return importlib.import_module("capintel.strategy.my_strategy")


def _daily(asset_class: str, ticker: str, lookback: int = 520) -> pd.DataFrame:
    """
    Унифицированная точка получения дневных баров из ядра.
    Ядро само ходит к Polygon и делает фолбэк при 429.
    Ожидается DataFrame с колонками o/h/l/c и DatetimeIndex.
    """
    RB = _load_rb_module()
    return RB._daily(asset_class, ticker, lookback, bars=None)


# ----------------------------------------------------------------------
# Пост-обработка сигнала
# ----------------------------------------------------------------------

def _rel_close(a: float, b: float, tol: float = 0.002) -> bool:
    """Относительное сравнение (по умолчанию 0.2%)."""
    if a == 0 or b == 0:
        return abs(a - b) <= tol
    return abs(a - b) / max(abs(a), abs(b)) <= tol


def _drop_duplicate_alt(base: Dict[str, Any], alt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Убираем альтернативу, если она фактически повторяет базовый план
    (entry/stop/TP1 совпадают в пределах допуска).
    """
    if not alt:
        return None
    if base.get("action") != alt.get("action"):
        return alt

    b_entry = float(base.get("entry") or 0.0)
    a_entry = float(alt.get("entry") or 0.0)
    b_stop  = float(base.get("stop") or 0.0)
    a_stop  = float(alt.get("stop") or 0.0)
    b_tp1   = float((base.get("take_profit") or [0.0])[0])
    a_tp1   = float((alt.get("take_profit") or [0.0])[0])

    if _rel_close(b_entry, a_entry) and _rel_close(b_stop, a_stop) and _rel_close(b_tp1, a_tp1):
        return None
    return alt


def _finalize(sig: Dict[str, Any]) -> Dict[str, Any]:
    """
    Аккуратная финализация:
    - удаляем пустые/дублирующие альтернативы;
    - для WAIT скрываем уровни в шапке карточки.
    """
    alts: List[Dict[str, Any]] = sig.get("alternatives", []) or []
    cleaned: List[Dict[str, Any]] = []
    for alt in alts:
        alt_clean = _drop_duplicate_alt(sig, alt)
        if alt_clean:
            cleaned.append(alt_clean)
    sig["alternatives"] = cleaned

    if sig.get("action") == "WAIT":
        sig["entry"] = None
        sig["take_profit"] = []
        sig["stop"] = None

    return sig


# ----------------------------------------------------------------------
# Риск/метаданные
# ----------------------------------------------------------------------

def _position_size_pct_nav(action: str, confidence: float) -> float:
    """
    Берём функцию из capintel.risk, иначе — мягкий дефолт 0.5..1.1% NAV.
    """
    try:
        from capintel.risk import position_size_pct_nav
        return float(position_size_pct_nav(action, confidence))
    except Exception:
        base = 0.5
        add = max(0.0, min(confidence - 0.5, 0.3)) * 2.0  # 0..0.6
        return round(base + add, 2)


def _expiry_for(horizon: str) -> timedelta:
    return {
        "intraday": timedelta(hours=6),
        "swing":    timedelta(days=7),
        "position": timedelta(days=30),
    }.get(horizon, timedelta(days=7))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Публичное API для UI
# ----------------------------------------------------------------------

def build_signal(
    ticker: str,
    asset_class: str,
    horizon: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Главная функция:
      1) тянем дневные бары из ядра;
      2) гарантируем DatetimeIndex (нужен для resample в ядре);
      3) вызываем функцию стратегии (STRATEGY_PATH | ML-обёртка);
      4) добавляем нарратив, чистим альтернативы, дополняем метаданными.
    """
    # 1) бары и last_px
    bars = _daily(asset_class, ticker, lookback=520)
    last_px = float(price) if price is not None else float(bars["c"].iloc[-1])

    # 2) гарантируем DatetimeIndex
    bars2 = bars.copy()
    if not isinstance(bars2.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
        if "dt" in bars2.columns:
            bars2["dt"] = pd.to_datetime(bars2["dt"])
            bars2 = bars2.set_index("dt")
    bars2 = bars2.sort_index()

    # 3) стратегия (по умолчанию ML-обёртка)
    strategy_path = os.getenv("STRATEGY_PATH", "capintel.strategy.my_strategy_ml:generate_signal_core")
    strategy_fn = _import_callable(strategy_path)

    sig: Dict[str, Any] = strategy_fn(
        ticker=ticker,
        asset_class=asset_class,
        horizon=horizon,
        last_price=last_px,
        bars=bars2,  # <-- передаём с DatetimeIndex
    )

    # 4) метаданные
    confidence = float(sig.get("confidence", 0.60))
    pos_size   = _position_size_pct_nav(sig.get("action", "WAIT"), confidence)
    created_at = _now_utc()
    expires_at = created_at + _expiry_for(horizon)

    sig.setdefault("id", f"{ticker}-{created_at.strftime('%Y%m%d%H%M%S')}-{horizon}")
    sig.setdefault("ticker", ticker)
    sig.setdefault("asset_class", asset_class)
    sig.setdefault("horizon", horizon)
    sig["position_size_pct_nav"] = round(pos_size, 2)
    sig["created_at"] = created_at.strftime("%Y-%m-%d %H:%M:%S")
    sig["expires_at"] = expires_at.strftime("%Y-%m-%d %H:%M:%S")

    # 5) нарратив + финализация
    sig["narrative_ru"] = make_narrative(sig)
    sig = _finalize(sig)

    sig["disclaimer"] = "Не инвестиционный совет. Торговля сопряжена с риском."
    return sig
