# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import uuid
import importlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, Callable, List

import pandas as pd

from capintel.narrator import make_narrative

# -----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНОЕ: загрузка функции стратегии и данных
# -----------------------------------------------------------------------------

def _import_callable(path: str) -> Callable[..., Dict[str, Any]]:
    """
    Загружает call-able по строке формата 'pkg.mod:func'.
    По умолчанию берём ML-обёртку, если не указано иное.
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
    """Подгружаем базовое ядро (для получения баров и пивотов)."""
    return importlib.import_module("capintel.strategy.my_strategy")


def _daily(asset_class: str, ticker: str, lookback: int = 520) -> pd.DataFrame:
    """
    Унифицированная точка получения дневных баров из rule-based ядра.
    Ядро само ходит к Polygon и умеет фолбэк при 429.
    """
    RB = _load_rb_module()
    return RB._daily(asset_class, ticker, lookback, bars=None)


# -----------------------------------------------------------------------------
# КОСМЕТИКА/POST-PROCESS
# -----------------------------------------------------------------------------

def _rel_close(a: float, b: float, tol: float = 0.002) -> bool:
    """Относительное сравнение (по умолчанию 0.2%)."""
    if a == 0 or b == 0:
        return abs(a - b) <= tol
    return abs(a - b) / max(abs(a), abs(b)) <= tol


def _drop_duplicate_alt(base: Dict[str, Any], alt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Убираем альтернативу, если она по сути дублирует базовый план."""
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
    Приводим результат к аккуратному виду:
    - убираем пустые альтернативы и дубли;
    - если action=WAIT — скрываем уровни у основного плана.
    """
    alt_list: List[Dict[str, Any]] = sig.get("alternatives", []) or []
    # чистка/удаление дублей
    cleaned: List[Dict[str, Any]] = []
    for alt in alt_list:
        alt_clean = _drop_duplicate_alt(sig, alt)
        if alt_clean:
            cleaned.append(alt_clean)
    sig["alternatives"] = cleaned

    if sig.get("action") == "WAIT":
        sig["entry"] = None
        sig["take_profit"] = []
        sig["stop"] = None

    return sig


# -----------------------------------------------------------------------------
# РИСК / МЕТАДАННЫЕ
# -----------------------------------------------------------------------------

def _position_size_pct_nav(action: str, confidence: float) -> float:
    """
    Пытаемся взять формулу из capintel.risk, иначе — мягкий дефолт.
    """
    try:
        from capintel.risk import position_size_pct_nav
        return float(position_size_pct_nav(action, confidence))
    except Exception:
        # дефолт: мягкая линейка 0.5%..1.2% NAV
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


# -----------------------------------------------------------------------------
# ПУБЛИЧНО: СБОРКА СИГНАЛА ДЛЯ UI
# -----------------------------------------------------------------------------

def build_signal(
    ticker: str,
    asset_class: str,
    horizon: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Главная функция для UI.
    1) Подтягивает бары (нужны ядру/ML для признаков).
    2) Вызывает функцию стратегии (STRATEGY_PATH или ML-обёртка по умолчанию).
    3) Добавляет нарратив, чистит альтернативы/дубли, добавляет метаданные.
    """
    # 1) бары и текущая цена (если не задана руками)
    bars = _daily(asset_class, ticker, lookback=520)
    last_px = float(price) if price is not None else float(bars["c"].iloc[-1])

    # 2) стратегия (из env или дефолт — ML-обёртка)
    strategy_path = os.getenv("STRATEGY_PATH", "capintel.strategy.my_strategy_ml:generate_signal_core")
    strategy_fn = _import_callable(strategy_path)

    sig: Dict[str, Any] = strategy_fn(
        ticker=ticker,
        asset_class=asset_class,
        horizon=horizon,
        last_price=last_px,
        bars=bars.reset_index().rename(columns={"index": "dt"})  # ядро ожидает колонку dt
    )

    # 3) позиционирование/метаданные
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

    # 4) человеко-пояснение и финальная чистка
    sig["narrative_ru"] = make_narrative(sig)
    sig = _finalize(sig)

    # дисклеймер для карточки
    sig["disclaimer"] = "Не инвестиционный совет. Торговля сопряжена с риском."

    return sig
