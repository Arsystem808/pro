# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import importlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Callable, List

import pandas as pd

from capintel.narrator import make_narrative

# ----------------------------------------------------------------------
# Загрузка стратегии и ядра
# ----------------------------------------------------------------------

def _import_callable(path: str) -> Callable[..., Dict[str, Any]]:
    path = path or "capintel.strategy.my_strategy_ml:generate_signal_core"
    if ":" not in path:
        raise ValueError("STRATEGY_PATH должен быть вида 'package.module:function'")
    mod_name, func_name = path.split(":", 1)
    fn = getattr(importlib.import_module(mod_name), func_name)
    if not callable(fn):
        raise TypeError(f"{path} не является функцией")
    return fn


def _rb():
    return importlib.import_module("capintel.strategy.my_strategy")


def _daily(asset_class: str, ticker: str, lookback: int = 520) -> pd.DataFrame:
    """Берём дневные бары из rule-based ядра (там Polygon/фолбэк уже настроен)."""
    return _rb()._daily(asset_class, ticker, lookback, bars=None)


# ----------------------------------------------------------------------
# Утилиты: индексы, фолбэки цены/баров, пост-обработка
# ----------------------------------------------------------------------

def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантируем DatetimeIndex (нужен для resample в ядре)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
        if "dt" in out.columns:
            out["dt"] = pd.to_datetime(out["dt"])
            out = out.set_index("dt")
    return out.sort_index()

def _synthetic_bars(px: float, days: int = 60) -> pd.DataFrame:
    """Плоские бары, если данных нет — чтобы стратегия не падала."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D")
    return pd.DataFrame({"o": px, "h": px, "l": px, "c": px}, index=idx)

def _fetch_last_price(asset_class: str, ticker: str) -> Optional[float]:
    """Пробуем вытащить последнюю цену из Polygon (если есть клиент и ключ)."""
    try:
        from capintel.providers.polygon_client import Client
        cli = Client()
        return float(cli.last_price(asset_class, ticker))
    except Exception:
        return None

def _rel_close(a: float, b: float, tol: float = 0.002) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) <= tol
    return abs(a - b) / max(abs(a), abs(b)) <= tol

def _drop_duplicate_alt(base: Dict[str, Any], alt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not alt:
        return None
    if base.get("action") != alt.get("action"):
        return alt
    b_entry = float(base.get("entry") or 0.0)
    a_entry = float(alt.get("entry") or 0.0)
    b_stop  = float(base.get("stop")  or 0.0)
    a_stop  = float(alt.get("stop")   or 0.0)
    b_tp1   = float((base.get("take_profit") or [0.0])[0])
    a_tp1   = float((alt .get("take_profit") or [0.0])[0])
    if _rel_close(b_entry, a_entry) and _rel_close(b_stop, a_stop) and _rel_close(b_tp1, a_tp1):
        return None
    return alt

def _finalize(sig: Dict[str, Any]) -> Dict[str, Any]:
    alts = [a for a in (sig.get("alternatives") or []) if a]
    cleaned = []
    for a in alts:
        x = _drop_duplicate_alt(sig, a)
        if x:
            cleaned.append(x)
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
    try:
        from capintel.risk import position_size_pct_nav
        return float(position_size_pct_nav(action, confidence))
    except Exception:
        base = 0.5
        add = max(0.0, min(confidence - 0.5, 0.3)) * 2.0  # 0..0.6
        return round(base + add, 2)

def _expiry_for(h: str) -> timedelta:
    return {"intraday": timedelta(hours=6), "swing": timedelta(days=7), "position": timedelta(days=30)}.get(h, timedelta(days=7))

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
    1) Тянем бары; если пусто — вытаскиваем цену из ввода/Polygon и строим синтетические бары.
    2) Гарантируем DatetimeIndex.
    3) Вызываем стратегию (STRATEGY_PATH | ML-обёртка).
    4) Добавляем нарратив/метаданные, чистим альтернативы.
    """
    # 1) Бары + last_px с фолбэками
    bars_raw = _daily(asset_class, ticker, lookback=520)
    bars_ok  = bars_raw is not None and not bars_raw.empty and ("c" in bars_raw.columns)

    if bars_ok:
        last_px = float(price) if price is not None else float(bars_raw["c"].iloc[-1])
        bars = _ensure_dtindex(bars_raw)
    else:
        # нет исторических баров
        if price is not None:
            last_px = float(price)
        else:
            lp = _fetch_last_price(asset_class, ticker)
            if lp is None:
                # последний шанс — попросим ввести цену в UI (через исключение, которое покажет Streamlit)
                raise RuntimeError("Не удалось получить исторические данные и текущую цену. "
                                   "Укажи текущую цену вручную в левой панели.")
            last_px = float(lp)
        bars = _synthetic_bars(last_px)

    # 2) Стратегия
    strategy_path = os.getenv("STRATEGY_PATH", "capintel.strategy.my_strategy_ml:generate_signal_core")
    strategy_fn = _import_callable(strategy_path)

    sig: Dict[str, Any] = strategy_fn(
        ticker=ticker,
        asset_class=asset_class,
        horizon=horizon,
        last_price=last_px,
        bars=bars,
    )

    # 3) Метаданные и нарратив
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

    sig["narrative_ru"] = make_narrative(sig)
    sig = _finalize(sig)
    sig["disclaimer"] = "Не инвестиционный совет. Торговля сопряжена с риском."
    return sig
