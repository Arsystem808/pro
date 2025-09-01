# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd

# --- Источник данных ---
from capintel.providers.polygon_client import daily_bars, intraday_bars

# --- Правил-основанная логика (твоя стратегия) ---
from capintel.strategy.my_strategy import generate_signal_core as rb_generate

# --- ML-надстройка опционально ---
try:
    from capintel.strategy.my_strategy_ml import generate_signal_core as ml_generate
except Exception:
    ml_generate = None  # ML может отсутствовать — работаем по правилам


def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Пустые бары: провайдер вернул пустой DataFrame")
    rename = {
        "open": "o", "high": "h", "low": "l", "close": "c",
        "Open": "o", "High": "h", "Low": "l", "Close": "c",
        "O": "o", "H": "h", "L": "l", "C": "c",
    }
    df = df.rename(columns=rename)
    need = ["o", "h", "l", "c"]
    if not set(need).issubset(df.columns):
        raise ValueError(f"Нет OHLC-колонок, есть: {list(df.columns)}")
    for k in need:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=need)
    if df.empty:
        raise ValueError("После очистки бары стали пустыми")
    return df


def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    horizon = horizon.lower().strip()
    if horizon == "intraday":
        return intraday_bars(asset_class, ticker)
    return daily_bars(asset_class, ticker)


def _merge(a: Dict[str, Any], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(b, dict):
        return a
    out = {**a, **b}
    return out


def build_signal(
    ticker: str,
    asset_class: str,
    horizon: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    bars = _standardize_ohlc(_fetch_bars(asset_class, ticker, horizon))
    last_px = float(price) if price is not None else float(bars["c"].iloc[-1])

    # 1) База (rules)
    base = rb_generate(ticker, asset_class, horizon, last_px, bars)
    spec: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}

    # 2) ML-надстройка (если есть)
    if ml_generate is not None:
        try:
            ml = ml_generate(ticker, asset_class, horizon, last_px, bars)
            spec = _merge(spec, ml)
            spec.setdefault("ml", {})["on"] = True
        except Exception as e:
            note = spec.get("ml_note", "")
            spec["ml_note"] = (note + f" [ML OFF] Ошибка ML: {e}").strip()
            spec.setdefault("ml", {})["on"] = False
    else:
        spec.setdefault("ml", {})["on"] = False
        spec.setdefault("ml_note", "[ML OFF] Модель не найдена — используется rule-based логика.")

    # Минимальный набор полей
    spec.setdefault("action", "WAIT")
    spec.setdefault("confidence", 0.54)
    spec.setdefault("position_size_pct_nav", 0.0)
    spec.setdefault("alternatives", [])

    # Вход/цели/стоп прячем, если решение — WAIT
    if str(spec["action"]).upper() == "WAIT":
        spec["entry"] = None
        spec["take_profit"] = []
        spec["stop"] = None
        spec["position_size_pct_nav"] = 0.0
    else:
        # Страхуем числовые поля
        spec.setdefault("entry", last_px)
        spec.setdefault("take_profit", [])
        spec.setdefault("stop", last_px)

    return spec
