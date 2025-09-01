# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

# --- Поставщики данных ---
try:
    from capintel.providers.polygon_client import daily_bars, intraday_bars
except Exception as e:
    raise ImportError(f"Не удалось импортировать провайдера данных: {e}")

# --- Стратегии ---
from capintel.strategy.my_strategy import generate_signal_core as rb_generate

# ML-часть опциональна
try:
    from capintel.strategy.my_strategy_ml import generate_signal_core as ml_generate
except Exception:
    ml_generate = None  # ML недоступен — работаем по правилам


# ---------- Вспомогательные ----------
def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим имена/типы колонок к ['o','h','l','c'] и проверяем непустоту."""
    if df is None or len(df) == 0:
        raise ValueError("Пустые бары: провайдер вернул пустой DataFrame")

    rename_map = {
        "open": "o", "high": "h", "low": "l", "close": "c",
        "Open": "o", "High": "h", "Low": "l", "Close": "c",
        "O": "o", "H": "h", "L": "l", "C": "c",
    }
    out = df.rename(columns=rename_map).copy()

    need = ["o", "h", "l", "c"]
    if not set(need).issubset(out.columns):
        raise ValueError(f"Отсутствуют OHLC-колонки. Есть: {list(out.columns)}")

    for k in need:
        out[k] = pd.to_numeric(out[k], errors="coerce")
    out = out.dropna(subset=need)
    if out.empty:
        raise ValueError("После приведения типов бары стали пустыми.")
    return out


def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    asset_class = asset_class.lower().strip()
    horizon = horizon.lower().strip()
    if horizon == "intraday":
        df = intraday_bars(asset_class, ticker)
    else:
        df = daily_bars(asset_class, ticker)
    return df


def _merge_specs(base: Dict[str, Any], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Аккуратно накладываем ML-результат поверх rule-based (если есть)."""
    if not isinstance(update, dict):
        return base
    merged = {**base, **update}  # ML поля перекрывают базовые
    return merged


# ---------- Публичное API ----------
def build_signal(
    ticker: str,
    asset_class: str,
    horizon: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Главная точка входа. Возвращает словарь с полями:
    action, entry, take_profit (list), stop, confidence, position_size_pct_nav, ...
    """
    bars_raw = _fetch_bars(asset_class, ticker, horizon)
    bars = _standardize_ohlc(bars_raw)

    # Последняя цена
    last_px = float(price) if price is not None else float(bars["c"].iloc[-1])

    # 1) Базовый rule-based сигнал
    spec_rb = rb_generate(ticker, asset_class, horizon, last_px, bars)
    spec = dict(spec_rb) if isinstance(spec_rb, dict) else {}

    # 2) Если ML доступен — попробуем усилить/заменить
    if ml_generate is not None:
        try:
            spec_ml = ml_generate(ticker, asset_class, horizon, last_px, bars)
            spec = _merge_specs(spec, spec_ml)
        except Exception as e:
            # ML упал — оставляем rule-based и добавляем заметку
            note = spec.get("ml_note") or ""
            note += f" [ML OFF] Ошибка ML: {e}"
            spec["ml_note"] = note.strip()

    # Страхуем минимальный набор полей
    spec.setdefault("action", "WAIT")
    spec.setdefault("entry", last_px)
    spec.setdefault("take_profit", [])
    spec.setdefault("stop", last_px)
    spec.setdefault("confidence", 0.5)
    spec.setdefault("position_size_pct_nav", 0.0)

    # Служебная пометка о статусе ML, если её ещё нет
    if "ml_note" not in spec:
        spec["ml_note"] = "[ML ON]" if ml_generate is not None else "[ML OFF] Модель не найдена — используется rule-based логика."

    return spec
