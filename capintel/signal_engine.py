from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json
import pandas as pd

# Провайдер данных
from capintel.providers.polygon_client import (
    intraday_bars,
    daily_bars,
)

# Стратегии (rule-based и ML-надстройка)
from capintel.strategy.my_strategy import generate_signal_core as rb_generate_signal_core
try:
    from capintel.strategy.my_strategy_ml import generate_signal_core as ml_generate_signal_core
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

# Нарратив (если есть)
try:
    from capintel.narrator import trader_tone_narrative_ru
except Exception:
    def trader_tone_narrative_ru(*args, **kwargs) -> str:
        return ""


# --------------------------
# Вспомогательные функции
# --------------------------

def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим любые варианты имен к единому стандарту:
      - Основные колонки: 'O','H','L','C'
      - Параллельно создаём алиасы в нижнем регистре: 'o','h','l','c'
    Так стратегии, которые обращаются и к верхнему, и к нижнему регистрам, будут работать без KeyError.
    """
    if df is None or df.empty:
        raise ValueError("Пустые бары: провайдер вернул пустой DataFrame")

    # Приведём имена столбцов
    alias = {
        "o": "O", "open": "O",
        "h": "H", "high": "H",
        "l": "L", "low": "L",
        "c": "C", "close": "C",
    }

    cols_map = {}
    for col in df.columns:
        low = str(col).lower()
        if low in alias:
            cols_map[col] = alias[low]
    out = df.rename(columns=cols_map).copy()

    need = ["O", "H", "L", "C"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        # Попробуем кейсы, когда приходят, например, 'Open','High',...
        alt = {c.lower(): c for c in out.columns}
        for k, v in alias.items():
            if v not in out.columns and k in alt:
                out.rename(columns={alt[k]: v}, inplace=True)
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise KeyError(f"Нет обязательных OHLC колонок: {missing}; получено: {list(out.columns)}")

    # Типы числовые
    out[["O", "H", "L", "C"]] = out[["O", "H", "L", "C"]].astype(float)

    # Создадим алиасы в нижнем регистре, если их нет
    out["o"] = out["O"]
    out["h"] = out["H"]
    out["l"] = out["L"]
    out["c"] = out["C"]

    # Индекс — DatetimeIndex, если можем
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
            if out.index.isna().any():
                # если индекса нормального нет — сбросим
                out = out.reset_index(drop=True)
        except Exception:
            out = out.reset_index(drop=True)

    return out


def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    """
    Получение баров из Polygon в зависимости от горизонта.
    intraday  -> минутки (последние ~2-5 дней, зависит от реализации intraday_bars)
    swing/position -> дневные (до 520 баров)
    """
    if horizon == "intraday":
        df = intraday_bars(asset_class, ticker)
    else:
        # для swing/position работаем на daily
        df = daily_bars(asset_class, ticker, limit=520)
    return df


def _load_meta_flag() -> bool:
    """Включать ли ML-надстройку. Считаем, что активна, если есть models/meta.pkl."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "meta.pkl")
    return os.path.exists(model_path) and _ML_AVAILABLE


# --------------------------
# Публичный интерфейс
# --------------------------

def build_signal(
    ticker: str,
    asset_class: str,
    horizon: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Главная функция для приложения.
    1) Тянем бары
    2) Нормализуем OHLC (и добавляем алиасы)
    3) Запускаем rule-based ядро (всегда)
    4) При наличии модели — прогоняем ML-мета-оценку (без падений при ошибке)
    5) Формируем нарратив
    """
    # 1) Данные
    bars_raw = _fetch_bars(asset_class, ticker, horizon)
    bars = _standardize_ohlc(bars_raw)

    # 2) Цена по умолчанию
    last_px = float(price) if price is not None else float(bars["C"].iloc[-1])

    # 3) Базовый сигнал по правилам
    spec = rb_generate_signal_core(ticker, asset_class, horizon, last_px, bars)

    # 4) ML-надстройка (если есть meta.pkl и сама функция доступна)
    ml_on = _load_meta_flag()
    if ml_on:
        try:
            spec = ml_generate_signal_core(ticker, asset_class, horizon, last_px, bars, base_spec=spec)
            spec.setdefault("ml", {})["on"] = True
        except Exception as e:
            # не валим приложение из-за ML
            spec.setdefault("ml", {})["on"] = False
            spec["ml"]["error"] = str(e)

    # 5) Нарратив (не критично, если не получится — не мешаем UI)
    try:
        note = trader_tone_narrative_ru(
            action=spec.get("action", "WAIT"),
            horizon=horizon,
            ticker=ticker,
            price=last_px,
            pivots=spec.get("pivots", {}),
            context=spec
        )
        if note:
            spec["narrative_ru"] = note
    except Exception:
        pass

    # Служебное
    spec["meta"] = spec.get("meta", {})
    spec["meta"]["debug_columns"] = list(bars.columns)
    return spec
