from __future__ import annotations

from typing import Any, Dict, Optional
import os
import pandas as pd

# Провайдер данных
from capintel.providers.polygon_client import intraday_bars, daily_bars

# Стратегии (rule-based и, при наличии модели, ML-надстройка)
from capintel.strategy.my_strategy import generate_signal_core as rb_generate_signal_core
try:
    from capintel.strategy.my_strategy_ml import generate_signal_core as ml_generate_signal_core
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

# Нарратив (не обязателен)
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
    Нормализуем имена колонок:
      - основа: 'O','H','L','C'
      - алиасы: 'o','h','l','c'
    """
    if df is None or df.empty:
        raise ValueError("Пустые бары: провайдер вернул пустой DataFrame")

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
        # Попытка поймать варианты с разным регистром
        alt = {c.lower(): c for c in out.columns}
        for k, v in alias.items():
            if v not in out.columns and k in alt:
                out.rename(columns={alt[k]: v}, inplace=True)

    missing = [c for c in need if c not in out.columns]
    if missing:
        raise KeyError(f"Нет обязательных OHLC колонок: {missing}; получено: {list(out.columns)}")

    out[["O", "H", "L", "C"]] = out[["O", "H", "L", "C"]].astype(float)

    # создаём алиасы в нижнем регистре
    out["o"] = out["O"]
    out["h"] = out["H"]
    out["l"] = out["L"]
    out["c"] = out["C"]

    # по возможности — DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
            if out.index.isna().any():
                out = out.reset_index(drop=True)
        except Exception:
            out = out.reset_index(drop=True)

    return out


def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    """
    Загружаем бары из провайдера.
    intraday -> минутки
    swing/position -> дневные, после загрузки сами обрезаем до последних 520 строк
    (без использования аргумента limit, которого нет в вашей реализации daily_bars)
    """
    if horizon == "intraday":
        df = intraday_bars(asset_class, ticker)
    else:
        df = daily_bars(asset_class, ticker)
        if df is not None and not df.empty and len(df) > 520:
            df = df.tail(520)
    return df


def _load_meta_flag() -> bool:
    """Включаем ML, если есть models/meta.pkl и доступен модуль my_strategy_ml."""
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
    1) Тянем бары
    2) Нормализуем OHLC
    3) Rule-based сигнал
    4) Опционально ML-надстройка
    5) Нарратив
    """
    # 1–2. Данные
    bars_raw = _fetch_bars(asset_class, ticker, horizon)
    bars = _standardize_ohlc(bars_raw)

    # Текущая цена
    last_px = float(price) if price is not None else float(bars["C"].iloc[-1])

    # 3. Базовый сигнал
    spec = rb_generate_signal_core(ticker, asset_class, horizon, last_px, bars)

    # 4. ML-надстройка
    if _load_meta_flag():
        try:
            spec = ml_generate_signal_core(ticker, asset_class, horizon, last_px, bars, base_spec=spec)
            spec.setdefault("ml", {})["on"] = True
        except Exception as e:
            spec.setdefault("ml", {})["on"] = False
            spec["ml"]["error"] = str(e)

    # 5. Нарратив (best-effort)
    try:
        note = trader_tone_narrative_ru(
            action=spec.get("action", "WAIT"),
            horizon=horizon,
            ticker=ticker,
            price=last_px,
            pivots=spec.get("pivots", {}),
            context=spec,
        )
        if note:
            spec["narrative_ru"] = note
    except Exception:
        pass

    # Отладочная инфа
    spec["meta"] = spec.get("meta", {})
    spec["meta"]["debug_columns"] = list(bars.columns)
    return spec
