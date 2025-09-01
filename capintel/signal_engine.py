from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from capintel.providers.polygon_client import daily_bars, intraday_bars, last_trade_price
from capintel.strategy.my_strategy import generate_signal_core as RB  # rule-based
from capintel.strategy.my_strategy_ml import prob_success            # meta (если есть)
from capintel.narrator import trader_tone_narrative_ru

MODEL_PATH = Path("models/meta.pkl")

def _std_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Пустые бары: провайдер вернул пустой DataFrame")
    # гарантируем нужные колонки
    need = ["o","h","l","c"]
    for col in need:
        if col not in df.columns:
            raise KeyError(f"Нет колонки {col} в данных провайдера")
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    return df[need].astype(float)

def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    if horizon == "intraday":
        df = intraday_bars(asset_class, ticker, minutes=390, mult=5)
    else:
        # на свинг/позицию для индикаторов всегда нужны дневки
        df = daily_bars(asset_class, ticker, lookback=520)
    return _std_ohlc(df)

def _load_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 5_000:  # 1 КБ — точно заглушка
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def build_signal(ticker: str, asset_class: str, horizon: str, price: float | None = None) -> dict:
    bars = _fetch_bars(asset_class, ticker, horizon)
    last_px = float(price) if price is not None else float(bars["c"].iloc[-1])

    # базовый (rule-based) сигнал по твоей стратегии
    spec = RB(ticker, asset_class, horizon, last_px, bars)

    # meta-labeling (если модель реально есть)
    model = _load_model()
    if model is not None:
        p_succ = prob_success(model, last_px, bars)
        spec["ml"] = {"on": True, "p_succ": float(p_succ)}
        # подстраиваем уверенность/размер
        spec["confidence"] = max(0.05, min(0.95, 0.5 + (p_succ - 0.5)*0.9))
        base = spec.get("position_size_pct_nav", 0.8)
        k = 0.5 + p_succ  # 0.5..1.5
        spec["position_size_pct_nav"] = round(base * k, 2)
    else:
        spec["ml"] = {"on": False}
        spec.setdefault("confidence", 0.54)

    # человеко-подобный комментарий
    spec["narrative_ru"] = trader_tone_narrative_ru(spec, bars)

    # тех. примечание
    spec["tech_note_ru"] = "[ML ON]" if spec["ml"]["on"] else "[ML OFF] Модель не найдена — используется rule-based логика."

    return spec
