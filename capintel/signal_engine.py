# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import Dict, Any
import numpy as np
import pandas as pd

from capintel.providers.polygon_client import daily_bars, intraday_bars
from capintel.strategy.my_strategy import generate_signal_core as rb_generate
from capintel.strategy.my_strategy_ml import prob_success
from capintel.strategy.narrator import trader_tone_narrative_ru

def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Пустые бары: провайдер вернул пустой DataFrame")
    cols = {c.lower(): c for c in df.columns}
    # допустим источники выдают и в верхнем, и в нижнем
    for need in ["o","h","l","c"]:
        if need not in cols and need.upper() in df.columns:
            cols[need] = need.upper()
    # переименуем к нижнему регистру
    out = df.rename(columns={v: k for k, v in cols.items() if k in ["o","h","l","c","v","t"]})
    keep = [c for c in ["t","o","h","l","c","v"] if c in out.columns]
    out = out[keep].copy()
    # индекс-время
    if "t" in out.columns:
        out["t"] = pd.to_datetime(out["t"], unit="ms", errors="ignore", utc=True)
        out = out.set_index("t")
    out = out.dropna()
    if out.empty:
        raise ValueError("Пустые бары после очистки")
    return out

def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    if horizon == "intraday":
        return intraday_bars(asset_class, ticker, interval="5m", limit=520)
    return daily_bars(asset_class, ticker, limit=520)

def build_signal(ticker: str, asset_class: str, horizon: str, price: float | None = None) -> Dict[str, Any]:
    bars_raw = _fetch_bars(asset_class, ticker, horizon)
    bars = _standardize_ohlc(bars_raw)

    # Базовый (rule-based)
    last_px = float(price) if price is not None else float(bars["c"].iloc[-1])
    sig: Dict[str, Any] = rb_generate(asset_class, ticker, horizon, last_px, bars.reset_index().rename(columns={"t":"ts"}))

    # Meta-labeling (если есть модель)
    ml_text = "[ML OFF] Модель не найдена — используется rule-based логика."
    try:
        p_succ = prob_success(last_px, bars)  # None, если модели нет
        if p_succ is not None:
            sig["confidence"] = int(round(100 * float(p_succ)))
            sig["score"] = float(np.interp(p_succ, [0.0, 1.0], [-2.0, 2.0]))
            ml_text = "[ML ON] Вероятность успеха p_succ≈{:.2f}".format(float(p_succ))
    except Exception:
        pass
    sig["ml_text"] = ml_text

    # Скрываем числа на WAIT
    if sig.get("action") == "WAIT":
        sig.update({"entry": None, "tp1": None, "tp2": None, "stop": None, "size": 0.0})

    # Тональность/комментарии
    sig["short_text"], sig["alt_text"] = trader_tone_narrative_ru(sig, bars)

    # score на шкалу по умолчанию
    sig.setdefault("score", 0.0)
    sig.setdefault("confidence", 54)

    return sig
