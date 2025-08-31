# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pathlib, joblib
from typing import Dict, Any
import numpy as np
import pandas as pd

# берём готовое rule-based ядро и утилиты
from capintel.strategy.my_strategy import (
    generate_signal_core as rb_core,
    _params, _daily, _prev_hlc, _fibo, _ha, _streak, _macd_hist, _atr, _near
)
from capintel.ml.featurizer import make_feature_row

# где искать модель
def _model_path() -> str:
    p = os.getenv("META_MODEL_PATH", "models/meta.pkl")
    pth = pathlib.Path(p)
    if not pth.is_absolute():
        pth = pathlib.Path.cwd() / p
    return str(pth)

def _load_model():
    try:
        mp = _model_path()
        if os.path.exists(mp):
            return joblib.load(mp)
    except Exception:
        pass
    return None

_MODEL = _load_model()

def _compose_narr(base: str, prob: float | None) -> str:
    if _MODEL is None:
        return "[ML OFF] Модель не найдена — используется rule-based. " + (base or "")
    if prob is None:
        return "[ML ON] Мета-оценка недоступна. " + (base or "")
    return f"[ML ON] p_succ≈{prob:.2f}. " + (base or "")

def generate_signal_core(
    ticker: str,
    asset_class: str,
    horizon: str,
    last_price: float,
    bars: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """
    ML-надстройка над rule-based сигналом.
    Если модели нет — просто возвращает rule-based (с пометкой ML OFF).
    """
    # 1) базовый сигнал (rule-based)
    base = rb_core(ticker, asset_class, horizon, last_price, bars)

    # без модели — сразу возврат
    if _MODEL is None:
        base["narrative_ru"] = _compose_narr(base.get("narrative_ru", ""), None)
        return base

    # 2) контекст для признаков (делаем так же, как в ядре)
    p = _params(horizon)
    daily = _daily(asset_class, ticker, 520, bars=bars)

    b = daily.copy() if (bars is None or len(bars) < 50) else bars.copy()
    if "dt" in b.columns:
        b = b.set_index(pd.to_datetime(b["dt"], utc=True))
    elif "t" in b.columns and not isinstance(b.index, pd.DatetimeIndex):
        b = b.set_index(pd.to_datetime(b["t"], unit="s", utc=True))
    b = b[["o", "h", "l", "c"]].astype(float).dropna()

    base_df = daily if not daily.empty else b
    H, L, C = _prev_hlc(base_df, p["period"])
    piv = _fibo(H, L, C)

    ha_o, ha_c = _ha(b)
    ha_delta = ha_c - ha_o
    hist = _macd_hist(b["c"])
    rsi = 50.0  # (тонкая настройка RSI — опционально) тут можно добавить собственный расчёт
    atr = _atr(b, 14)
    px = float(last_price)
    last_atr = float(atr.iloc[-1]) if len(atr) else max(px * 0.008, 1e-6)

    ha_pos = _streak(ha_delta, True)
    ha_neg = _streak(ha_delta, False)
    mac_pos = _streak(hist, True)
    mac_neg = _streak(hist, False)

    X = make_feature_row(
        px=px, piv=piv, last_atr=last_atr,
        ha_streak_pos=ha_pos, ha_streak_neg=ha_neg,
        macd_streak_pos=mac_pos, macd_streak_neg=mac_neg,
        rsi=rsi
    )

    # 3) предсказание вероятности успеха базовой сделки
    try:
        proba = float(_MODEL.predict_proba(X)[0, 1])
    except Exception:
        proba = None

    # 4) агрегируем: усиливаем/ослабляем rule-based
    action = base["action"]
    conf = float(base.get("confidence", 0.56))
    alt = base.get("alt", {})
    entry, tp1, tp2, stop = base["entry"], base["take_profit"][0], base["take_profit"][1], base["stop"]

    if proba is not None:
        # BUY/SHORT — понижаем до WAIT если p<0.45; усиливаем если p>0.65
        if action in ("BUY", "SHORT"):
            if proba < 0.45:
                action = "WAIT"
                # уровни в UI при WAIT скрываются — оставим как есть
                conf = 0.54
            elif proba > 0.65:
                conf = min(0.9, 0.55 + 0.4 * proba)
        else:  # WAIT/CLOSE — можно «подтолкнуть» альтернативу, если p высока
            if proba > 0.70 and isinstance(alt, dict) and alt.get("action") in ("BUY", "SHORT"):
                action = alt["action"]
                entry = alt.get("entry", entry)
                tp = alt.get("take_profit", [tp1, tp2])
                tp1, tp2 = tp[0], tp[1]
                stop = alt.get("stop", stop)
                conf = max(conf, 0.60)

    base.update(
        action=action,
        entry=entry,
        take_profit=[tp1, tp2],
        stop=stop,
        confidence=conf,
        narrative_ru=_compose_narr(base.get("narrative_ru", ""), proba),
    )
    return base
