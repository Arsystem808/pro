# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd

_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "meta.pkl")

def _load():
    global _MODEL
    if _MODEL is None and os.path.exists(_MODEL_PATH):
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL

def _basic_feats(px: float, df: pd.DataFrame) -> np.ndarray:
    # мини-набор признаков, чтобы не расходиться с обучением
    c = df["c"].astype(float)
    r = c.pct_change().fillna(0.0)
    vol = r.rolling(20).std().fillna(0.0).iloc[-1]
    mom = (c.iloc[-1] / c.iloc[-20] - 1.0) if len(c) >= 20 else 0.0
    ma_fast = c.rolling(10).mean().iloc[-1]
    ma_slow = c.rolling(50).mean().iloc[-1] if len(c) >= 50 else c.rolling(min(20,len(c))).mean().iloc[-1]
    trend = float(ma_fast > ma_slow)
    dist = float((c.iloc[-1] - ma_fast) / max(1e-6, ma_fast))
    return np.array([[vol, mom, trend, dist, px]])

def prob_success(px: float, bars: pd.DataFrame) -> float | None:
    m = _load()
    if m is None:
        return None
    X = _basic_feats(px, bars)
    # поддержим как predict_proba, так и decision_function
    if hasattr(m, "predict_proba"):
        p = float(m.predict_proba(X)[0, -1])
    else:
        # приведем к 0..1
        z = float(m.decision_function(X)[0])
        p = 1.0 / (1.0 + np.exp(-z))
    return p
