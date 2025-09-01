# capintel/strategy/my_strategy_ml.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# базовая (rule-based) стратегия
from capintel.strategy.my_strategy import generate_signal_core as rb_generate_signal_core
# единый фичерайзер
from capintel.ml.featurizer import FEATURES_V1, make_feature_row


def _load_model() -> Optional[Any]:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "meta.pkl")
    path = os.path.abspath(path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _prob_success(last_price: float, bars: pd.DataFrame, horizon: str) -> Optional[float]:
    model = _load_model()
    if model is None:
        return None

    row = make_feature_row(last_price, bars, horizon)
    # формируем X в ИМЕННО ТАКОМ порядке, как в FEATURES_V1
    x = np.array([[row[k] for k in FEATURES_V1]], dtype=float)

    # у scikit-learn LogisticRegression всегда метод predict_proba
    try:
        p = float(model.predict_proba(x)[0, 1])
    except AttributeError:
        # на случай, если модель без predict_proba
        p = float(1 / (1 + np.exp(-model.decision_function(x))))
    # клипнем p в [0,1]
    return max(0.0, min(1.0, p))


def generate_signal_core(
    ticker: str,
    asset_class: str,
    horizon: str,
    last_price: float,
    bars: pd.DataFrame,
) -> Dict[str, Any]:
    """
    1) строим базовый (rule-based) сигнал;
    2) если есть модель — считаем p_succ и добавляем в `ml`.
    """
    sig = rb_generate_signal_core(ticker, asset_class, horizon, last_price, bars)

    p = _prob_success(last_price, bars, horizon)
    if p is None:
        sig["ml"] = {"on": False}
        return sig

    sig["ml"] = {"on": True, "p_succ": float(p)}
    # лёгкое смещение confidence по p_succ (опционально)
    try:
        base_conf = float(sig.get("confidence", 0.6))
        sig["confidence"] = max(0.0, min(1.0, 0.5 * base_conf + 0.5 * p))
    except Exception:
        pass

    return sig
