# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional
import math, os, joblib, pathlib

from capintel.strategy import my_strategy as RB  # ваше rule-based ядро

# Пороги meta-классификатора
P_LOW  = 0.45   # ниже — понижаем приоритет до WAIT
P_HIGH = 0.70   # выше — усиливаем сигнал
P_ALT  = 0.60   # если base=WAIT и p_succ >= P_ALT → можно продвигать условный сценарий

# где лежит модель (совместимо со Streamlit Secrets: META_MODEL_PATH)
MODEL_PATH = pathlib.Path(os.getenv("META_MODEL_PATH", "models/meta.pkl"))

_model = None
def _load_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model

def _prob_success(px: float, piv: Dict[str,float], feats: Dict[str,float]) -> Optional[float]:
    """Минимальные признаки → p_succ от meta-модели. Возвращаем None, если модели нет."""
    m = _load_model()
    if m is None:
        return None
    # простые фичи (ровно те же, что в train_meta.py/make_feature_row)
    X = [[
        px,
        px - piv["P"], px - piv["R1"], px - piv["R2"], px - piv["R3"],
        px - piv["S1"], px - piv["S2"], px - piv["S3"],
        feats.get("atr", 0.0),
        feats.get("ha_pos", 0), feats.get("ha_neg", 0),
        feats.get("mac_pos", 0), feats.get("mac_neg", 0),
        feats.get("rsi", 50.0),
    ]]
    p = float(m.predict_proba(X)[0,1])
    return max(0.0, min(1.0, p))

def _almost_eq(a: float, b: float, tol: float=0.002) -> bool:
    # относительное сравнение (0.2%)
    if a == 0 or b == 0:
        return abs(a-b) <= tol
    return abs(a-b)/max(abs(a),abs(b)) <= tol

def _drop_duplicate_alt(base: Dict[str,Any], alt: Optional[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    if not alt:
        return None
    if base.get("action") != alt.get("action"):
        return alt
    a1 = float(base.get("entry", 0)); b1 = float(alt.get("entry", 0))
    s1 = float(base.get("stop", 0));  s2 = float(alt.get("stop", 0))
    tp1 = float(base.get("take_profit", [0])[0]); tp2 = float(alt.get("take_profit", [0])[0])
    if _almost_eq(a1,b1) and _almost_eq(s1,s2) and _almost_eq(tp1,tp2):
        return None
    return alt

def generate_signal_core(
    ticker: str,
    asset_class: str,
    horizon: str,
    last_price: float,
    bars=None
) -> Dict[str, Any]:
    """
    Оборачиваем rule-based сигнал мета-моделью:
    - p_succ < P_LOW  → WAIT (с альтернативой как условный план).
    - base=WAIT и p_succ >= P_ALT → продвигаем условный сценарий (из RB.alternatives) как base, помечая conditional.
    - p_succ > P_HIGH → усиливаем confidence/size.
    И никогда не показываем одинаковые base и alternative.
    """
    # базовый сигнал из ваших правил
    base = RB.generate_signal_core(ticker, asset_class, horizon, last_price, bars=bars)

    # вытаскиваем вспомогательные величины для признаков
    p = RB._params(horizon)
    H,L,C = RB._prev_hlc(bars[["h","l","c"]] if bars is not None else None, p["period"])
    piv   = RB._fibo(H, L, C)
    ha_o, ha_c = RB._ha(bars[["o","h","l","c"]] if bars is not None else None)
    ha_delta = ha_c - ha_o
    hist = RB._macd_hist((bars["c"] if bars is not None else None))
    atr  = RB._atr(bars[["o","h","l","c"]] if bars is not None else None, 14)
    feats = {
        "atr": float(atr.iloc[-1]) if atr is not None and len(atr) else max(last_price*0.008, 1e-6),
        "ha_pos": RB._streak(ha_delta, True),
        "ha_neg": RB._streak(ha_delta, False),
        "mac_pos": RB._streak(hist, True),
        "mac_neg": RB._streak(hist, False),
        "rsi": 50.0,
    }

    p_succ = _prob_success(last_price, piv, feats)
    base["ml"] = {"on": p_succ is not None, "p_succ": p_succ}

    # 1) если модели нет — просто вернём base + чистую альтернативу (если отличается)
    alt = base.get("alternatives", [])
    alt0 = alt[0] if alt else None

    if p_succ is None:
        base["alternatives"] = [_drop_duplicate_alt(base, alt0)] if alt0 else []
        return base

    # 2) модель есть — корректируем
    act = base.get("action", "WAIT")

    if p_succ < P_LOW:
        # понижаем приоритет до WAIT, уровни скрываем
        base.update({"action": "WAIT", "entry": None, "take_profit": [], "stop": None, "confidence": 0.5})
        # альтернатива: если была и отличается — оставим как условный план
        alt_clean = _drop_duplicate_alt(base, alt0)
        base["alternatives"] = [alt_clean] if alt_clean else []
    elif act == "WAIT" and p_succ >= P_ALT and alt0:
        # продвигаем условный план в базовый (но помечаем, что он conditional)
        promoted = alt0.copy()
        promoted["conditional"] = True
        base.update(promoted)
        base["confidence"] = max(base.get("confidence", 0.6), 0.62)
        base["alternatives"] = []  # чтобы не дублировать
    else:
        # усиливаем/оставляем, но альтернативу чистим от дубляжа
        if p_succ >= P_HIGH:
            base["confidence"] = max(base.get("confidence", 0.6), 0.66)
        base["alternatives"] = [_drop_duplicate_alt(base, alt0)] if alt0 else []

    return base
