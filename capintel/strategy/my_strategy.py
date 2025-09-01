# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["h"].values, df["l"].values, df["c"].shift(1).fillna(df["c"]).values
    tr = np.maximum(h-l, np.maximum(np.abs(h-c), np.abs(l-c)))
    return pd.Series(tr, index=df.index).rolling(n, min_periods=n).mean()

def _pivots_last(df: pd.DataFrame) -> Dict[str,float]:
    # классика: прошлый день
    d = df.iloc[-2]
    p = (d["h"] + d["l"] + d["c"]) / 3.0
    r1 = 2*p - d["l"]; s1 = 2*p - d["h"]
    r2 = p + (d["h"] - d["l"]); s2 = p - (d["h"] - d["l"])
    r3 = d["h"] + 2*(p - d["l"]); s3 = d["l"] - 2*(d["h"] - p)
    return dict(P=p, R1=r1, S1=s1, R2=r2, S2=s2, R3=r3, S3=s3)

def generate_signal_core(asset_class: str, ticker: str, horizon: str, px: float, bars: pd.DataFrame) -> Dict[str, Any]:
    """
    Простейшая логика:
    - если цена ниже P и ниже R1/S1 смещается к S1 — SHORT
    - если выше P и тянется к R1 — LONG
    - иначе WAIT
    """
    df = bars.rename(columns=str.lower).set_index("ts").sort_index()
    df = df[["o","h","l","c"]].astype(float).dropna()
    piv = _pivots_last(df)
    atr = _atr(df).iloc[-1]
    step = max(atr, 0.002 * px)

    action = "WAIT"
    tp1 = tp2 = stop = None
    if px > piv["P"] and px < piv["R1"]:
        action = "LONG"
        entry = px
        tp1, tp2 = piv["R1"], piv["R2"]
        stop = max(piv["P"] - step, df["l"].iloc[-2])
    elif px < piv["P"] and px > piv["S1"]:
        action = "SHORT"
        entry = px
        tp1, tp2 = piv["S1"], piv["S2"]
        stop = min(piv["P"] + step, df["h"].iloc[-2])
    else:
        entry = None

    short_text = "Идея формируется у опорных уровней. Ждём подтверждения и стабилизации импульса." if action=="WAIT" else \
                 (f"{action}: работа от уровня P={piv['P']:.2f} к целям R1/S1.")
    alt_text = ""
    if action != "WAIT":
        alt_text = f"Если отобьёмся от {('верхней' if action=='SHORT' else 'нижней')} кромки — подтвердить {action.lower()} от {px:.2f} → TP1 {tp1:.2f}, TP2 {tp2:.2f}, стоп {stop:.2f}"

    size = round(0.6 if horizon=="intraday" else (0.9 if horizon=="swing" else 1.2), 1)

    return dict(
        action=action,
        entry=entry, tp1=tp1, tp2=tp2, stop=stop,
        confidence=60, size=size,
        short_text=short_text, alt_text=alt_text,
        pivots=piv
    )

