# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

FEATURES = [
    "px",
    "dist_P", "dist_R1", "dist_R2", "dist_R3",
    "dist_S1", "dist_S2", "dist_S3",
    "atr_pct",
    "ha_streak_pos", "ha_streak_neg",
    "macd_streak_pos", "macd_streak_neg",
    "rsi01",
    "above_P", "above_R2", "below_S2",
]

def _dist(px: float, lvl: float) -> float:
    if px == 0: return 0.0
    return (px - float(lvl)) / float(px)

def make_feature_row(px: float, piv: dict, last_atr: float,
                     ha_streak_pos: int, ha_streak_neg: int,
                     macd_streak_pos: int, macd_streak_neg: int,
                     rsi: float) -> np.ndarray:
    row = [
        float(px),
        _dist(px, piv["P"]),  _dist(px, piv["R1"]), _dist(px, piv["R2"]), _dist(px, piv["R3"]),
        _dist(px, piv["S1"]), _dist(px, piv["S2"]), _dist(px, piv["S3"]),
        float(last_atr) / float(px) if px else 0.0,
        float(ha_streak_pos), float(ha_streak_neg),
        float(macd_streak_pos), float(macd_streak_neg),
        float(rsi) / 100.0,
        1.0 if px >= piv["P"]  else 0.0,
        1.0 if px >= piv["R2"] else 0.0,
        1.0 if px <= piv["S2"] else 0.0,
    ]
    return np.array(row, dtype=float).reshape(1, -1)
