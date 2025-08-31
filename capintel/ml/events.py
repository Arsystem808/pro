from __future__ import annotations
from typing import List, Dict
import numpy as np, pandas as pd
from .features import rsi_wilder, atr_wilder, macd_hist, heikin_ashi, fibo_pivots
def _near(price: float, level: float, tol: float) -> bool:
    if level <= 0: return False
    return abs(price-level)/level <= tol
