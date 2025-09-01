# capintel/ml/featurizer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


# Фиксированный список признаков (17 шт.) — обучение и инференс используют один и тот же порядок
FEATURES_V1 = [
    "ret_1",            # доходность 1д
    "ret_3",            # доходность 3д
    "ret_5",            # доходность 5д
    "atr14_n",          # ATR(14) / close
    "rsi14_n",          # RSI(14) / 100
    "macd_hist",        # MACD histogram (12,26,9)
    "macd_slope3",      # наклон MACD-hist за 3 бара
    "ha_streak",        # длина текущей серии HA (+вверх/-вниз)
    "vol_std5",         # std(ret, 5)
    "vol_std14",        # std(ret, 14)
    "ema20_dist",       # close/ema20 - 1
    "ema50_dist",       # close/ema50 - 1
    "pos_in_20d",       # позиция цены внутри 20д диапазона [0..1]
    "month_sin",        # сезонность (месяц)
    "month_cos",
    "gap_open",         # (close - open)/close
    "range_n",          # (high - low)/close
]

# -------------------------- тех. индикаторы --------------------------

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def _rsi14(c: pd.Series) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/14, adjust=False).mean()
    rs = np.where(roll_dn == 0, np.nan, roll_up / roll_dn)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=c.index)

def _atr14(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False).mean()

def _macd_hist(c: pd.Series) -> pd.Series:
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    return macd - signal

def _ha_close(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    return (o + h + l + c) / 4.0

def _streak_signed(x: pd.Series) -> int:
    """Длина последней серии по знаку дифференциала x (положит./отриц.), со знаком."""
    d = x.diff().fillna(0)
    sgn = np.sign(d)
    last = sgn.iloc[-1]
    if last == 0:
        return 0
    k = 0
    for v in reversed(sgn.values):
        if v == last:
            k += 1
        else:
            break
    return int(k if last > 0 else -k)

# -------------------------- фичерайзер --------------------------

def make_feature_row(last_price: float, bars: pd.DataFrame, horizon: str) -> Dict[str, Any]:
    """Возвращает dict с ключами из FEATURES_V1. Все NaN → 0.0."""
    df = bars.copy().astype(float)

    # safety
    for col in ("o", "h", "l", "c"):
        if col not in df.columns:
            df[col] = last_price
    df = df.dropna()

    c = df["c"]
    o = df["o"]
    h = df["h"]
    l = df["l"]

    ret = c.pct_change().fillna(0.0)
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    atr14 = _atr14(o, h, l, c)
    rsi14 = _rsi14(c)
    mh = _macd_hist(c)
    mh_slope3 = (mh - mh.shift(3)) / 3.0
    ha_c = _ha_close(o, h, l, c)

    # позиция в 20д диапазоне
    win = c.tail(20)
    pos_in_20d = 0.0
    if len(win) > 1:
        lo, hi = float(win.min()), float(win.max())
        pos_in_20d = 0.0 if hi == lo else float((c.iloc[-1] - lo) / (hi - lo))

    month = bars.index[-1].month if hasattr(bars.index, "month") else pd.Timestamp.utcnow().month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    row = {
        "ret_1":        float((c.iloc[-1] / c.iloc[-2] - 1.0) if len(c) >= 2 else 0.0),
        "ret_3":        float((c.iloc[-1] / c.iloc[-4] - 1.0) if len(c) >= 4 else 0.0),
        "ret_5":        float((c.iloc[-1] / c.iloc[-6] - 1.0) if len(c) >= 6 else 0.0),
        "atr14_n":      float((atr14.iloc[-1] / c.iloc[-1]) if len(atr14) else 0.0),
        "rsi14_n":      float((rsi14.iloc[-1] / 100.0) if len(rsi14) else 0.0),
        "macd_hist":    float(mh.iloc[-1]) if len(mh) else 0.0,
        "macd_slope3":  float(mh_slope3.iloc[-1]) if len(mh_slope3) else 0.0,
        "ha_streak":    float(_streak_signed(ha_c.tail(min(60, len(ha_c)))) if len(ha_c) else 0.0),
        "vol_std5":     float(ret.tail(5).std(ddof=0) if len(ret) >= 5 else 0.0),
        "vol_std14":    float(ret.tail(14).std(ddof=0) if len(ret) >= 14 else 0.0),
        "ema20_dist":   float((c.iloc[-1] / ema20.iloc[-1] - 1.0) if len(ema20) else 0.0),
        "ema50_dist":   float((c.iloc[-1] / ema50.iloc[-1] - 1.0) if len(ema50) else 0.0),
        "pos_in_20d":   float(pos_in_20d),
        "month_sin":    float(month_sin),
        "month_cos":    float(month_cos),
        "gap_open":     float(((c.iloc[-1] - o.iloc[-1]) / c.iloc[-1]) if len(o) else 0.0),
        "range_n":      float(((h.iloc[-1] - l.iloc[-1]) / c.iloc[-1]) if len(h) and len(l) else 0.0),
    }

    # убедимся, что все 17 фич присутствуют
    for k in FEATURES_V1:
        if k not in row or pd.isna(row[k]):
            row[k] = 0.0

    return row
