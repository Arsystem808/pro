# -*- coding: utf-8 -*-
"""
Обучение мета-классификатора (успех сделки по rule-based сигналу).
Запуск локально:  python scripts/train_meta.py
После — положи models/meta.pkl в репозиторий.
"""
from __future__ import annotations
import os, pathlib, joblib, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from capintel.strategy import my_strategy as RB
from capintel.ml.featurizer import make_feature_row

warnings.filterwarnings("ignore")

TICKERS = [
    ("equity", "AAPL"),
    ("equity", "MSFT"),
    ("equity", "QQQ"),
    ("crypto", "BTCUSD"),
    ("crypto", "ETHUSD"),
]

HORIZON = "swing"         # на чём обучаемся
FWD_DAYS = 5              # окно проверки исхода (TP1 vs Stop)
OUT_PATH = pathlib.Path("models/meta.pkl")

def _future_outcome(df: pd.DataFrame, i: int, action: str, tp1: float, stop: float, fwd: int) -> int | None:
    """1 если TP1 раньше Stop, 0 если Stop раньше TP1, None если ни то ни то."""
    fut = df.iloc[i+1:i+1+fwd]
    if fut.empty: return None
    if action == "BUY":
        hit_tp = (fut["h"] >= tp1)
        hit_st = (fut["l"] <= stop)
    else:  # SHORT
        hit_tp = (fut["l"] <= tp1)
        hit_st = (fut["h"] >= stop)
    # первая дата срабатывания
    tp_idx = np.where(hit_tp.values)[0]
    st_idx = np.where(hit_st.values)[0]
    if len(tp_idx)==0 and len(st_idx)==0: return None
    if len(tp_idx)==0: return 0
    if len(st_idx)==0: return 1
    return 1 if tp_idx[0] < st_idx[0] else 0

def _daily_bars(asset_class: str, ticker: str) -> pd.DataFrame:
    return RB._daily(asset_class, ticker, 520, bars=None)

def build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for asset_class, ticker in TICKERS:
        df = _daily_bars(asset_class, ticker)
        if df.empty: 
            print(f"[skip] {ticker}: no data")
            continue
        df = df[["o","h","l","c","v"]].dropna().astype(float)
        # идём по дням, генерируем базовый сигнал и исход
        for i in range(60, len(df)-FWD_DAYS-1):
            sub = df.iloc[:i+1].copy()
            px = float(sub["c"].iloc[-1])
            sig = RB.generate_signal_core(ticker, asset_class, HORIZON, px, bars=sub.reset_index().rename(columns={"index":"dt"}))
            act = sig["action"]
            if act not in ("BUY","SHORT"): 
                continue  # обучаемся только на сделках
            tp1, stop = float(sig["take_profit"][0]), float(sig["stop"])

            # контекст для признаков
            p = RB._params(HORIZON)
            H,L,C = RB._prev_hlc(sub, p["period"])
            piv   = RB._fibo(H,L,C)
            ha_o, ha_c = RB._ha(sub)
            ha_delta   = ha_c - ha_o
            hist       = RB._macd_hist(sub["c"])
            atr        = RB._atr(sub,14)
            last_atr   = float(atr.iloc[-1]) if len(atr) else max(px*0.008,1e-6)
            ha_pos, ha_neg = RB._streak(ha_delta, True), RB._streak(ha_delta, False)
            mac_pos, mac_neg = RB._streak(hist, True), RB._streak(hist, False)
            rsi = 50.0  # опционально можно добавить реальный RSI

            Xi = make_feature_row(px, piv, last_atr, ha_pos, ha_neg, mac_pos, mac_neg, rsi)
            outcome = _future_outcome(df, i, act, tp1, stop, FWD_DAYS)
            if outcome is None:
                continue
            X.append(Xi[0]); y.append(int(outcome))
        print(f"[ok] {ticker}: samples={sum(1 for _ in y)} (cum)")

    if not X:
        raise RuntimeError("Нет данных для обучения. Увеличь список TICKERS или период.")
    return np.array(X, dtype=float), np.array(y, dtype=int)

def main():
    X, y = build_dataset()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
    clf.fit(X, y)
    preds = clf.predict(X)
    prob  = clf.predict_proba(X)[:,1]
    print(f"acc={accuracy_score(y,preds):.3f}  auc={roc_auc_score(y,prob):.3f}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_PATH)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
