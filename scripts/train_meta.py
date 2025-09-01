# -*- coding: utf-8 -*-
"""
Обучение мета-классификатора, который оценивает вероятность успешности
rule-based сделки (TP1 раньше Stop за FWD_DAYS).

Запуск из корня проекта:
    export PYTHONPATH="$(pwd)"
    export POLYGON_API_KEY="pk_ВАШ_КЛЮЧ"
    python scripts/train_meta.py
После обучения модель будет сохранена в models/meta.pkl
"""
from __future__ import annotations
import pathlib, joblib, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# rule-based ядро и фичи
from capintel.strategy import my_strategy as RB
from capintel.ml.featurizer import make_feature_row

warnings.filterwarnings("ignore")

# ---- МИНИМАЛЬНЫЙ набор тикеров, чтобы не упираться в лимиты Polygon ----
TICKERS: List[Tuple[str, str]] = [
    ("equity", "AAPL"),
    ("crypto", "BTCUSD"),
]

HORIZON   = "swing"   # на каком горизонте генерировать сигналы при обучении
FWD_DAYS  = 5         # окно, в котором считаем исход (TP1 vs Stop)
OUT_PATH  = pathlib.Path("models/meta.pkl")


def _future_outcome(df: pd.DataFrame, i: int, action: str, tp1: float, stop: float, fwd: int) -> int | None:
    """
    Возвращает 1, если TP1 сработал раньше стопа в ближайшие fwd баров,
    0 — если раньше сработал стоп, None — если ни то ни другое.
    """
    fut = df.iloc[i+1:i+1+fwd]
    if fut.empty:
        return None
    if action == "BUY":
        hit_tp = (fut["h"] >= tp1)
        hit_st = (fut["l"] <= stop)
    else:  # SHORT
        hit_tp = (fut["l"] <= tp1)
        hit_st = (fut["h"] >= stop)

    tp_idx = np.where(hit_tp.values)[0]
    st_idx = np.where(hit_st.values)[0]
    if len(tp_idx) == 0 and len(st_idx) == 0:
        return None
    if len(tp_idx) == 0:
        return 0
    if len(st_idx) == 0:
        return 1
    return 1 if tp_idx[0] < st_idx[0] else 0


def _daily_bars(asset_class: str, ticker: str) -> pd.DataFrame:
    """Дневные бары (RB._daily сам делает фолбэк при 429/ошибке)."""
    return RB._daily(asset_class, ticker, 520, bars=None)


def build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for asset_class, ticker in TICKERS:
        df = _daily_bars(asset_class, ticker)
        if df.empty:
            print(f"[skip] {ticker}: no data")
            continue
        df = df[["o", "h", "l", "c", "v"]].dropna().astype(float)

        # идём по дням, генерируем базовый сигнал и исход
        for i in range(60, len(df) - FWD_DAYS - 1):
            sub = df.iloc[:i+1].copy()
            px = float(sub["c"].iloc[-1])

            # bars для ядра в формате с колонкой dt
            bars_for_core = sub.reset_index().rename(columns={"index": "dt"})
            sig = RB.generate_signal_core(ticker, asset_class, HORIZON, px, bars=bars_for_core)
            act = sig.get("action", "WAIT")
            if act not in ("BUY", "SHORT"):
                continue

            tp1 = float(sig["take_profit"][0])
            stop = float(sig["stop"])

            # признаки
            p = RB._params(HORIZON)
            H, L, C = RB._prev_hlc(sub, p["period"])
            piv     = RB._fibo(H, L, C)
            ha_o, ha_c = RB._ha(sub)
            ha_delta   = ha_c - ha_o
            hist       = RB._macd_hist(sub["c"])
            atr        = RB._atr(sub, 14)
            last_atr   = float(atr.iloc[-1]) if len(atr) else max(px * 0.008, 1e-6)
            ha_pos, ha_neg  = RB._streak(ha_delta, True), RB._streak(ha_delta, False)
            mac_pos, mac_neg = RB._streak(hist, True), RB._streak(hist, False)
            rsi = 50.0  # (при желании можно добавить реальный RSI)

            Xi = make_feature_row(px, piv, last_atr, ha_pos, ha_neg, mac_pos, mac_neg, rsi)

            # целевая метка
            out = _future_outcome(df, i, act, tp1, stop, FWD_DAYS)
            if out is None:
                continue

            X.append(Xi[0]); y.append(int(out))

        print(f"[ok] {ticker}: samples={len(y)} (cum)")

    if not X:
        raise RuntimeError("Нет данных для обучения. Попробуйте другой тикер/период или подождите из-за лимитов API.")
    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    X, y = build_dataset()
    # проверка, есть ли оба класса
    if len(np.unique(y)) < 2:
        raise RuntimeError("Нужно минимум по одному примеру каждого класса (успех/неуспех).")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    preds = clf.predict(X)
    prob  = clf.predict_proba(X)[:, 1]
    print(f"acc={accuracy_score(y, preds):.3f}  auc={roc_auc_score(y, prob):.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

