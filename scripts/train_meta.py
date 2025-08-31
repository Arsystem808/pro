# -*- coding: utf-8 -*-
"""
Обучение meta-классификатора (LightGBM) поверх базовых сигналов.
Сохраняет модель в models/meta.pkl
"""
from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from capintel.providers.polygon_client import get_agg_bars
from capintel.ml.features import rsi_wilder, atr_wilder, macd_hist, heikin_ashi, fibo_pivots
from capintel.ml.events import make_base_events, triple_barrier_label

ASSET_CLASS = os.getenv("ASSET_CLASS", "crypto")
HORIZON     = os.getenv("HORIZON", "swing")
TICKERS     = [t.strip() for t in os.getenv("TICKERS", "BTCUSDT,ETHUSDT").split(",") if t.strip()]
DAYS_BACK   = int(os.getenv("DAYS_BACK", "360"))
OUT_PATH    = Path(os.getenv("OUT_MODEL", "models/meta.pkl"))

PARAMS = {
    "intraday": dict(ts=("minute",5), tol=0.0065, ha=4, macd=4,  atr_tp=0.8, atr_sl=0.8, horizon_bars=48),
    "swing":    dict(ts=("hour",1),   tol=0.0090, ha=5, macd=6,  atr_tp=1.1, atr_sl=1.0, horizon_bars=72),
    "position": dict(ts=("day",1),    tol=0.0120,ha=6, macd=8,  atr_tp=1.5, atr_sl=1.2, horizon_bars=90),
}[HORIZON]

def resample_daily(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x["dt"] = pd.to_datetime(x["t"], unit="s", utc=True)
    x = x.set_index("dt")[["o","h","l","c","v"]].astype(float)
    d = pd.DataFrame({
        "o": x["o"].resample("1D").first(),
        "h": x["h"].resample("1D").max(),
        "l": x["l"].resample("1D").min(),
        "c": x["c"].resample("1D").last(),
        "v": x["v"].resample("1D").sum(),
    }).dropna()
    return d

def prev_period_hlc(daily: pd.DataFrame, horizon: str):
    if horizon=="intraday": freq="W-MON"
    elif horizon=="swing":  freq="M"
    else: freq="Y"
    agg = daily.groupby(pd.Grouper(freq=freq, label="right")).agg({"h":"max","l":"min","c":"last"}).dropna()
    if len(agg) < 2:
        tail = daily.tail(22 if horizon!="position" else 252)
        return float(tail["h"].max()), float(tail["l"].min()), float(tail["c"].iloc[-1])
    return float(agg.iloc[-2]["h"]), float(agg.iloc[-2]["l"]), float(agg.iloc[-2]["c"])

def build_dataset() -> pd.DataFrame:
    rows = []
    for tkr in TICKERS:
        bars = get_agg_bars(ASSET_CLASS, tkr, HORIZON, limit=DAYS_BACK*4)
        if bars.empty: 
            continue
        x = bars.copy()
        x["dt"] = pd.to_datetime(x["t"], unit="s", utc=True)
        x = x.set_index("dt")[["o","h","l","c","v"]].astype(float).dropna()

        daily = resample_daily(bars)
        H,L,C = prev_period_hlc(daily, HORIZON)
        piv   = fibo_pivots(H,L,C)

        ev = make_base_events(x, piv, PARAMS["tol"], PARAMS["ha"], PARAMS["macd"])
        if ev.empty:
            continue

        ha_o, ha_c = heikin_ashi(x); ha_delta = ha_c - ha_o
        hist = macd_hist(x["c"]); rsi = rsi_wilder(x["c"], 14)
        atr  = atr_wilder(x,14)

        for _, e in ev.iterrows():
            t0 = e["time"]; side = e["side"]
            if t0 not in x.index: 
                continue
            i = x.index.get_loc(t0)
            if i < 30: 
                continue

            px = float(x["c"].iloc[i])
            feats = dict(
                ticker=tkr, time=t0, side=1 if side=="BUY" else 0,
                dist_R2=(px/piv["R2"]-1.0), dist_R3=(px/piv["R3"]-1.0),
                dist_S2=(px/piv["S2"]-1.0), dist_S3=(px/piv["S3"]-1.0),
                rsi=float(rsi.iloc[i]), atr=float(atr.iloc[i]), macd_hist=float(hist.iloc[i]),
                ha_green=int((ha_delta.iloc[:i+1]>0)[::-1].cumprod().sum()),
                ha_red=int((ha_delta.iloc[:i+1]<0)[::-1].cumprod().sum()),
            )
            y = triple_barrier_label(x, t0, side,
                                     PARAMS["atr_tp"], PARAMS["atr_sl"], PARAMS["horizon_bars"])
            if y < 0:
                continue
            feats["y"] = int(y)
            rows.append(feats)
    return pd.DataFrame(rows)

def main():
    df = build_dataset()
    if df.empty:
        raise SystemExit("Пустой датасет. Увеличьте DAYS_BACK/список TICKERS.")
    df["group"] = df["time"].dt.isocalendar().week.astype(int)

    X_cols = ["side","dist_R2","dist_R3","dist_S2","dist_S3","rsi","atr","macd_hist","ha_green","ha_red"]
    X = df[X_cols].values; y = df["y"].values; groups = df["group"].values

    cv = GroupKFold(n_splits=5)
    oof = np.zeros(len(df))
    models = []
    for k,(tr, va) in enumerate(cv.split(X, y, groups)):
        clf = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
            min_child_samples=30, random_state=42+k
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:,1]
        oof[va] = p
        models.append(clf)
        print(f"Fold {k}: AUC={roc_auc_score(y[va], p):.3f}")
    print(f"OOF AUC: {roc_auc_score(y, oof):.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": X_cols}, OUT_PATH)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()

