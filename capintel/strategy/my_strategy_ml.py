from __future__ import annotations
from typing import Dict, Any
import os, joblib, numpy as np, pandas as pd
from capintel.ml.features import rsi_wilder, atr_wilder, macd_hist, heikin_ashi, fibo_pivots
from capintel.strategy.my_strategy import generate_signal_core as base_strategy
MODEL=None; FEATURES=None
def _load_model():
    global MODEL,FEATURES
    if MODEL is not None: return
    path=os.getenv("META_MODEL_PATH","models/meta.pkl")
    if not os.path.exists(path): MODEL,FEATURES=None,None; return
    d=joblib.load(path); MODEL=d.get("models"); FEATURES=d.get("features")
def _predict_proba(x_row: np.ndarray) -> float:
    ps=[m.predict_proba(x_row.reshape(1,-1))[:,1][0] for m in MODEL]; return float(np.mean(ps))
def generate_signal_core(ticker: str, asset_class: str, horizon: str, last_price: float, bars: pd.DataFrame | None = None) -> Dict[str, Any]:
    _load_model()
    if MODEL is None:
        spec=base_strategy(ticker,asset_class,horizon,last_price,bars)
        spec["narrative_ru"]="[ML OFF] Модель не найдена — используется rule-based. "+spec.get("narrative_ru","")
        return spec
    if bars is None or len(bars)<80:
        from capintel.providers.polygon_client import get_agg_bars
        bars=get_agg_bars(asset_class,ticker,horizon,limit=400)
    b=bars.copy(); 
    if "dt" not in b.columns: b["dt"]=pd.to_datetime(b["t"], unit="s", utc=True)
    b=b.set_index("dt")[["o","h","l","c","v"]].astype(float).dropna()
    daily=pd.DataFrame({"o":b["o"].resample("1D").first(),"h":b["h"].resample("1D").max(),"l":b["l"].resample("1D").min(),"c":b["c"].resample("1D").last(),"v":b["v"].resample("1D").sum()}).dropna()
    tail=daily.tail(22 if horizon!='position' else 252); piv=fibo_pivots(float(tail['h'].max()), float(tail['l'].min()), float(tail['c'].iloc[-1]))
    ha_o,ha_c=heikin_ashi(b); ha_delta=ha_c-ha_o; hist=macd_hist(b["c"]); rsi=rsi_wilder(b["c"],14); atr=atr_wilder(b,14)
    px=float(last_price); side_buy=1 if px<=piv["S2"] or px<=piv["S3"] else 0; side_short=1 if px>=piv["R2"] or px>=piv["R3"] else 0
    if not side_buy and not side_short:
        entry=px; a=float(atr.iloc[-1] or (px*0.006))
        return dict(action="WAIT", entry=entry, take_profit=[entry+0.6*a, entry+1.1*a], stop=entry-0.8*a,
                    confidence=0.54, narrative_ru="ИИ: нет краевых условий (R2/R3,S2/S3) — ждём сетап.",
                    alt=dict(if_condition="подход к R2/R3 или S2/S3 с признаками остановки", action="WAIT", entry=entry,
                             take_profit=[entry+0.6*a, entry+1.1*a], stop=entry-0.8*a))
    feats={"side":1 if side_buy else 0,"dist_R2":(px/piv["R2"]-1.0),"dist_R3":(px/piv["R3"]-1.0),
           "dist_S2":(px/piv["S2"]-1.0),"dist_S3":(px/piv["S3"]-1.0),"rsi":float(rsi.iloc[-1]),"atr":float(atr.iloc[-1]),
           "macd_hist":float(hist.iloc[-1]),"ha_green":int((ha_delta>0).tail(100)[::-1].cumprod().sum()),"ha_red":int((ha_delta<0).tail(100)[::-1].cumprod().sum())}
    x=np.array([feats[k] for k in FEATURES], dtype=float); p=_predict_proba(x)
    action="BUY" if feats["side"]==1 else "SHORT"; conf_boost=0.03 if (px<=piv["S3"] and action=="BUY") or (px>=piv["R3"] and action=="SHORT") else 0.0
    confidence=float(max(0.52, min(0.90, 0.5 + 0.8*abs(p-0.5) + conf_boost)))
    entry=px; a=float(atr.iloc[-1] or (px*0.006))
    if action=="BUY": tp1,tp2,stop=entry+0.6*a, entry+1.2*a, entry-0.8*a
    else:             tp1,tp2,stop=entry-0.6*a, entry-1.2*a, entry+0.8*a
    return dict(action=action, entry=entry, take_profit=[float(tp1), float(tp2)], stop=float(stop),
                confidence=confidence, narrative_ru=f"ИИ meta: p(success)={p:.2f}. Пивоты(Fibo): P={piv['P']:.4f}, R2={piv['R2']:.4f}, R3={piv['R3']:.4f}, S2={piv['S2']:.4f}, S3={piv['S3']:.4f}.",
                alt=dict(if_condition="подтверждение импульса/отказ от уровня", action=action, entry=entry,
                         take_profit=[float(tp1), float(tp2)], stop=float(stop)))
