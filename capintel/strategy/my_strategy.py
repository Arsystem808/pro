from __future__ import annotations
from typing import Dict, Any, List
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
import httpx
from capintel.providers import polygon_client as poly
def _ema(s: pd.Series, span: int, wilder: bool=False) -> pd.Series:
    alpha=(1.0/span) if wilder else (2.0/(span+1.0)); return s.ewm(alpha=alpha, adjust=False).mean()
def _rsi(c: pd.Series, n: int=14) -> pd.Series:
    d=c.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0); au=_ema(up,n,True); ad=_ema(dn,n,True).replace(0,np.nan); rs=au/ad
    return (100-100/(1+rs)).fillna(50)
def _atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    h,l,c=df["h"].astype(float), df["l"].astype(float), df["c"].astype(float); c1=c.shift(1)
    tr=pd.concat([h-l,(h-c1).abs(),(l-c1).abs()],axis=1).max(axis=1); return _ema(tr,n,True)
def _macd_hist(c: pd.Series) -> pd.Series: m=_ema(c,12)-_ema(c,26); s=_ema(m,9); return m-s
def _ha(df: pd.DataFrame):
    o,h,l,c=[df[k].astype(float).values for k in ("o","h","l","c")]
    ha_c=(o+h+l+c)/4.0; ha_o=np.zeros_like(ha_c); ha_o[0]=(o[0]+c[0])/2.0
    for i in range(1,len(ha_o)): ha_o[i]=(ha_o[i-1]+ha_c[i-1])/2.0
    return pd.Series(ha_o, index=df.index), pd.Series(ha_c, index=df.index)
def _streak(x: pd.Series, pos=True)->int:
    if x.empty: return 0
    arr=np.sign(x.values); want=1 if pos else -1; n=0
    for v in arr[::-1]:
        if (v>0 and want>0) or (v<0 and want<0): n+=1
        elif v==0: break
        else: break
    return n
def _fibo(H,L,C):
    P=(H+L+C)/3.0; d=H-L; return {"P":P,"R1":P+0.382*d,"R2":P+0.618*d,"R3":P+1.0*d,"S1":P-0.382*d,"S2":P-0.618*d,"S3":P-1.0*d}
def _daily(asset_class, ticker, days=520):
    if asset_class=='crypto': b,q=poly._norm_crypto_pair(ticker); tkr=f"X:{b}{q}"
    else: tkr=ticker.upper()
    to=datetime.now(timezone.utc).date(); fr=(to - timedelta(days=days))
    url=f"{poly.BASE}/v2/aggs/ticker/{tkr}/range/1/day/{fr}/{to}?adjusted=true&limit=50000&sort=asc"
    with httpx.Client(timeout=20) as c: r=c.get(url, headers=poly._headers()); r.raise_for_status(); d=r.json()
    rows=(d or {}).get("results") or []
    if not rows: return pd.DataFrame(columns=["o","h","l","c","v"])
    df=pd.DataFrame(rows)[["t","o","h","l","c","v"]].copy()
    if df["t"].max()>1e12: df["t"]=(df["t"]/1000).astype(int)
    df["dt"]=pd.to_datetime(df["t"], unit="s", utc=True); df.set_index("dt", inplace=True)
    return df[["o","h","l","c","v"]]
def _prev_hlc(daily: pd.DataFrame, period: str):
    if period=='W': g=daily.groupby(pd.Grouper(freq='W-MON',label='right'))
    elif period=='M': g=daily.groupby(pd.Grouper(freq='M',label='right'))
    else: g=daily.groupby(pd.Grouper(freq='Y',label='right'))
    agg=g.agg({'h':'max','l':'min','c':'last'}).dropna()
    if len(agg)<2:
        tail=daily.tail({'W':5,'M':22,'Y':252}[period]); return float(tail['h'].max()), float(tail['l'].min()), float(tail['c'].iloc[-1])
    return float(agg.iloc[-2]['h']), float(agg.iloc[-2]['l']), float(agg.iloc[-2]['c'])
def _near(px, lv, tol): 
    if lv<=0: return False
    return abs(px-lv)/lv <= tol
def _params(h): return {'intraday':dict(ha=4,macd=4,tol=0.0065,period='W'),
                        'swing':   dict(ha=5,macd=6,tol=0.0090,period='M'),
                        'position':dict(ha=6,macd=8,tol=0.0120,period='Y')}[h]
def generate_signal_core(ticker, asset_class, horizon, last_price, bars=None)->Dict[str,Any]:
    p=_params(horizon); tol=p['tol']
    daily=_daily(asset_class, ticker, 520); H,L,C=_prev_hlc(daily, p['period']); piv=_fibo(H,L,C)
    b = daily.copy() if (bars is None or len(bars)<50) else bars.copy()
    if 'dt' in b.columns: b=b.set_index(pd.to_datetime(b['dt'], utc=True))
    elif 't' in b.columns and not isinstance(b.index, pd.DatetimeIndex): b=b.set_index(pd.to_datetime(b['t'], unit='s', utc=True))
    b=b[['o','h','l','c']].astype(float).dropna()
    ha_o,ha_c=_ha(b); ha_delta=ha_c-ha_o; hist=_macd_hist(b['c']); rsi=_rsi(b['c'],14); atr=_atr(b,14)
    px=float(last_price); last_atr=float(atr.iloc[-1]) if len(atr) else px*0.006
    nR2=_near(px,piv['R2'],tol); nR3=_near(px,piv['R3'],tol); nS2=_near(px,piv['S2'],tol); nS3=_near(px,piv['S3'],tol)
    haG=_streak(ha_delta,True)>=p['ha']; haR=_streak(ha_delta,False)>=p['ha']
    macG=_streak(hist,True)>=p['macd']; macR=_streak(hist,False)>=p['macd']
    overheat=(nR2 or nR3) and (haG or macG); oversold=(nS2 or nS3) and (haR or macR)
    action='WAIT'; entry=px; tp1=tp2=stop=px
    if overheat:
        action='SHORT'
        if nR3: tp1,tp2,stop=piv['R2'],piv['P'],piv['R3']*(1+tol)
        else:   tp1,tp2,stop=(piv['P']+piv['S1'])/2.0,piv['S1'],piv['R2']*(1+tol)
    elif oversold:
        action='BUY'
        if nS3: tp1,tp2,stop=piv['S2'],piv['P'],piv['S3']*(1-tol)
        else:   tp1,tp2,stop=(piv['P']+piv['R1'])/2.0,piv['R1'],piv['S2']*(1-tol)
    else:
        tp1,tp2,stop=entry+0.6*last_atr, entry+1.1*last_atr, entry-0.8*last_atr
    conf=0.6 if action in ('BUY','SHORT') else 0.54
    nar=f"Пивоты(Fibo): P={piv['P']:.4f}, R2={piv['R2']:.4f}, R3={piv['R3']:.4f}, S2={piv['S2']:.4f}, S3={piv['S3']:.4f}."
    alt=dict(if_condition='подтверждение на уровне (rejection/смена цвета HA)', action=action, entry=entry, take_profit=[tp1,tp2], stop=stop)
    return dict(action=action, entry=entry, take_profit=[float(tp1),float(tp2)], stop=float(stop), confidence=float(conf), narrative_ru=nar, alt=alt)
