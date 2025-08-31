from __future__ import annotations
from typing import Tuple, Dict
import numpy as np, pandas as pd
def ema(s: pd.Series, span: int, wilder: bool=False) -> pd.Series:
    alpha=(1.0/span) if wilder else (2.0/(span+1.0)); return s.ewm(alpha=alpha, adjust=False).mean()
def rsi_wilder(c: pd.Series, n: int=14) -> pd.Series:
    d=c.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0); au=ema(up,n,True); ad=ema(dn,n,True).replace(0,np.nan); rs=au/ad
    return (100-100/(1+rs)).fillna(50)
def atr_wilder(df: pd.DataFrame, n: int=14) -> pd.Series:
    h,l,c=df["h"].astype(float), df["l"].astype(float), df["c"].astype(float); c1=c.shift(1)
    tr=pd.concat([h-l,(h-c1).abs(),(l-c1).abs()],axis=1).max(axis=1); return ema(tr,n,True)
def macd_hist(c: pd.Series) -> pd.Series:
    m=ema(c,12)-ema(c,26); s=ema(m,9); return m-s
def heikin_ashi(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    o,h,l,c=[df[k].astype(float).values for k in ("o","h","l","c")]
    ha_c=(o+h+l+c)/4.0; ha_o=np.zeros_like(ha_c); ha_o[0]=(o[0]+c[0])/2.0
    for i in range(1,len(ha_o)): ha_o[i]=(ha_o[i-1]+ha_c[i-1])/2.0
    return pd.Series(ha_o, index=df.index), pd.Series(ha_c, index=df.index)
def fibo_pivots(H: float, L: float, C: float) -> Dict[str,float]:
    P=(H+L+C)/3.0; d=H-L; return {"P":P,"R1":P+0.382*d,"R2":P+0.618*d,"R3":P+1.0*d,"S1":P-0.382*d,"S2":P-0.618*d,"S3":P-1.0*d}
