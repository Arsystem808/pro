# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
import httpx
from capintel.providers import polygon_client as poly

# ---- индикаторы ----
def _ema(s: pd.Series, span: int, wilder: bool=False) -> pd.Series:
    alpha = (1.0/span) if wilder else (2.0/(span+1.0))
    return s.ewm(alpha=alpha, adjust=False).mean()

def _rsi(c: pd.Series, n: int=14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    au = _ema(up,n,True); ad = _ema(dn,n,True).replace(0,np.nan)
    rs = au/ad
    return (100-100/(1+rs)).fillna(50)

def _atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    h,l,c = df["h"].astype(float), df["l"].astype(float), df["c"].astype(float)
    c1 = c.shift(1)
    tr = pd.concat([h-l,(h-c1).abs(),(l-c1).abs()],axis=1).max(axis=1)
    return _ema(tr,n,True)

def _macd_hist(c: pd.Series) -> pd.Series:
    m = _ema(c,12)-_ema(c,26); s = _ema(m,9); return m-s

def _ha(df: pd.DataFrame):
    o,h,l,c = [df[k].astype(float).values for k in ("o","h","l","c")]
    ha_c = (o+h+l+c)/4.0
    ha_o = np.zeros_like(ha_c); ha_o[0] = (o[0]+c[0])/2.0
    for i in range(1,len(ha_o)):
        ha_o[i] = (ha_o[i-1]+ha_c[i-1])/2.0
    return pd.Series(ha_o, index=df.index), pd.Series(ha_c, index=df.index)

def _streak(x: pd.Series, pos=True) -> int:
    if x.empty: return 0
    arr = np.sign(x.values); want = 1 if pos else -1; n = 0
    for v in arr[::-1]:
        if (v>0 and want>0) or (v<0 and want<0): n += 1
        elif v == 0: break
        else: break
    return n

def _fibo(H,L,C):
    P=(H+L+C)/3.0; d=H-L
    return {"P":P,"R1":P+0.382*d,"R2":P+0.618*d,"R3":P+1.0*d,"S1":P-0.382*d,"S2":P-0.618*d,"S3":P-1.0*d}

# ---- дневка с фолбэком при 429 ----
def _daily(asset_class, ticker, days=520, bars=None):
    try:
        if asset_class == 'crypto':
            b,q = poly._norm_crypto_pair(ticker); tkr = f"X:{b}{q}"
        else:
            tkr = ticker.upper()
        to = datetime.now(timezone.utc).date()
        fr = (to - timedelta(days=days))
        url = f"{poly.BASE}/v2/aggs/ticker/{tkr}/range/1/day/{fr}/{to}?adjusted=true&limit=50000&sort=asc"
        with httpx.Client(timeout=20) as c:
            r = c.get(url, headers=poly._headers())
            if r.status_code == 429:
                raise RuntimeError("rate-limit")
            r.raise_for_status()
            d = r.json()
        rows = (d or {}).get("results") or []
        if not rows: raise RuntimeError("empty")
        df = pd.DataFrame(rows)[["t","o","h","l","c","v"]].copy()
        if df["t"].max() > 1e12:
            df["t"] = (df["t"]/1000).astype(int)
        df["dt"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df.set_index("dt", inplace=True)
        return df[["o","h","l","c","v"]].astype(float)
    except Exception:
        pass
    # фолбэк: ресемплим bars до дневки
    try:
        if bars is None or len(bars)==0:
            return pd.DataFrame(columns=["o","h","l","c","v"])
        b = bars.copy()
        if "dt" in b.columns:
            b = b.set_index(pd.to_datetime(b["dt"], utc=True))
        elif "t" in b.columns and not isinstance(b.index, pd.DatetimeIndex):
            b = b.set_index(pd.to_datetime(b["t"], unit="s", utc=True))
        b = b[["o","h","l","c","v"]].astype(float).dropna()
        d = pd.DataFrame({
            "o": b["o"].resample("1D").first(),
            "h": b["h"].resample("1D").max(),
            "l": b["l"].resample("1D").min(),
            "c": b["c"].resample("1D").last(),
            "v": b["v"].resample("1D").sum(),
        }).dropna()
        return d.tail(days+5)
    except Exception:
        return pd.DataFrame(columns=["o","h","l","c","v"])

def _prev_hlc(daily: pd.DataFrame, period: str):
    if period=='W': g = daily.groupby(pd.Grouper(freq='W-MON',label='right'))
    elif period=='M': g = daily.groupby(pd.Grouper(freq='M',label='right'))
    else: g = daily.groupby(pd.Grouper(freq='Y',label='right'))
    agg = g.agg({'h':'max','l':'min','c':'last'}).dropna()
    if len(agg) < 2:
        tail = daily.tail({'W':5,'M':22,'Y':252}[period])
        return float(tail['h'].max()), float(tail['l'].min()), float(tail['c'].iloc[-1])
    return float(agg.iloc[-2]['h']), float(agg.iloc[-2]['l']), float(agg.iloc[-2]['c'])

def _near(px, lv, tol):
    if lv <= 0: return False
    return abs(px-lv)/lv <= tol

def _params(h):
    return {
        'intraday': dict(ha=4, macd=4, tol=0.0065, period='W'),
        'swing':    dict(ha=5, macd=6, tol=0.0090, period='M'),
        'position': dict(ha=6, macd=8, tol=0.0120, period='Y'),
    }[h]

# ---- генератор по ТЗ ----
def generate_signal_core(ticker, asset_class, horizon, last_price, bars=None) -> Dict[str,Any]:
    p   = _params(horizon); tol = p['tol']
    daily = _daily(asset_class, ticker, 520, bars=bars)

    # резервы, если bars пустые
    b = daily.copy() if (bars is None or len(bars) < 50) else bars.copy()
    if 'dt' in b.columns:
        b = b.set_index(pd.to_datetime(b['dt'], utc=True))
    elif 't' in b.columns and not isinstance(b.index, pd.DatetimeIndex):
        b = b.set_index(pd.to_datetime(b['t'], unit='s', utc=True))
    b = b[['o','h','l','c']].astype(float).dropna()

    H,L,C = _prev_hlc(daily if not daily.empty else b, p['period'])
    piv   = _fibo(H,L,C)

    ha_o,ha_c = _ha(b); ha_delta = ha_c - ha_o
    hist      = _macd_hist(b['c']); atr = _atr(b,14)

    px = float(last_price)
    last_atr = float(atr.iloc[-1]) if len(atr) else max(px*0.008, 1e-6)

    nR2 = _near(px,piv['R2'],tol); nR3=_near(px,piv['R3'],tol)
    nS2 = _near(px,piv['S2'],tol); nS3=_near(px,piv['S3'],tol)
    haG = _streak(ha_delta,True)  >= p['ha']
    haR = _streak(ha_delta,False) >= p['ha']
    macG= _streak(hist,True)      >= p['macd']
    macR= _streak(hist,False)     >= p['macd']

    overheat = (nR2 or nR3) and (haG or macG)
    oversold = (nS2 or nS3) and (haR or macR)

    # --- решение ---
    action='WAIT'; entry=px; tp1=tp2=stop=None  # для WAIT уровни не показываем
    alt = dict(if_condition='', action='WAIT', entry=px, take_profit=[px,px], stop=px)

    if horizon == 'position':
        # LT: у крыши при перегреве — базово WAIT, агрессивный SHORT как альтернатива
        if overheat:
            action = 'WAIT'
            if nR3:
                alt = dict(
                    if_condition='для опытных: шорт от R3 при подтверждении',
                    action='SHORT', entry=px, take_profit=[piv['R2'], piv['P']], stop=piv['R3']*(1+tol)
                )
            else:
                alt = dict(
                    if_condition='для опытных: шорт от R2 при подтверждении',
                    action='SHORT', entry=px, take_profit=[(piv['P']+piv['S1'])/2.0, piv['S1']], stop=piv['R2']*(1+tol)
                )
        elif oversold:
            action='BUY'
            if nS3:
                entry=px; tp1, tp2, stop = piv['S2'], piv['P'], piv['S3']*(1-tol)
            else:
                entry=px; tp1, tp2, stop = (piv['P']+piv['R1'])/2.0, piv['R1'], piv['S2']*(1-tol)
            alt = dict(if_condition='если импульс сломается — набираем по факту разворота', action='BUY',
                       entry=entry, take_profit=[tp1,tp2], stop=stop)
        else:
            # далеко от краёв — ждём перезагрузку в коридор
            action='WAIT'
            if px >= piv['R1']:
                alt = dict(if_condition='если появится отказ от верхней кромки — аккуратный шорт', action='SHORT',
                           entry=px, take_profit=[piv['P'], (piv['P']+piv['S1'])/2.0], stop=piv['R2']*(1+tol))
            elif px <= piv['S1']:
                alt = dict(if_condition='если удержим нижнюю кромку — осторожные покупки', action='BUY',
                           entry=px, take_profit=[piv['P'], (piv['P']+piv['R1'])/2.0], stop=piv['S2']*(1-tol))
            else:
                alt = dict(if_condition='ждать перезагрузку в коридор P–S1/S2', action='WAIT',
                           entry=px, take_profit=[px,px], stop=px)
        conf = 0.54 if action=='WAIT' else 0.62
        nar  = ("LT: ориентир на **годовые** пивоты. "
                "У крыши при перегреве — базово *WAIT*; покупки — после перезагрузки в коридор P–S1/S2 "
                "и признаков восстановления. Агрессивный шорт — только опция.")
       else:
        # ---------- INTRADAY / SWING ----------
        # базовые уровни-инвалидации (через допуск tol)
        thr_up_R3 = piv['R3'] * (1 + tol)
        thr_up_R2 = piv['R2'] * (1 + tol)
        thr_dn_S3 = piv['S3'] * (1 - tol)
        thr_dn_S2 = piv['S2'] * (1 - tol)

        if overheat:
            # базовый — шорт от "крыши"
            action = 'SHORT'
            if nR3:
                entry = px
                tp1, tp2, stop = piv['R2'], piv['P'], thr_up_R3
                # альтернатива — ИНВАЛИДАЦИЯ: пробой выше R3 -> BUY по импульсу
                alt = dict(
                    if_condition=f"если закрепится ВЫШЕ ~{thr_up_R3:.4f}",
                    action='BUY',
                    entry=px,
                    take_profit=[px + 0.6*last_atr, px + 1.1*last_atr],
                    stop=piv['R3']   # возврат ниже R3 — стоп
                )
            else:  # nR2
                entry = px
                tp1, tp2, stop = (piv['P']+piv['S1'])/2.0, piv['S1'], thr_up_R2
                # альтернатива — пробой выше R2 -> BUY в сторону R3
                alt = dict(
                    if_condition=f"если закрепится ВЫШЕ ~{thr_up_R2:.4f}",
                    action='BUY',
                    entry=px,
                    take_profit=[piv['R3'], piv['R3'] + 0.6*last_atr],
                    stop=piv['R2']
                )

        elif oversold:
            # базовый — лонг от "дна"
            action = 'BUY'
            if nS3:
                entry = px
                tp1, tp2, stop = piv['S2'], piv['P'], thr_dn_S3
                # альтернатива — ИНВАЛИДАЦИЯ: пробой ниже S3 -> SHORT по импульсу
                alt = dict(
                    if_condition=f"если закрепится НИЖЕ ~{thr_dn_S3:.4f}",
                    action='SHORT',
                    entry=px,
                    take_profit=[px - 0.6*last_atr, px - 1.1*last_atr],
                    stop=piv['S3']
                )
            else:  # nS2
                entry = px
                tp1, tp2, stop = (piv['P']+piv['R1'])/2.0, piv['R1'], thr_dn_S2
                # альтернатива — пробой ниже S2 -> SHORT в сторону S3
                alt = dict(
                    if_condition=f"если закрепится НИЖЕ ~{thr_dn_S2:.4f}",
                    action='SHORT',
                    entry=px,
                    take_profit=[piv['S3'], piv['S3'] - 0.6*last_atr],
                    stop=piv['S2']
                )

        else:
            # ни перегрева, ни дна — нейтрально; базовый WAIT с "коридорными" условиями
            action='WAIT'
            tp1 = entry + 0.6*last_atr; tp2 = entry + 1.1*last_atr; stop = entry - 0.8*last_atr
            # альтернатива — два сценария в зависимости от положения к P
            if px >= piv['R1']:
                alt = dict(
                    if_condition='если отобьёмся от верхней кромки — подтверждённый шорт',
                    action='SHORT', entry=px,
                    take_profit=[piv['P'], (piv['P']+piv['S1'])/2.0],
                    stop=piv['R2']*(1+tol)
                )
            elif px <= piv['S1']:
                alt = dict(
                    if_condition='если удержим нижнюю кромку — подтверждённый лонг',
                    action='BUY', entry=px,
                    take_profit=[piv['P'], (piv['P']+piv['R1'])/2.0],
                    stop=piv['S2']*(1-tol)
                )
            else:
                alt = dict(
                    if_condition='ждать перезагрузку в коридор P–S1/S2',
                    action='WAIT', entry=px, take_profit=[px,px], stop=px
                )

        conf = 0.6 if action in ('BUY','SHORT') else 0.54
        nar  = (
            f"Пивоты(Fibo): P={piv['P']:.4f}, R2={piv['R2']:.4f}, R3={piv['R3']:.4f}, "
            f"S2={piv['S2']:.4f}, S3={piv['S3']:.4f}. "
            "Работаем от краёв: база — контртренд, альтернатива — инвалидация через пробой уровня."
        )

    return dict(
        action=action, entry=entry,
        take_profit=[tp1 if tp1 is not None else entry,
                     tp2 if tp2 is not None else entry],
        stop=(stop if stop is not None else entry),
        confidence=float(conf), narrative_ru=nar, alt=alt
    )
