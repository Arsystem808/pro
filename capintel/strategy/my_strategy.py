# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import httpx

from capintel.providers import polygon_client as poly


# =========================
# БАЗОВЫЕ ИНДИКАТОРЫ
# =========================
def _ema(s: pd.Series, span: int, wilder: bool = False) -> pd.Series:
    alpha = (1.0 / span) if wilder else (2.0 / (span + 1.0))
    return s.ewm(alpha=alpha, adjust=False).mean()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = _ema(up, n, True)
    ad = _ema(dn, n, True).replace(0, np.nan)
    rs = au / ad
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["h"].astype(float), df["l"].astype(float), df["c"].astype(float)
    c1 = c.shift(1)
    tr = pd.concat([h - l, (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
    return _ema(tr, n, True)


def _macd_hist(c: pd.Series) -> pd.Series:
    m = _ema(c, 12) - _ema(c, 26)
    s = _ema(m, 9)
    return m - s


def _ha(df: pd.DataFrame):
    o, h, l, c = [df[k].astype(float).values for k in ("o", "h", "l", "c")]
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.zeros_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_o)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return pd.Series(ha_o, index=df.index), pd.Series(ha_c, index=df.index)


def _streak(x: pd.Series, pos: bool = True) -> int:
    """Длина последнего однонаправленного отрезка по знаку x."""
    if x.empty:
        return 0
    arr = np.sign(x.values)
    want = 1 if pos else -1
    n = 0
    for v in arr[::-1]:
        if (v > 0 and want > 0) or (v < 0 and want < 0):
            n += 1
        elif v == 0:
            break
        else:
            break
    return n


def _fibo(H: float, L: float, C: float) -> Dict[str, float]:
    P = (H + L + C) / 3.0
    d = H - L
    return {
        "P": P,
        "R1": P + 0.382 * d,
        "R2": P + 0.618 * d,
        "R3": P + 1.000 * d,
        "S1": P - 0.382 * d,
        "S2": P - 0.618 * d,
        "S3": P - 1.000 * d,
    }


# =========================
# ДНЕВКА С ФОЛБЭКОМ (429)
# =========================
def _daily(asset_class, ticker, days: int = 520, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Пытаемся взять 1D через Polygon; при ошибке строим дневку из bars ресемплингом."""
    # 1) Прямой вызов Polygon
    try:
        if asset_class == "crypto":
            b, q = poly._norm_crypto_pair(ticker)
            tkr = f"X:{b}{q}"
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
        if not rows:
            raise RuntimeError("empty")
        df = pd.DataFrame(rows)[["t", "o", "h", "l", "c", "v"]].copy()
        # ms → s
        if df["t"].max() > 1e12:
            df["t"] = (df["t"] / 1000).astype(int)
        df["dt"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df.set_index("dt", inplace=True)
        return df[["o", "h", "l", "c", "v"]].astype(float)
    except Exception:
        pass

    # 2) Фолбэк: ресемплим bars до дневки
    try:
        if bars is None or len(bars) == 0:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
        b = bars.copy()
        if "dt" in b.columns:
            b = b.set_index(pd.to_datetime(b["dt"], utc=True))
        elif "t" in b.columns and not isinstance(b.index, pd.DatetimeIndex):
            b = b.set_index(pd.to_datetime(b["t"], unit="s", utc=True))
        b = b[["o", "h", "l", "c", "v"]].astype(float).dropna()
        d = pd.DataFrame(
            {
                "o": b["o"].resample("1D").first(),
                "h": b["h"].resample("1D").max(),
                "l": b["l"].resample("1D").min(),
                "c": b["c"].resample("1D").last(),
                "v": b["v"].resample("1D").sum(),
            }
        ).dropna()
        return d.tail(days + 5)
    except Exception:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"])


def _prev_hlc(daily: pd.DataFrame, period: str) -> tuple[float, float, float]:
    """Берём H/L/C прошлого периода: W|M|Y."""
    if period == "W":
        g = daily.groupby(pd.Grouper(freq="W-MON", label="right"))
    elif period == "M":
        g = daily.groupby(pd.Grouper(freq="M", label="right"))
    else:
        g = daily.groupby(pd.Grouper(freq="Y", label="right"))
    agg = g.agg({"h": "max", "l": "min", "c": "last"}).dropna()
    if len(agg) < 2:
        tail = daily.tail({"W": 5, "M": 22, "Y": 252}[period])
        return float(tail["h"].max()), float(tail["l"].min()), float(tail["c"].iloc[-1])
    return float(agg.iloc[-2]["h"]), float(agg.iloc[-2]["l"]), float(agg.iloc[-2]["c"])


def _near(px: float, lv: float, tol: float) -> bool:
    if lv <= 0:
        return False
    return abs(px - lv) / lv <= tol


def _params(h: str) -> Dict[str, Any]:
    return {
        "intraday": dict(ha=4, macd=4, tol=0.0065, period="W"),
        "swing": dict(ha=5, macd=6, tol=0.0090, period="M"),
        "position": dict(ha=6, macd=8, tol=0.0120, period="Y"),
    }[h]


# =========================
# ЯДРО СТРАТЕГИИ
# =========================
def generate_signal_core(
    ticker: str,
    asset_class: str,
    horizon: str,
    last_price: float,
    bars: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    p = _params(horizon)
    tol = p["tol"]

    # дневка для сводных уровней и пивотов
    daily = _daily(asset_class, ticker, 520, bars=bars)

    # рабочие бары для индикаторов
    b = daily.copy() if (bars is None or len(bars) < 50) else bars.copy()
    if "dt" in b.columns:
        b = b.set_index(pd.to_datetime(b["dt"], utc=True))
    elif "t" in b.columns and not isinstance(b.index, pd.DatetimeIndex):
        b = b.set_index(pd.to_datetime(b["t"], unit="s", utc=True))
    b = b[["o", "h", "l", "c"]].astype(float).dropna()

    # пивоты по прошлому периоду
    base_df = daily if not daily.empty else b
    H, L, C = _prev_hlc(base_df, p["period"])
    piv = _fibo(H, L, C)

    # индикаторы
    ha_o, ha_c = _ha(b)
    ha_delta = ha_c - ha_o
    hist = _macd_hist(b["c"])
    atr = _atr(b, 14)

    px = float(last_price)
    last_atr = float(atr.iloc[-1]) if len(atr) else max(px * 0.008, 1e-6)

    # условия «края»: у крыши/дна + длинные серии
    nR2 = _near(px, piv["R2"], tol)
    nR3 = _near(px, piv["R3"], tol)
    nS2 = _near(px, piv["S2"], tol)
    nS3 = _near(px, piv["S3"], tol)
    haG = _streak(ha_delta, True) >= p["ha"]
    haR = _streak(ha_delta, False) >= p["ha"]
    macG = _streak(hist, True) >= p["macd"]
    macR = _streak(hist, False) >= p["macd"]

    overheat = (nR2 or nR3) and (haG or macG)   # у крыши
    oversold = (nS2 or nS3) and (haR or macR)   # у дна

    # ---- решение + альтернатива ----
    action = "WAIT"
    entry = px
    tp1 = tp2 = stop = None
    alt: Dict[str, Any] = dict(if_condition="ждать перезагрузку", action="WAIT", entry=px, take_profit=[px, px], stop=px)

    if horizon == "position":
        # ===== ДОЛГОСРОК (LT, годовые уровни) =====
        if overheat:
            # базово WAIT; агрессивный SHORT — опция
            action = "WAIT"
            if nR3:
                alt = dict(
                    if_condition="для опытных: шорт от R3 при подтверждении",
                    action="SHORT",
                    entry=px,
                    take_profit=[piv["R2"], piv["P"]],
                    stop=max(px, piv["R3"]) * (1 + tol),  # стоп ВЫШЕ входа
                )
            else:
                alt = dict(
                    if_condition="для опытных: шорт от R2 при подтверждении",
                    action="SHORT",
                    entry=px,
                    take_profit=[(piv["P"] + piv["S1"]) / 2.0, piv["S1"]],
                    stop=max(px, piv["R2"]) * (1 + tol),  # стоп ВЫШЕ входа
                )
            conf = 0.54
            nar = (
                "LT: у годовой крыши базово **WAIT**. Агрессивный шорт — только опция при явном отказе от уровня. "
                "Рабочие покупки — после перезагрузки в коридор P–S1/S2."
            )
        elif oversold:
            # покупки от «дна»
            action = "BUY"
            if nS3:
                entry = px
                tp1, tp2, stop = piv["S2"], piv["P"], min(px, piv["S3"]) * (1 - tol)  # стоп НИЖЕ входа
            else:
                entry = px
                tp1, tp2, stop = (piv["P"] + piv["R1"]) / 2.0, piv["R1"], min(px, piv["S2"]) * (1 - tol)  # стоп НИЖЕ входа
            conf = 0.62
            nar = "LT: у годового «дна» допускаем аккуратные покупки с целями S2→P или P/R1→R1."
            alt = dict(
                if_condition="если импульс сломается — набираем после разворота",
                action="BUY",
                entry=entry,
                take_profit=[tp1, tp2],
                stop=stop,
            )
        else:
            # далеко от краёв — ждём
            action = "WAIT"
            conf = 0.54
            nar = "LT: сигнал неочевиден. Ждём перезагрузки к коридору и понятной реакции."
            if px >= piv["R1"]:
                alt = dict(
                    if_condition="если появится отказ от верхней кромки — аккуратный шорт",
                    action="SHORT",
                    entry=px,
                    take_profit=[piv["P"], (piv["P"] + piv["S1"]) / 2.0],
                    stop=max(px, piv["R2"]) * (1 + tol),  # стоп ВЫШЕ входа
                )
            elif px <= piv["S1"]:
                alt = dict(
                    if_condition="если удержим нижнюю кромку — осторожные покупки",
                    action="BUY",
                    entry=px,
                    take_profit=[piv["P"], (piv["P"] + piv["R1"]) / 2.0],
                    stop=min(px, piv["S2"]) * (1 - tol),  # стоп НИЖЕ входа
                )
            else:
                alt = dict(
                    if_condition="ждать перезагрузку в коридор P–S1/S2",
                    action="WAIT",
                    entry=px,
                    take_profit=[px, px],
                    stop=px,
                )

    else:
        # ===== INTRADAY / SWING =====
        # пороги «инвалидации» (пробой уровня через допуск)
        thr_up_R3 = piv["R3"] * (1 + tol)
        thr_up_R2 = piv["R2"] * (1 + tol)
        thr_dn_S3 = piv["S3"] * (1 - tol)
        thr_dn_S2 = piv["S2"] * (1 - tol)

        if overheat:
            # базовый — SHORT от «крыши»
            action = "SHORT"
            if nR3:
                entry = px
                tp1, tp2, stop = piv["R2"], piv["P"], max(px, thr_up_R3)  # стоп ВЫШЕ входа
                # альтернатива — ИНВАЛИДАЦИЯ: пробой выше R3 → BUY по импульсу
                alt = dict(
                    if_condition=f"если закрепится ВЫШЕ ~{thr_up_R3:.4f}",
                    action="BUY",
                    entry=px,
                    take_profit=[px + 0.6 * last_atr, px + 1.1 * last_atr],
                    stop=piv["R3"],  # возврат ниже R3 — стоп
                )
            else:
                entry = px
                tp1, tp2, stop = (piv["P"] + piv["S1"]) / 2.0, piv["S1"], max(px, thr_up_R2)  # стоп ВЫШЕ входа
                # альтернатива — пробой выше R2 → BUY в сторону R3
                alt = dict(
                    if_condition=f"если закрепится ВЫШЕ ~{thr_up_R2:.4f}",
                    action="BUY",
                    entry=px,
                    take_profit=[piv["R3"], piv["R3"] + 0.6 * last_atr],
                    stop=piv["R2"],
                )

            conf = 0.60
            nar = (
                f"Пивоты(Fibo): P={piv['P']:.4f}, R2={piv['R2']:.4f}, R3={piv['R3']:.4f}. "
                "У крыши работаем от отката; альтернатива — импульсный пробой в рост."
            )

        elif oversold:
            # базовый — BUY от «дна»
            action = "BUY"
            if nS3:
                entry = px
                tp1, tp2, stop = piv["S2"], piv["P"], min(px, thr_dn_S3)  # стоп НИЖЕ входа
                # альтернатива — ИНВАЛИДАЦИЯ: пробой ниже S3 → SHORT по импульсу
                alt = dict(
                    if_condition=f"если закрепится НИЖЕ ~{thr_dn_S3:.4f}",
                    action="SHORT",
                    entry=px,
                    take_profit=[px - 0.6 * last_atr, px - 1.1 * last_atr],
                    stop=piv["S3"],
                )
            else:
                entry = px
                tp1, tp2, stop = (piv["P"] + piv["R1"]) / 2.0, piv["R1"], min(px, thr_dn_S2)  # стоп НИЖЕ входа
                # альтернатива — пробой ниже S2 → SHORT в сторону S3
                alt = dict(
                    if_condition=f"если закрепится НИЖЕ ~{thr_dn_S2:.4f}",
                    action="SHORT",
                    entry=px,
                    take_profit=[piv["S3"], piv["S3"] - 0.6 * last_atr],
                    stop=piv["S2"],
                )

            conf = 0.60
            nar = (
                f"Пивоты(Fibo): P={piv['P']:.4f}, S2={piv['S2']:.4f}, S3={piv['S3']:.4f}. "
                "У дна работаем от разворота; альтернатива — импульсный пробой вниз."
            )

        else:
            # нейтральная зона — WAIT; альтернативы от ближайшей кромки коридора
            action = "WAIT"
            conf = 0.54
            # «коридорные» цели на случай быстрых сценариев (в UI скрыты при WAIT)
            tp1 = entry + 0.6 * last_atr
            tp2 = entry + 1.1 * last_atr
            stop = entry - 0.8 * last_atr

            if px >= piv["R1"]:
                alt = dict(
                    if_condition="если отобьёмся от верхней кромки — подтверждённый шорт",
                    action="SHORT",
                    entry=px,
                    take_profit=[piv["P"], (piv["P"] + piv["S1"]) / 2.0],
                    stop=max(px, piv["R2"]) * (1 + tol),  # стоп ВЫШЕ входа
                )
            elif px <= piv["S1"]:
                alt = dict(
                    if_condition="если удержим нижнюю кромку — подтверждённый лонг",
                    action="BUY",
                    entry=px,
                    take_profit=[piv["P"], (piv["P"] + piv["R1"]) / 2.0],
                    stop=min(px, piv["S2"]) * (1 - tol),  # стоп НИЖЕ входа
                )
            else:
                alt = dict(
                    if_condition="ждать перезагрузку в коридор P–S1/S2",
                    action="WAIT",
                    entry=px,
                    take_profit=[px, px],
                    stop=px,
                )
            nar = "Сигнал неочевиден — ждём подтверждения от уровня и стабилизации импульса."

    # возврат
    return dict(
        action=action,
        entry=entry,
        take_profit=[
            float(tp1) if tp1 is not None else float(entry),
            float(tp2) if tp2 is not None else float(entry),
        ],
        stop=float(stop) if stop is not None else float(entry),
        confidence=float(conf),
        narrative_ru=nar,
        alt=alt,
    )

