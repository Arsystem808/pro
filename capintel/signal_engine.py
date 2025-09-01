# capintel/signal_engine.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# провайдеры
from capintel.providers.polygon_client import daily_bars, intraday_bars

# стратегия/ML
from capintel.strategy.my_strategy import generate_signal_core as rb_generate
from capintel.strategy.my_strategy_ml import prob_success
from joblib import load as joblib_load

# ---- утилиты ----

ROOT = Path(__file__).resolve().parents[1]

def _find_model_path() -> Optional[Path]:
    # Ищем и локально, и в докер/облаке
    candidates = [
        ROOT / "models" / "meta.pkl",
        Path.cwd() / "models" / "meta.pkl",
        Path("/mount/src/pro/models/meta.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _load_model():
    mp = _find_model_path()
    if mp is None:
        return None
    try:
        return joblib_load(mp)
    except Exception:
        return None

def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # варианты имён колонок
    for want, alts in {
        "o": ["o","open","op","opening_price"],
        "h": ["h","high","hi","max"],
        "l": ["l","low","lo","min"],
        "c": ["c","close","cl","closing_price","price"],
        "v": ["v","volume","vol"]
    }.items():
        for a in alts:
            if a in cols:
                mapping[want] = cols[a]
                break
    # если чего-то не хватило — не падаем, берём что смогли
    for k in ["o","h","l","c"]:
        if k not in mapping:
            # иногда приходит только 'c'
            if "c" in mapping:
                mapping[k] = mapping["c"]
    out = df.rename(columns={v:k for k,v in mapping.items()})
    # только нужные
    keep = [c for c in ["o","h","l","c","v"] if c in out.columns]
    out = out[keep].copy()
    # индекс к DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    # числа
    for c in ["o","h","l","c","v"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    return out.sort_index()

def _fetch_bars(asset_class: str, ticker: str, horizon: str) -> pd.DataFrame:
    if horizon == "intraday":
        df = intraday_bars(asset_class, ticker, timespan="5m", lookback_days=7)
    elif horizon in ("swing","position"):
        # обе используют дневные бары (разная логика в стратегии)
        df = daily_bars(asset_class, ticker, days=520)
    else:
        df = daily_bars(asset_class, ticker, days=520)
    return _standardize_ohlc(df)

# ---- публичная точка входа ----

def build_signal(ticker: str, asset_class: str, horizon: str, price: Optional[float]=None) -> Dict[str,Any]:
    bars = _fetch_bars(asset_class, ticker, horizon)
    if bars is None or bars.empty or not {"o","h","l","c"}.issubset(set(bars.columns)):
        return {
            "action": "WAIT",
            "confidence": 0.6,
            "show_numbers": False,
            "ml_on": False,
            "comment_lt": "Данные недоступны: провайдер вернул пустые/некорректные бары. Проверь ключ API/тикер/часовой пояс.",
        }

    # базовая стратегия
    sig = rb_generate(bars=bars, asset_class=asset_class, ticker=ticker)
    # sig: {action, entry, tp1, tp2, stop, short_text, alt_text}

    # ML meta-label (если модель загрузилась и фичи совпадают)
    model = _load_model()
    ml_on = model is not None
    p_succ = None
    if ml_on:
        try:
            feats = _make_feats_for_meta(bars)  #  shape (1, n_features)
            # сверяем число фич
            want = getattr(model, "n_features_in_", None)
            if want and feats.shape[1] == want:
                p_succ = float(prob_success(model, feats))  # 0..1
            else:
                ml_on = False
        except Exception:
            ml_on = False
            p_succ = None

    # оформление ответа
    action = sig.get("action","WAIT")
    show_numbers = action != "WAIT" and all(k in sig for k in ("entry","tp1","tp2","stop"))
    ans = {
        "action": action,
        "entry": float(sig["entry"]) if show_numbers else None,
        "tp1": float(sig["tp1"]) if show_numbers else None,
        "tp2": float(sig["tp2"]) if show_numbers else None,
        "stop": float(sig["stop"]) if show_numbers else None,
        "confidence": 0.6 if p_succ is None else max(0.5, min(0.95, p_succ)),
        "position_size_nav": 0.0,
        "ml_on": ml_on,
        "comment_ml": "[ML ON]" if ml_on else "[ML OFF] Модель не найдена — используется базовая логика.",
        "comment_lt": sig.get("short_text") or (
            "Пока без сделки: ждём касания/реакции на уровень и стабилизации импульса."
        ),
        "alt": (sig.get("alt_text") or "").strip(),
        "show_numbers": show_numbers,
    }
    return ans

# ---- простая фича-матрица для meta-модели ----
def _make_feats_for_meta(bars: pd.DataFrame) -> np.ndarray:
    # Минимальный набор устойчивых фич (пример)
    df = bars[["o","h","l","c"]].copy()
    df["ret1"] = df["c"].pct_change()
    df["rng"] = (df["h"]-df["l"]).div(df["c"].replace(0,np.nan)).fillna(0)
    df["ma10"] = df["c"].rolling(10).mean()
    df["ma20"] = df["c"].rolling(20).mean()
    last = df.tail(20)
    feats = [
        last["ret1"].tail(5).mean(),
        last["ret1"].tail(5).std(ddof=0),
        float(last["rng"].tail(10).mean()),
        float(last["c"].iloc[-1] / (last["ma10"].iloc[-1] or np.nan)),
        float(last["ma10"].iloc[-1] / (last["ma20"].iloc[-1] or np.nan)),
    ]
    feats = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in feats]
    return np.array(feats, dtype=float).reshape(1, -1)
