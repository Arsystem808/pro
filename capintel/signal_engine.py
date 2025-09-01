# capintel/signal_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Провайдер данных (Polygon)
from capintel.providers.polygon_client import (
    daily_bars,
    intraday_bars,
    latest_price,
)

# Стратегия: rule-based (ядро по твоим правилам)
from capintel.strategy.my_strategy import generate_signal_core as rb_generate

# (опционально) ML-надстройка: вероятности/мета-лейблинг
try:
    from capintel.strategy.my_strategy_ml import (
        prob_success as ml_prob_success,  # если реализован
    )
except Exception:
    ml_prob_success = None  # будет фоллбек на joblib-модель

# Фича-инженерия для ML (если используем joblib модель)
try:
    from capintel.ml.featurizer import make_feature_row
except Exception:
    make_feature_row = None  # не критично

# Комментарий «живым языком»
from capintel.narrator import trader_tone_narrative_ru


# ------------------------- Утилиты ------------------------- #

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _dedup_alternatives(base: Dict[str, Any],
                        alts: List[Dict[str, Any]],
                        eps: float = 1e-6) -> List[Dict[str, Any]]:
    """Убираем альтернативы, которые совпадают с базовым планом."""
    def _same(a: Optional[float], b: Optional[float]) -> bool:
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= eps

    base_entry = _safe_float(base.get("entry"))
    base_stop = _safe_float(base.get("stop"))
    base_tp = [ _safe_float(v) for v in (base.get("take_profit") or []) ]

    clean: List[Dict[str, Any]] = []
    for alt in alts or []:
        same_action = (alt.get("action") == base.get("action"))
        same_entry = _same(_safe_float(alt.get("entry")), base_entry)
        same_stop  = _same(_safe_float(alt.get("stop")), base_stop)

        tp = [ _safe_float(v) for v in (alt.get("take_profit") or []) ]
        same_tp = len(tp) == len(base_tp) and all(_same(a, b) for a, b in zip(tp, base_tp))

        if same_action and same_entry and same_stop and same_tp:
            # дубликат — пропускаем
            continue
        clean.append(alt)
    return clean


def _norm_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим столбцы к O/H/L/C с DatetimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["O", "H", "L", "C"])
    d = df.copy()
    # допускаем варианты названий
    cols_map = {
        "open": "O", "o": "O", "O": "O",
        "high": "H", "h": "H", "H": "H",
        "low":  "L", "l": "L", "L": "L",
        "close": "C", "c": "C", "C": "C",
    }
    d = d.rename(columns={k: v for k, v in cols_map.items() if k in d.columns})
    # только нужные
    for need in ["O", "H", "L", "C"]:
        if need not in d.columns:
            d[need] = np.nan
    # индекс во времени
    if not isinstance(d.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # пробуем конвертировать, если есть столбец времени
        for time_col in ["t", "time", "timestamp", "datetime"]:
            if time_col in d.columns:
                d.index = pd.to_datetime(d[time_col], errors="coerce")
                break
    d = d[["O", "H", "L", "C"]].astype(float)
    d = d.dropna(how="any")
    return d


def _load_joblib_model() -> Optional[Any]:
    """Пытаемся загрузить models/meta.pkl (если есть)."""
    try:
        from joblib import load
    except Exception:
        return None
    path = Path("models") / "meta.pkl"
    if not path.exists():
        return None
    try:
        return load(path.as_posix())
    except Exception:
        return None


# ------------------------- Основной API ------------------------- #

def build_signal(
    ticker: str,
    asset_class: str,
    horizon: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Строим торговую карточку:
    - тянем бары,
    - считаем rule-based сигнал,
    - (опц.) даём ML p_succ,
    - оформляем комментарий и альтернативы.
    """

    # 1) Данные
    asset_class = (asset_class or "").lower().strip()
    horizon = (horizon or "").lower().strip()

    try:
        if horizon == "intraday":
            bars_raw = intraday_bars(asset_class, ticker, minutes=390)  # последняя сессия
        else:
            # swing/position — ежедневные бары
            bars_raw = daily_bars(asset_class, ticker, n=520)
    except Exception as e:
        raise RuntimeError(f"Не удалось получить бары для {ticker}: {e}")

    bars = _norm_bars(bars_raw)
    if bars.empty:
        raise ValueError(f"Нет котировок для {ticker} ({asset_class}, {horizon}).")

    last_px = _safe_float(price) if price is not None else _safe_float(bars["C"].iloc[-1])
    if last_px is None:
        # пробуем спросить провайдера спот-цену
        try:
            last_px = _safe_float(latest_price(asset_class, ticker))
        except Exception:
            last_px = None

    # 2) Rule-based сигнал (ядро твоей стратегии)
    spec: Dict[str, Any] = rb_generate(
        ticker=ticker,
        asset_class=asset_class,
        horizon=horizon,
        last_price=last_px if last_px is not None else np.nan,
        bars=bars,
    )

    # нормализуем ключи/поля, чтобы дальше не падать
    action = (spec.get("action") or "WAIT").upper()
    entry = _safe_float(spec.get("entry"))
    take_profit = [ _safe_float(x) for x in (spec.get("take_profit") or []) if _safe_float(x) is not None ]
    stop = _safe_float(spec.get("stop"))
    confidence = float(spec.get("confidence") or 0.55)
    pos_size = float(spec.get("position_size_pct_nav") or 0.01)
    pivots = spec.get("pivots") or spec.get("piv") or None
    alts_in = spec.get("alternatives") or []
    alts_in = [_coerce_alt_dict(a) for a in alts_in]  # типизируем
    alts = _dedup_alternatives(
        {"action": action, "entry": entry, "take_profit": take_profit, "stop": stop},
        alts_in
    )

    # 3) ML-оценка (best-effort)
    ml_info: Dict[str, Any] = {"on": False}
    try:
        p_succ = None
        if ml_prob_success is not None:
            # реализация из my_strategy_ml.py
            p_succ = ml_prob_success(bars, last_px, spec.get("features"))
        else:
            # joblib-модель + featurizer (если доступны)
            model = _load_joblib_model()
            if model is not None and make_feature_row is not None:
                row = make_feature_row(ticker=ticker, asset_class=asset_class,
                                       horizon=horizon, last_price=last_px, bars=bars)
                X = np.asarray([row], dtype=float)
                # Мягкий вызов: некоторые пайплайны требуют predict_proba, некоторые — decision_function
                try:
                    if hasattr(model, "predict_proba"):
                        p = float(model.predict_proba(X)[0, 1])
                    elif hasattr(model, "decision_function"):
                        # приводим к [0..1]
                        z = float(model.decision_function(X)[0])
                        p = 1.0 / (1.0 + np.exp(-z))
                    else:
                        p = None
                except Exception:
                    p = None
                p_succ = p

        if isinstance(p_succ, (int, float)):
            ml_info = {"on": True, "p_succ": float(np.clip(p_succ, 0.0, 1.0))}
    except Exception:
        # Любая ML-ошибка не должна валить карточку
        ml_info = {"on": False}

    # 4) Комментарий
    narrative = trader_tone_narrative_ru(
        ticker=ticker,
        asset_class=asset_class,
        horizon=horizon,
        action=action,
        entry=entry,
        take_profit=take_profit,
        stop=stop,
        confidence=confidence,
        last_price=last_px if last_px is not None else np.nan,
        pivots=pivots,
        context={"source": "rule_based"},
        ml=ml_info,
    )

    # 5) Сборка финального словаря
    out: Dict[str, Any] = {
        "ticker": ticker,
        "asset_class": asset_class,
        "horizon": horizon,
        "action": action,
        "entry": entry,
        "take_profit": take_profit,
        "stop": stop,
        "confidence": confidence,
        "position_size_pct_nav": pos_size,
        "narrative_ru": narrative,
        "alternatives": alts,
        "ml": ml_info,
    }

    # Прокинем пивоты, чтобы их можно было показать в синем блоке
    if isinstance(pivots, dict) and pivots:
        out["pivots"] = pivots

    return out


def _coerce_alt_dict(alt: Dict[str, Any]) -> Dict[str, Any]:
    """Мягко приводим альтернативу к стандартизированному виду."""
    return {
        "if_condition": alt.get("if_condition") or alt.get("if") or "",
        "action": (alt.get("action") or "WAIT").upper(),
        "entry": _safe_float(alt.get("entry")),
        "take_profit": [
            _safe_float(x) for x in (alt.get("take_profit") or []) if _safe_float(x) is not None
        ],
        "stop": _safe_float(alt.get("stop")),
    }
