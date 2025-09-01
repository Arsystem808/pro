# capintel/narrator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional

def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{float(x):.2f}"

def _fmt_tps(tps: List[float]) -> str:
    if not tps:
        return "-"
    pts = [f"TP{i+1} {_fmt(v)}" for i, v in enumerate(tps[:2])]
    return ", ".join(pts)

def _pivots_hint(piv: Optional[Dict[str, float]]) -> str:
    if not isinstance(piv, dict):
        return ""
    keys = ["P","R1","R2","R3","S1","S2","S3"]
    parts = [f"{k}={piv[k]:.2f}" for k in keys if k in piv and isinstance(piv[k], (int,float))]
    return (" Пивоты: " + ", ".join(parts) + "." ) if parts else ""

def trader_tone_narrative_ru(
    *,
    ticker: str,
    asset_class: str,
    horizon: str,
    action: str,
    entry: Optional[float],
    take_profit: List[float],
    stop: Optional[float],
    confidence: float,
    last_price: float,
    pivots: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
    ml: Optional[Dict[str, Any]] = None,
) -> str:
    """Короткий человеческий комментарий под карточку сигнала."""
    conf_pct = int(round(confidence * 100))
    ml_note = ""
    if isinstance(ml, dict) and ml.get("on") and isinstance(ml.get("p_succ"), (int, float)):
        ml_note = f" ML-оценка p_succ≈{float(ml['p_succ']):.2f}."

    if action == "BUY":
        return (
            f"Покупка {ticker} ({horizon}). Вход ~{_fmt(entry)}; {_fmt_tps(take_profit)}; стоп {_fmt(stop)}. "
            f"Уверенность ~{conf_pct}%. Действуем аккуратно, риск контролируем.{_pivots_hint(piv)}{ml_note}"
        )
    if action == "SHORT":
        return (
            f"Шорт {ticker} ({horizon}). Вход ~{_fmt(entry)}; {_fmt_tps(take_profit)}; стоп {_fmt(stop)}. "
            f"Уверенность ~{conf_pct}%. Играем от отката/перезагрузки.{_pivots_hint(piv)}{ml_note}"
        )
    if action == "CLOSE":
        return (
            f"Фиксация позиции по {ticker} ({horizon}). Цель выполнена/сигнал ослаб. "
            f"Уверенность ~{conf_pct}%.{_pivots_hint(piv)}{ml_note}"
        )
    # WAIT и прочие
    hint = _pivots_hint(piv)
    level_hint = ""
    if isinstance(piv, dict):
        # Подсказываем, чего ждём
        if "R2" in piv or "R3" in piv:
            level_hint = " Ждём реакции на верхние уровни (R2/R3) или возврата к опорам."
        elif "S2" in piv or "S3" in piv:
            level_hint = " Ждём остановку снижения у S2/S3 и признаки разворота."
        elif "P" in piv:
            level_hint = " База — район P; ждём подтверждения импульса."
    return f"Сигнал неочевиден — бережём капитал и ждём понятного паттерна.{level_hint}{hint}{ml_note}"
