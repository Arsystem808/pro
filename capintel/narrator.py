# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

def trader_tone_narrative_ru(spec: Dict[str, Any], horizon: str) -> str:
    act = str(spec.get("action", "WAIT")).upper()
    hru = {"intraday": "INTRADAY", "swing": "SWING", "position": "LT"}.get(horizon, horizon).upper()

    if act == "WAIT":
        base = f"{hru}: идея пока неочевидна — ждём реакции на уровень и стабилизации импульса."
    elif act == "BUY":
        base = f"{hru}: покупка уместна — есть подтверждение по уровням/импульсу."
    elif act == "SHORT":
        base = f"{hru}: шорт уместен — есть признаки перегрева/ослабления."
    else:
        base = f"{hru}: решение {act}."

    # Лёгкая пометка по ML
    ml = spec.get("ml", {})
    if ml.get("on") and "p_succ" in ml:
        base += f" ML оценивает шанс успеха ≈ {round(100*float(ml['p_succ'])):.0f}%."
    elif not ml.get("on"):
        base += " (ML выключен)"

    return base
