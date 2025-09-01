# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any
import pandas as pd

def trader_tone_narrative_ru(sig: Dict[str, Any], bars: pd.DataFrame) -> Tuple[str, str]:
    act = sig.get("action","WAIT")
    base = sig.get("short_text") or ""
    alt  = sig.get("alt_text") or ""

    if act == "WAIT":
        base = "Идея не сформирована. Ждём касание/реакцию на уровень и стабилизацию импульса."
        alt  = ""
    else:
        side = "лонг" if act=="LONG" else "шорт"
        base = f"Базовый {side}: от уровня, цели по ближайшим пивотам. Следим за импульсом и объёмом."
        if not alt:
            alt = f"Если реакция от кромки коридора — подтверждённый {side} по рынку малым объёмом."
    return base, alt
