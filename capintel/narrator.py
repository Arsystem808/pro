# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

def make_narrative(sig: Dict[str,Any]) -> str:
    ml = sig.get("ml", {})
    p  = ml.get("p_succ", None)
    ml_on = ml.get("on", False)

    def ptxt():
        return f"[ML ON] p_succ≈{p:.2f}. " if (ml_on and p is not None) else "[ML OFF] "

    act = sig.get("action", "WAIT")
    cond = sig.get("conditional", False)

    if act == "WAIT":
        # WAIT — без уровней в тексте, без противоречий
        if ml_on and p is not None and p < 0.45:
            return ptxt() + "Модель отклонила сетап — бережём капитал. Действуем только при явном подтверждении от уровня."
        return ptxt() + "Сигнал пока не подтверждён. Ждём реакции на ключевые уровни и стабилизации импульса."
    else:
        # BUY/SHORT
        side = "покупку" if act == "BUY" else "шорт"
        if cond:
            return ptxt() + f"Условный сетап на {side}: работаем только после подтверждения от уровня/свечи-rejection."
        if ml_on and p is not None and p >= 0.70:
            return ptxt() + f"Сетап сильный по модели — допускаем {side} с аккуратным риском за экстремум."
        if ml_on and p is not None and p < 0.55:
            return ptxt() + f"Сетап слабый — если и входить в {side}, то после явного подтверждения/замедления импульса."
        return ptxt() + f"Ищем {side} от уровня с контролем риска и подтверждением цены."
