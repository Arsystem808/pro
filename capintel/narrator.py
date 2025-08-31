from typing import Literal
Action = Literal["BUY","SHORT","CLOSE","WAIT"]
def trader_tone_narrative_ru(action: Action, horizon: str, last_price: float) -> str:
    base = {"intraday":"Действуем аккуратно: важна текущая динамика.","swing":"Смотрим на 1–3 дня: уровни и реакция цены.","position":"Фокус на устойчивости движений и ключевых зонах."}[horizon]
    return {"BUY":f"Покупатели удерживают инициативу — берём вход с контролем риска. {base}",
            "SHORT":f"Рынок теряет импульс — ищем отбой для короткой позиции. {base}",
            "CLOSE":f"Движение выработано — фиксируем результат. {base}",
            "WAIT":f"Сигнал неочевиден — ждём подтверждения и бережём капитал. {base}"}[action]
