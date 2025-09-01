# app/app.py
# -*- coding: utf-8 -*-

import math
from typing import Any, Dict, List, Optional

import streamlit as st

# ядро
from capintel.signal_engine import build_signal
# svg-шкала (устойчивый рендер и на строку, и на кортеж)
from capintel.visuals_svg import render_gauge_svg


# -----------------------------
# Вспомогательные функции
# -----------------------------
def _fmt(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def _fmt_pct(x: Optional[float], digits: int = 0) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.{digits}f}%"
    except Exception:
        return "—"

def _score_from(sig: Dict[str, Any]) -> float:
    """
    Нормируем «оценку» на шкалу [-2; +2].
    Основано на решении и confidence; если есть ML p_succ — слегка учитываем.
    """
    base = {"BUY": 1.0, "LONG": 1.0, "SHORT": -1.0, "SELL": -1.0, "WAIT": 0.0}
    action = str(sig.get("action", "WAIT")).upper()
    conf = float(sig.get("confidence", 0.5) or 0.5)  # 0..1
    ml = sig.get("ml") or {}
    p_succ = float(ml.get("p_succ", 0.5) or 0.5)

    # базовая амплитуда: 0.8…2.0 в зависимости от confidence
    amp = 0.8 + 1.2 * conf
    s = amp * base.get(action, 0.0)

    # мягкая поправка ML к направлению, если решение не WAIT
    if action != "WAIT":
        tilt = (p_succ - 0.5) * 1.2  # -0.6..+0.6
        s += math.copysign(abs(tilt), s)

    return max(-2.0, min(2.0, s))


def _render_gauge(score: float, title: str = "Общая оценка") -> None:
    """Аккуратно рисуем шкалу: поддерживаем и строку, и кортеж (svg, height)."""
    try:
        gauge = render_gauge_svg(score=score, title=title)
        if isinstance(gauge, tuple):
            svg, h = gauge
            st.components.v1.html(svg, height=int(h), scrolling=False)
        else:
            st.markdown(gauge, unsafe_allow_html=True)
    except Exception:
        st.info("⚠️ Индикатор временно скрыт (ошибка рендера).")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="CapIntel — AI Signals (Crypto & Equities)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если найден `models/meta.pkl`.")

# --- Sidebar
st.sidebar.header("Параметры")
asset_class = st.sidebar.selectbox(
    "Класс актива", options=["equity", "crypto"], index=0
)
horizon = st.sidebar.selectbox(
    "Горизонт", options=["intraday", "swing", "position"], index=1
)
default_ticker = "AAPL" if asset_class == "equity" else "BTCUSD"
ticker = st.sidebar.text_input("Тикер", value=default_ticker).strip().upper()

price_override_str = st.sidebar.text_input("Текущая цена (необязательно)", value="")
show_gauge = st.sidebar.toggle("Показывать индикатор", value=True)
dev_mode = st.sidebar.toggle("Режим разработчика", value=False)

if st.sidebar.button("Сгенерировать сигнал"):
    st.session_state._go = True

# Чтобы не требовать обязательной кнопки
go = st.session_state.get("_go", True)

# -----------------------------
# Генерация сигнала
# -----------------------------
if go:
    if not ticker:
        st.warning("Введите тикер и нажмите «Сгенерировать сигнал».")
        st.stop()

    price_override: Optional[float] = None
    if price_override_str:
        try:
            price_override = float(price_override_str)
        except Exception:
            st.warning("Не удалось распознать цену. Игнорируем поле «Текущая цена».")
            price_override = None

    try:
        sig: Dict[str, Any] = build_signal(
            ticker=ticker,
            asset_class=asset_class,
            horizon=horizon,
            price=price_override
        )
    except Exception as e:
        st.error("Ошибка при генерации: " + repr(e))
        if dev_mode:
            import traceback, sys
            st.exception(e)
        st.stop()

    # -----------------------------
    # Заголовок карточки
    # -----------------------------
    st.subheader(f"{ticker} · {asset_class.upper()} · {horizon}")

    action = str(sig.get("action", "WAIT")).upper()
    entry = sig.get("entry")
    tps: List[float] = sig.get("take_profit") or []
    stop = sig.get("stop")
    conf = sig.get("confidence")
    pos_sz = sig.get("position_size_pct_nav")

    colA, colB, colC, colD, colE = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])

    with colA:
        st.markdown(f"### ➤ Решение: **{action}**")

    # Показываем числа ТОЛЬКО если не WAIT
    show_numbers = action != "WAIT"

    with colB:
        st.write("**Вход:**", _fmt(entry) if show_numbers else "—")
    with colC:
        if show_numbers and tps:
            st.write("**Цели:**", " · ".join([f"TP{i+1} {_fmt(tp)}" for i, tp in enumerate(tps)]))
        else:
            st.write("**Цели:** —")
    with colD:
        st.write("**Стоп:**", _fmt(stop) if show_numbers else "—")
    with colE:
        st.write("**Уверенность:**", _fmt_pct(conf, 0))

    # Размер позиции (может быть None или 0)
    st.write("**Размер позиции:**", _fmt_pct(pos_sz, 2))

    # --- ML статус / тех. ремарка
    ml = sig.get("ml") or {}
    if ml.get("on"):
        p_succ = ml.get("p_succ")
        msg = "[ML ON]"
        if p_succ is not None:
            msg += f" p_succ≈{_fmt(p_succ, 2)}"
        st.info(msg)
    else:
        st.info(sig.get("tech_note_ru", "[ML OFF] Модель не найдена — используется rule-based логика."))

    # --- Индикатор (шкала)
    if show_gauge:
        score = _score_from(sig)
        _render_gauge(score, title="Общая оценка")

    # --- Комментарий
    note = sig.get("narrative_ru") or sig.get("note_ru") or sig.get("note") or ""
    if note:
        st.markdown(f"> {note}")

    # --- Альтернативы
    alts = sig.get("alternatives") or []
    if alts:
        st.markdown("### Альтернативный план")
        for alt in alts:
            alt_action = str(alt.get("action", "")).upper()
            alt_entry = alt.get("entry")
            alt_tps = alt.get("take_profit") or []
            alt_stop = alt.get("stop")
            cond = alt.get("if_condition") or alt.get("when") or ""
            line = []
            if cond:
                line.append(f"*{cond}*: ")
            line.append(f"**{alt_action}** от {_fmt(alt_entry)}")
            if alt_tps:
                line.append(" → " + " · ".join([f"TP{i+1} {_fmt(tp)}" for i, tp in enumerate(alt_tps)]))
            if alt_stop is not None:
                line.append(f", стоп {_fmt(alt_stop)}")
            st.markdown("• " + "".join(line))

    # --- Тех. JSON
    with st.expander("JSON", expanded=False):
        st.json(sig)

    # --- Дисклеймер
    st.caption("Не инвестиционный совет. Торговля сопряжена с риском.")
