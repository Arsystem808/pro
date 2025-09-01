# -*- coding: utf-8 -*-
from __future__ import annotations

# ---------- Путь к пакету capintel ----------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ../
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- Импорты после фикса пути ----------
import streamlit as st
from typing import Any, Dict, List, Optional

# теперь пакет capintel будет найден
from capintel.signal_engine import build_signal
from capintel.narrator import trader_tone_narrative_ru
from capintel.visuals_svg import render_gauge_svg

# ---------- UI ----------
st.set_page_config(page_title="CapIntel — AI Signals", layout="wide")

st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если найден `models/meta.pkl`.")

# Sidebar
st.sidebar.header("Параметры")
asset_class = st.sidebar.selectbox("Класс актива", ["equity", "crypto"])
horizon = st.sidebar.selectbox("Горизонт", ["intraday", "swing", "position"])
ticker = st.sidebar.text_input("Тикер", "QQQ")
price_raw = st.sidebar.text_input("Текущая цена (необязательно)", "")
show_gauge = st.sidebar.toggle("Показывать индикатор", value=True)
dev = st.sidebar.toggle("Режим разработчика", value=False)
run = st.sidebar.button("Сгенерировать идею")

def fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{float(x):.2f}"

def render_targets(tps: List[float]) -> str:
    if not tps:
        return "—"
    return " • ".join([f"TP{i} {float(v):.2f}" for i, v in enumerate(tps, start=1)])

if run:
    # безопасно парсим цену
    try:
        price_val = float(price_raw.replace(",", ".")) if price_raw.strip() else None
    except Exception:
        price_val = None

    # генерируем идею
    spec: Dict[str, Any] = build_signal(ticker.upper().strip(), asset_class, horizon, price_val)

    st.subheader(f"{ticker.upper()} • {asset_class.upper()} • {horizon}")
    st.markdown(f"### ➤ Решение: **{spec.get('action', 'WAIT')}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"**Вход:** {fmt(spec.get('entry'))}")
    with c2: st.markdown(f"**Цели:** {render_targets(spec.get('take_profit', []))}")
    with c3: st.markdown(f"**Стоп:** {fmt(spec.get('stop'))}")
    with c4:
        conf = spec.get("confidence", 0.0)
        st.markdown(f"**Уверенность:** {round(100*float(conf))}%")

    c5, c6 = st.columns(2)
    with c5:
        psz = spec.get("position_size_pct_nav", 0.0)
        st.markdown(f"**Размер позиции:** {round(100*float(psz),2)}% NAV")
    with c6:
        st.markdown("&nbsp;")

    # Индикатор ([-2..+2], если нет — 0)
    if show_gauge:
        score = float(spec.get("rating", 0.0) or 0.0)
        st.markdown(render_gauge_svg(score=score, title="Общая оценка"),
                    unsafe_allow_html=True)

    # Блок ML
    ml = spec.get("ml", {})
    ml_note = spec.get("ml_note", "")
    if ml.get("on"):
        txt = "[ML ON]"
        p = ml.get("p_succ")
        if p is not None:
            try: txt += f" p_succ≈{round(100*float(p)):.0f}%."
            except Exception: pass
        if ml_note: txt += " " + str(ml_note)
        st.info(txt)
    else:
        st.info(ml_note or "[ML OFF]")

    # Наратив «по-человечески»
    narrative = spec.get("narrative_ru") or trader_tone_narrative_ru(spec, horizon)
    narrative = narrative.replace("сигнал", "идея").replace("Сигнал", "Идея")
    st.info(narrative)

    # Альтернативы: показываем только отличающиеся
    alts = spec.get("alternatives", []) or []
    def _differs(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        keys = ["action", "entry", "take_profit", "stop"]
        return any(str(a.get(k)) != str(b.get(k)) for k in keys)
    base_cmp = {k: spec.get(k) for k in ["action","entry","take_profit","stop"]}
    shown = []
    for alt in alts:
        if isinstance(alt, dict) and _differs(alt, base_cmp):
            shown.append(alt)
    if shown:
        st.markdown("### Альтернативный план")
        for alt in shown:
            cond = alt.get("if_condition") or "условие"
            act  = alt.get("action","")
            entry = fmt(alt.get("entry"))
            tps   = render_targets(alt.get("take_profit", []))
            stop  = fmt(alt.get("stop"))
            st.markdown(f"- **если {cond}** → **{act}** от **{entry}** → {tps}, **стоп** {stop}")

    if dev:
        with st.expander("JSON"):
            st.code(spec, language="json")

else:
    st.caption("Заполни параметры слева и нажми «Сгенерировать идею».")

