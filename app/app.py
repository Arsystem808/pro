# -*- coding: utf-8 -*-
import os, sys, json, time
from datetime import datetime
import streamlit as st

# --- чтобы импорт capintel работал и локально, и в облаке
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capintel import build_signal
from capintel.visuals_svg import render_gauge_svg

st.set_page_config(page_title="CapIntel — AI Signals", layout="wide")

st.sidebar.header("Параметры")
asset_class = st.sidebar.selectbox("Класс актива", ["equity", "crypto"], index=0)
horizon     = st.sidebar.selectbox("Горизонт", ["intraday", "swing", "position"], index=1)
ticker      = st.sidebar.text_input("Тикер", value="AAPL")
price_str   = st.sidebar.text_input("Текущая цена (необязательно)", value="")
dev_mode    = st.sidebar.toggle("Режим разработчика", value=False)

run = st.sidebar.button("Сгенерировать сигнал")

st.title("🙂 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если найден `models/meta.pkl`.")

def _fmt(v):
    return "—" if v is None else (f"{v:.2f}" if isinstance(v,(int,float)) else str(v))

if run:
    t0 = time.time()
    try:
        price = float(price_str) if price_str.strip() else None
        sig = build_signal(ticker.strip().upper(), asset_class, horizon, price)

        colL, colR = st.columns([2,1])
        with colL:
            st.markdown(f"### {ticker.upper()} • {asset_class.upper()} • {horizon}")
            st.markdown(f"### ➤ Решение: **{sig['action']}**")

            show_numbers = sig["action"] != "WAIT"
            entry = f"{sig['entry']:.2f}" if show_numbers and sig.get("entry") is not None else "—"
            tp1   = f"{sig['tp1']:.2f}"   if show_numbers and sig.get("tp1")   is not None else "—"
            tp2   = f"{sig['tp2']:.2f}"   if show_numbers and sig.get("tp2")   is not None else "—"
            stop  = f"{sig['stop']:.2f}"  if show_numbers and sig.get("stop")  is not None else "—"

            cols = st.columns(4)
            cols[0].markdown(f"**Вход:** {entry}")
            cols[1].markdown(f"**Цели:** TP1 {tp1} • TP2 {tp2}")
            cols[2].markdown(f"**Стоп:** {stop}")
            cols[3].markdown(f"**Уверенность:** {_fmt(sig.get('confidence'))}%")

            st.markdown(f"Размер позиции: {_fmt(sig.get('size'))}% NAV")

        with colR:
            score = float(sig.get("score", 0.0))
            st.markdown(render_gauge_svg(score, title="Общая оценка"), unsafe_allow_html=True)

        # Бейдж про ML
        st.info(sig.get("ml_text", "[ML OFF] Модель не найдена — используется rule-based логика."))

        # Короткий и альтернативный план
        base = (sig.get("short_text") or "").strip()
        if base:
            st.markdown(f"> {base}")
        alt = (sig.get("alt_text") or "").strip()
        if alt and alt != base:
            st.markdown("### Альтернативный план")
            st.markdown(f"• {alt}")

        meta_cols = st.columns(3)
        meta_cols[0].caption(f"Создан: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        meta_cols[1].caption(f"Истекает: —")
        meta_cols[2].caption(f"ID: —")

        if dev_mode:
            with st.expander("JSON"):
                st.code(json.dumps(sig, ensure_ascii=False, indent=2), language="json")

        st.caption("Не инвестиционный совет. Торговля сопряжена с риском.")
        st.caption(f"Готово за {time.time()-t0:.2f}s")

    except Exception as e:
        st.error("Ошибка при генерации сигнала:")
        st.exception(e)
else:
    st.caption("Выберите параметры слева и нажмите «Сгенерировать сигнал».")

