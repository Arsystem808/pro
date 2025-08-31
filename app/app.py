# app/app.py
import os
import streamlit as st
from capintel.signal_engine import build_signal

st.set_page_config(page_title="CapIntel — AI Signals", layout="wide")

st.sidebar.header("Параметры")
asset_class = st.sidebar.selectbox("Класс актива", ["equity", "crypto"])
horizon     = st.sidebar.selectbox("Горизонт", ["intraday", "swing", "position"])
ticker      = st.sidebar.text_input("Тикер", "AAPL").upper()
price_in    = st.sidebar.text_input("Текущая цена (опционально)", "")
price       = float(price_in.replace(",", ".")) if price_in.strip() else None

if st.sidebar.button("Сгенерировать сигнал"):
    sig = build_signal(ticker, asset_class, horizon, price)

    st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
    st.subheader(f"{ticker} · {asset_class.upper()} · {horizon}")

    st.markdown(f"### ➤ Действие: **{sig['action']}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Вход:** {sig['entry'] if sig['entry'] is not None else '—'}")
    with col2:
        tp = sig.get("take_profit", [])
        st.markdown("**Цели:** " + (" · ".join(f"TP{i+1} {v}" for i, v in enumerate(tp)) if tp else "—"))
    with col3:
        st.markdown(f"**Стоп:** {sig['stop'] if sig['stop'] is not None else '—'}")

    st.markdown(f"**Уверенность:** {int(round(sig.get('confidence',0)*100))}%")
    st.markdown(f"**Размер позиции:** {sig.get('position_size_pct_nav', 0)}% NAV")

    st.info(sig["narrative_ru"])

    # Альтернативы
    if sig.get("alternatives"):
        st.markdown("### Альтернативный план")
        for alt in sig["alternatives"]:
            st.markdown(
                f"- **{alt['action']}** от {alt['entry']} → "
                + " · ".join(f"TP{i+1} {v}" for i, v in enumerate(alt.get('take_profit', [])))
                + f", стоп {alt['stop']}"
            )

    st.caption("Не инвестиционный совет. Торговля сопряжена с риском.")
else:
    st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
    st.write("Выбери параметры слева и нажми **Сгенерировать сигнал**.")
