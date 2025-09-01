import os
import streamlit as st
from datetime import datetime, timezone
from capintel.signal_engine import build_signal
from capintel.visuals_svg import render_gauge_svg

st.set_page_config(page_title="CapIntel — AI Signals", layout="wide")

st.sidebar.header("Параметры")
asset_class = st.sidebar.selectbox("Класс актива", ["equity", "crypto"], index=0)
horizon     = st.sidebar.selectbox("Горизонт", ["intraday","swing","position"], index=1)
ticker      = st.sidebar.text_input("Тикер", "AAPL" if asset_class=="equity" else "BTCUSD")
price_str   = st.sidebar.text_input("Текущая цена (опционально)", "")
show_gauge  = st.sidebar.toggle("Показывать индикатор", value=True)

st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если найден models/meta.pkl.")

# Кнопка
if st.sidebar.button("Сгенерировать сигнал"):
    try:
        price = float(price_str) if price_str.strip() else None
    except:
        price = None

    try:
        sig = build_signal(ticker, asset_class, horizon, price)
    except Exception as e:
        st.error(f"Ошибка при генерации сигнала:\n\n{type(e).__name__}: {e}")
    else:
        # Заголовок
        st.subheader(f"{ticker.upper()} · {asset_class.upper()} · {horizon}")
        st.markdown(f"### ➤ Действие: **{sig['action']}**")

        # Референсные числа
        cols = st.columns(3)
        cols[0].markdown(f"**Вход:** {sig.get('entry', '—')}")
        tps = sig.get('take_profit', [])
        if tps:
            cols[1].markdown("**Цели:** " + " · ".join([f"TP{i+1} {v:.2f}" for i,v in enumerate(tps)]))
        else:
            cols[1].markdown("**Цели:** —")
        cols[2].markdown(f"**Стоп:** {sig.get('stop','—')}")

        cols = st.columns(2)
        cols[0].markdown(f"**Уверенность:** {int(round(sig.get('confidence',0)*100))}%")
        pos = sig.get('position_size_pct_nav', None)
        cols[1].markdown(f"**Размер позиции:** {f'{pos:.1f}% NAV' if pos is not None else '—% NAV'}")

        # Синяя карточка с комментариями
        st.info(sig.get("narrative_ru",""))

        # Шкала-настроение
        if show_gauge:
            # sentiment: -2..+2 (SHORT -> отриц., BUY -> полож., WAIT/CLOSE -> 0)
            act = sig["action"].upper()
            sentiment = {"BUY": +1.2, "SHORT": -1.2, "WAIT": 0.0, "CLOSE": 0.0}.get(act, 0.0)
            svg = render_gauge_svg(sentiment=sentiment, title="Общая оценка")
            st.write(svg, unsafe_allow_html=True)

        # Альтернативный план
        alts = sig.get("alternatives", [])
        if alts:
            st.subheader("Альтернативный план")
            for alt in alts:
                a = alt.get("action","")
                en = alt.get("entry","—")
                tps = alt.get("take_profit",[])
                stop = alt.get("stop","—")
                cond = alt.get("if_condition","")
                line = f"- **{cond}** → **{a}** от **{en}**"
                if tps:
                    line += " → " + " · ".join([f"TP{i+1} {v:.2f}" for i,v in enumerate(tps)])
                line += f", **стоп {stop}**"
                st.markdown(line)

        # Техническая плашка
        st.caption(sig.get("tech_note_ru",""))

else:
    st.write("Выбери параметры слева и нажми **Сгенерировать сигнал**.")
