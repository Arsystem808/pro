# -*- coding: utf-8 -*-
import os, sys
from dotenv import load_dotenv; load_dotenv()

os.environ.setdefault("STRATEGY_PATH", "capintel.strategy.my_strategy_ml:generate_signal_core")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from streamlit.components.v1 import html as st_html

from capintel.signal_engine import build_signal
from capintel.providers.polygon_client import get_last_price, PolygonError
from capintel.visuals_svg import render_gauge_svg
from capintel.narrator import trader_tone_narrative_ru  # фолбэк-текст

st.set_page_config(page_title="CapIntel — AI Signals", page_icon="🤖", layout="wide")
st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. Fibo-пивоты + HA + MACD + RSI + ATR. ML включится, если есть models/meta.pkl.")

with st.sidebar:
    st.header("Параметры")
    dev_mode   = st.toggle("Режим разработчика", value=False)
    show_gauge = st.toggle("Показывать индикатор", value=True)
    st.caption(f"Активная стратегия: {os.getenv('STRATEGY_PATH','<fallback>')}")
    asset_class = st.selectbox("Класс актива", ["crypto","equity"], index=0)
    horizon     = st.selectbox("Горизонт", ["intraday","swing","position"], index=2)
    ticker      = st.text_input("Тикер", value=("BTCUSDT" if asset_class=='crypto' else "AAPL"))
    manual_price= st.text_input("Текущая цена (необязательно)", value="")
    go = st.button("Сгенерировать сигнал", type="primary", use_container_width=True)

if not go:
    st.markdown("> Выбери параметры слева и нажми **Сгенерировать сигнал**.")
    st.stop()

# 1) цена
price = None
if manual_price.strip():
    try:
        price = float(manual_price.replace(",", "."))
    except Exception:
        st.error("Неверный формат цены."); st.stop()
if price is None:
    try:
        price = float(get_last_price(asset_class, ticker))
    except PolygonError as e:
        st.warning(str(e))
        st.info("Можно ввести цену вручную в поле «Текущая цена».")
        st.stop()

# 2) сигнал
sig = build_signal(ticker, asset_class, horizon, price)

col1, col2 = st.columns([1.0, 1.15], gap="large")

def _fmt_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"

with col1:
    st.subheader(f"{sig.ticker} · {sig.asset_class.upper()} · {sig.horizon}")
    st.markdown(f"### ➤ Действие: **{sig.action}**")

    # НЕ показываем уровни для WAIT/CLOSE
    if sig.action in ("BUY", "SHORT"):
        st.markdown(
            f"**Вход:** `{_fmt_num(sig.entry)}`  \n"
            f"**Цели:** `TP1 {_fmt_num(sig.take_profit[0])}` · `TP2 {_fmt_num(sig.take_profit[1])}`  \n"
            f"**Стоп:** `{_fmt_num(sig.stop)}`  \n"
        )
    else:
        st.markdown("**Вход:** `—`  \n**Цели:** `—`  \n**Стоп:** `—`  \n")

    st.markdown(f"**Уверенность:** `{int(sig.confidence*100)}%`  \n**Размер позиции:** `{sig.position_size_pct_nav}% NAV`")

    # Комментарий — всегда; если пустой в сигнале — даём фолбэк-текст
    narr = getattr(sig, "narrative_ru", None) or trader_tone_narrative_ru(sig.action, sig.horizon, price)
    st.info(narr)

    # Альтернатива: не дублировать WAIT
    if sig.alternatives and sig.alternatives[0].get("action") in ("BUY","SHORT"):
        alt = sig.alternatives[0]
        st.markdown("**Альтернативный план**")
        st.markdown(
            f"- {alt.get('if_condition','условие')}: **{alt['action']}** от `{_fmt_num(alt['entry'])}` → "
            f"TP1 `{_fmt_num(alt['take_profit'][0])}`, TP2 `{_fmt_num(alt['take_profit'][1])}`, "
            f"стоп `{_fmt_num(alt['stop'])}`"
        )

with col2:
    # Счёт для индикатора
    score = 0.0
    if sig.action == "BUY":
        score = min(2.0, max(0.0, (sig.confidence - 0.5) / 0.4 * 2.0))
    elif sig.action == "SHORT":
        score = -min(2.0, max(0.0, (sig.confidence - 0.5) / 0.4 * 2.0))

    if show_gauge:
        try:
            prev = st.session_state.get("prev_score")
            svg  = render_gauge_svg(score, prev_score=prev, max_width=660, scale=0.85, font_scale=0.9, animate=True)
            st_html(svg, height=int(660*0.85*0.60*1.05))
            st.session_state["prev_score"] = score
        except Exception as e:
            st.warning(f"Индикатор временно недоступен: {e}")
