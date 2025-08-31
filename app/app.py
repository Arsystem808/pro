# -*- coding: utf-8 -*-
import os, sys
from dotenv import load_dotenv; load_dotenv()

# ВАЖНО: правильный путь к стратегии
os.environ.setdefault("STRATEGY_PATH", "capintel.strategy.my_strategy_ml:generate_signal_core")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from streamlit.components.v1 import html as st_html

from capintel.signal_engine import build_signal
from capintel.providers.polygon_client import get_last_price, PolygonError
from capintel.visuals_svg import render_gauge_svg
from capintel.narrator import trader_tone_narrative_ru

st.set_page_config(page_title="CapIntel — AI Signals", page_icon="🤖", layout="wide")
st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если есть models/meta.pkl.")

with st.sidebar:
    st.header("Параметры")
    dev_mode   = st.toggle("Режим разработчика", value=False)
    show_gauge = st.toggle("Показывать индикатор", value=True)
    st.caption(f"Активная стратегия: {os.getenv('STRATEGY_PATH','<fallback>')}")
    asset_class = st.selectbox("Класс актива", ["crypto","equity"], index=0)
    horizon     = st.selectbox("Горизонт", ["intraday","swing","position"], index=1)
    ticker      = st.text_input("Тикер", value=("BTCUSDT" if asset_class=='crypto' else "AAPL"))
    manual_price= st.text_input("Текущая цена (необязательно)", value="")
    go = st.button("Сгенерировать сигнал", type="primary", use_container_width=True)

if not go:
    st.markdown("> Выбери параметры слева и нажми **Сгенерировать сигнал**.")
    st.stop()

# ---- цена
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

# ---- сигнал
sig = build_signal(ticker, asset_class, horizon, price)

col1, col2 = st.columns([1.0, 1.15], gap="large")

def _fmt(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"

with col1:
    st.subheader(f"{sig.ticker} · {sig.asset_class.upper()} · {sig.horizon}")
    st.markdown(f"### ➤ Действие: **{sig.action}**")

    show_levels = sig.action in ("BUY","SHORT")
    if show_levels:
        st.markdown(
            f"**Вход:** `{_fmt(sig.entry)}`  \n"
            f"**Цели:** `TP1 {_fmt(sig.take_profit[0])}` · `TP2 {_fmt(sig.take_profit[1])}`  \n"
            f"**Стоп:** `{_fmt(sig.stop)}`  \n"
        )
        st.markdown(
            f"**Уверенность:** `{int(sig.confidence*100)}%`  \n"
            f"**Размер позиции:** `{sig.position_size_pct_nav}% NAV`"
        )
    else:
        # при WAIT/CLOSE уровни/уверенность/размер скрываем
        st.markdown("**Вход:** `—`  \n**Цели:** `—`  \n**Стоп:** `—`  \n")
        st.markdown("**Уверенность:** `—`  \n**Размер позиции:** `—`")

    # Комментарий — всегда (если пуст в сигнале, подставим живой фолбэк)
    narr = getattr(sig, "narrative_ru", None) or trader_tone_narrative_ru(sig.action, sig.horizon, price)
    st.info(narr)

    # Альтернативный план — безопасный доступ (dict или объект), и не дублируем WAIT
    alt = None
    alts = getattr(sig, "alternatives", None)
    if isinstance(alts, list) and len(alts) > 0:
        alt = alts[0]
    if alt is not None:
        # допускаем и dict, и модель
        a_action = getattr(alt, "action", None) or (alt.get("action") if isinstance(alt, dict) else None)
        if a_action in ("BUY","SHORT"):
            a_if   = getattr(alt, "if_condition", None) or (alt.get("if_condition") if isinstance(alt, dict) else "")
            a_ent  = getattr(alt, "entry", None)        or (alt.get("entry") if isinstance(alt, dict) else None)
            a_tp   = getattr(alt, "take_profit", None)  or (alt.get("take_profit") if isinstance(alt, dict) else [None,None])
            a_stop = getattr(alt, "stop", None)         or (alt.get("stop") if isinstance(alt, dict) else None)
            st.markdown("**Альтернативный план**")
            st.markdown(
                f"- {a_if or 'условие'}: **{a_action}** от `{_fmt(a_ent)}` → "
                f"TP1 `{_fmt(a_tp[0])}`, TP2 `{_fmt(a_tp[1])}`, стоп `{_fmt(a_stop)}`"
            )

with col2:
    # Индикатор — только для BUY/SHORT имеет смысл. Для WAIT — нейтральная стрелка ≈ 0.
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
