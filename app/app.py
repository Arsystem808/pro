import sys, os, json
from dotenv import load_dotenv; load_dotenv()
os.environ.setdefault("STRATEGY_PATH","capintel.strategy.my_strategy_ml:generate_signal_core")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)
import streamlit as st
from streamlit.components.v1 import html as st_html
from capintel.signal_engine import build_signal
from capintel.providers.polygon_client import get_last_price, PolygonError
from capintel.visuals_svg import render_gauge_svg
st.set_page_config(page_title="CapIntel — AI Signals", page_icon="🤖", layout="wide")
st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
with st.sidebar:
    st.header("Параметры")
    dev_mode = st.toggle("Режим разработчика", value=False)
    show_gauge = st.toggle("Показывать индикатор", value=True)
    st.caption(f"Активная стратегия: {os.getenv('STRATEGY_PATH','<fallback>')}")
    asset_class = st.selectbox("Класс актива", ["crypto","equity"], index=0)
    horizon = st.selectbox("Горизонт", ["intraday","swing","position"], index=1)
    ticker = st.text_input("Тикер", value=("BTCUSDT" if asset_class=='crypto' else "AAPL"))
    manual_price = st.text_input("Текущая цена (необязательно)", value="")
    go = st.button("Сгенерировать сигнал", type="primary", use_container_width=True)
if go:
    price=None
    if manual_price.strip():
        try: price=float(manual_price)
        except: st.error("Неверный формат цены."); st.stop()
    if price is None:
        try: price=float(get_last_price(asset_class, ticker))
        except PolygonError as e: st.error(str(e)); st.stop()
    sig=build_signal(ticker, asset_class, horizon, price)
    col1,col2=st.columns([1.0,1.15])
    with col1:
        st.subheader(f"{sig.ticker} · {sig.asset_class.upper()} · {sig.horizon}")
        st.markdown(f"### ➤ Действие: **{sig.action}**")
        st.markdown(f"**Вход:** `{sig.entry}`  \n**Цели:** `TP1 {sig.take_profit[0]}` · `TP2 {sig.take_profit[1]}`  \n**Стоп:** `{sig.stop}`  \n**Уверенность:** `{int(sig.confidence*100)}%`  \n**Размер позиции:** `{sig.position_size_pct_nav}% NAV`")
        st.info(sig.narrative_ru)
        alt=sig.alternatives[0]
        st.markdown("**Альтернативный план**")
        st.markdown(f"- {alt.if_condition}: **{alt.action}** от `{alt.entry}` → TP1 `{alt.take_profit[0]}`, TP2 `{alt.take_profit[1]}`, стоп `{alt.stop}`")
    with col2:
        score=0.0
        if sig.action=="BUY": score=min(2.0, max(0.0, (sig.confidence-0.5)/0.4*2.0))
        elif sig.action=="SHORT": score=-min(2.0, max(0.0, (sig.confidence-0.5)/0.4*2.0))
        if show_gauge:
            prev=st.session_state.get("prev_score")
            svg=render_gauge_svg(score, prev_score=prev, max_width=660, scale=0.85, font_scale=0.9, animate=True, duration_ms=900)
            st_html(svg, height=int(660*0.85*0.60*1.05))
            st.session_state["prev_score"]=score
else:
    st.markdown("> Выбери параметры слева и нажми **Сгенерировать сигнал**.")
