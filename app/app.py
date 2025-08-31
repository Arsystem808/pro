# app/app.py
import os
import sys
from typing import Any, Dict, List

# --- чтобы импортировался локальный пакет capintel ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from capintel.signal_engine import build_signal


def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.2f}".replace(",", " ")
    except Exception:
        return str(v)


st.set_page_config(page_title="CapIntel — AI Signals", layout="wide")

st.sidebar.header("Параметры")
asset_class = st.sidebar.selectbox("Класс актива", ["equity", "crypto"], index=0)
horizon     = st.sidebar.selectbox("Горизонт", ["intraday", "swing", "position"], index=1)
ticker      = st.sidebar.text_input("Тикер", "AAPL").upper().strip()
price_in    = st.sidebar.text_input("Текущая цена (опционально)", "")
price       = None
if price_in.strip():
    try:
        price = float(price_in.replace(",", "."))
    except ValueError:
        st.sidebar.error("Цена должна быть числом (например, 232.10)")

run = st.sidebar.button("Сгенерировать сигнал")

st.title("🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если найден models/meta.pkl.")

if not run:
    st.write("Выбери параметры слева и нажми **Сгенерировать сигнал**.")
    st.stop()

# --- Генерация сигнала ---
try:
    sig: Dict[str, Any] = build_signal(ticker, asset_class, horizon, price)
except Exception as e:
    st.error("Ошибка при генерации сигнала:")
    st.exception(e)
    st.stop()

# --- Шапка ---
st.subheader(f"{ticker} · {asset_class.upper()} · {horizon}")

st.markdown(f"### ➤ Действие: **{sig.get('action','WAIT')}**")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Вход:** {_fmt_num(sig.get('entry'))}")
with c2:
    tps: List[float] = sig.get("take_profit") or []
    tps_str = " · ".join([f"TP{i+1} {_fmt_num(v)}" for i, v in enumerate(tps)]) if tps else "—"
    st.markdown(f"**Цели:** {tps_str}")
with c3:
    st.markdown(f"**Стоп:** {_fmt_num(sig.get('stop'))}")

# --- Метрики уверенности / размер ---
conf = float(sig.get("confidence", 0.60))
st.markdown(f"**Уверенность:** {int(round(conf*100))}%")
st.markdown(f"**Размер позиции:** {_fmt_num(sig.get('position_size_pct_nav'))}% NAV")

# --- ML статус ---
ml = sig.get("ml") or {}
if ml.get("on") and ml.get("p_succ") is not None:
    st.success(f"[ML ON] Вероятность успеха сделки p_succ≈{ml['p_succ']:.2f}")
elif ml.get("on") is False:
    st.info("[ML OFF] Модель не найдена — используется rule-based логика.")

# --- Нарратив ---
if sig.get("narrative_ru"):
    st.info(sig["narrative_ru"])

# --- Альтернативный план ---
alts = sig.get("alternatives") or []
if alts:
    st.markdown("### Альтернативный план")
    for alt in alts:
        tp_str = " · ".join([f"TP{i+1} {_fmt_num(v)}" for i, v in enumerate(alt.get("take_profit", []))]) or "—"
        st.markdown(
            f"- **{alt.get('action','WAIT')}** от {_fmt_num(alt.get('entry'))} → {tp_str}, стоп {_fmt_num(alt.get('stop'))}"
            + (" _(условный)_"
               if alt.get("conditional") else "")
        )

# --- Метаданные / JSON ---
cA, cB = st.columns(2)
with cA:
    st.caption(
        f"Создан: {sig.get('created_at','—')} · Истекает: {sig.get('expires_at','—')} · ID: {sig.get('id','—')}"
    )
    st.caption("Не инвестиционный совет. Торговля сопряжена с риском.")
with cB:
    with st.expander("JSON"):
        import json
        st.code(json.dumps(sig, ensure_ascii=False, indent=2), language="json")
