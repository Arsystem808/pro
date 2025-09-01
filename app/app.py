# -*- coding: utf-8 -*-
# CapIntel — AI Signals (Crypto & Equities)

# --- bootstrap PYTHONPATH so "capintel" is importable ---
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

# Пакет приложения
try:
    from capintel.signal_engine import build_signal
except Exception as e:
    st.error(
        "Не удаётся импортировать `capintel.signal_engine`. "
        "Проверь структуру репозитория: рядом с папкой `app/` должна быть папка "
        "`capintel/` с файлами `__init__.py` и `signal_engine.py`.\n\n"
        f"Подробности: {e}"
    )
    st.stop()

# Визуализация (может отсутствовать — обработаем мягко)
_render_svg = None
try:
    from capintel.visuals_svg import render_gauge_svg as _render_svg  # type: ignore
except Exception:
    _render_svg = None

# Наратив (опционально)
_narr = None
try:
    from capintel.narrator import trader_tone_narrative_ru as _narr  # type: ignore
except Exception:
    _narr = None


# ---------------- UI helpers ----------------
def pill(value: Any) -> str:
    return f"<span style='background:#1f7a1f22;border:1px solid #39d37a44;" \
           f"padding:2px 8px;border-radius:8px;font-weight:600'>{value}</span>"


def pct_to_str(p: Optional[float]) -> str:
    if p is None:
        return "—"
    # Принимаем как 0..1 или 0..100 — нормализуем
    v = p if p > 1.001 else p * 100.0
    return f"{v:.0f}%"


def compute_score(sig: Dict[str, Any]) -> float:
    """
    Универсальная оценка [-2..+2] для шкалы.
    1) Если стратегия вернула 'score' — используем.
    2) Иначе — эвристика по действию и уверенности.
    """
    if isinstance(sig.get("score"), (int, float)):
        s = float(sig["score"])
        # безопасно обрежем
        return max(-2.0, min(2.0, s))

    action = (sig.get("action") or "").upper()
    conf = sig.get("confidence")
    if conf is None:
        conf = 0.5
    conf = conf if conf > 1.001 else conf * 100.0  # в %
    base = 0.0
    if action in ("LONG", "BUY"):
        base = 1.0
    elif action in ("SHORT", "SELL"):
        base = -1.0
    else:  # WAIT / CLOSE / прочее
        base = 0.0
    # усилим по уверенности: >70% ближе к 2, <40% ближе к 0
    k = 0.5 + max(0.0, (conf - 40.0)) / 60.0  # 0.5..1.5
    score = base * k
    return max(-2.0, min(2.0, score))


def show_gauge(sig: Dict[str, Any]) -> None:
    score = compute_score(sig)
    if _render_svg is None:
        st.caption("Индикатор недоступен: модуль визуализации не найден.")
        return
    # Пытаемся вызвать с разными сигнатурами, чтобы не падать,
    # т.к. у тебя могла быть другая версия visuals_svg.
    svg = None
    try:
        svg = _render_svg(score)  # самый простой вызов
    except TypeError:
        try:
            svg = _render_svg(score, title="Общая оценка")  # вариант с заголовком
        except Exception:
            svg = None
    if svg:
        st.markdown(
            f"<div style='display:flex;justify-content:center;'>{svg}</div>",
            unsafe_allow_html=True,
        )


# ---------------- Page config ----------------
st.set_page_config(
    page_title="CapIntel — AI Signals",
    layout="wide",
)

st.markdown("## 🤖 CapIntel — AI Signals (Crypto & Equities)")
st.caption("Meta-labeling поверх правил. ML включится, если найден `models/meta.pkl`.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Параметры")
    asset_class = st.selectbox("Класс актива", ["equity", "crypto"], index=0)
    horizon = st.selectbox("Горизонт", ["intraday", "swing", "position"], index=1)
    ticker = st.text_input("Тикер", value="AAPL" if asset_class == "equity" else "BTCUSD")
    price_in = st.text_input("Текущая цена (необязательно)", value="").strip()
    show_g = st.toggle("Показывать индикатор", value=True)
    dev = st.toggle("Режим разработчика", value=False)
    submitted = st.button("Сгенерировать сигнал", type="primary")

# ---------------- Main logic ----------------
if submitted:
    if not ticker:
        st.warning("Введите тикер.")
        st.stop()

    # Парсим цену
    price_val: Optional[float] = None
    if price_in:
        try:
            price_val = float(price_in.replace(",", "."))
        except Exception:
            st.warning("Не удалось разобрать поле 'Текущая цена'. Будет использована рыночная цена провайдера.")
            price_val = None

    # Генерируем сигнал
    try:
        sig: Dict[str, Any] = build_signal(
            ticker=ticker.strip().upper(),
            asset_class=asset_class.strip().lower(),
            horizon=horizon.strip().lower(),
            price=price_val,
        )
    except ValueError as ve:
        st.error(f"Ошибка при генерации сигнала: {ve}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Заголовок инструмента
    st.markdown(
        f"### {ticker.strip().upper()} • {asset_class.upper()} • {horizon.lower()}",
    )

    # --- Линия ключевых полей ---
    cols = st.columns(3)
    def _fmt_num(x: Any) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "—"

    entry = sig.get("entry")
    stop = sig.get("stop")
    tps: List[float] = []
    if isinstance(sig.get("take_profit"), (list, tuple)):
        tps = [float(x) for x in sig["take_profit"] if x is not None]

    with cols[0]:
        st.markdown(f"**Действие:** {sig.get('action','—')}")
        st.markdown(f"**Вход:** {_fmt_num(entry)}")
    with cols[1]:
        if tps:
            tp_txt = " • ".join([f"TP{i+1} {_fmt_num(tp)}" for i, tp in enumerate(tps[:2])])
        else:
            tp_txt = "—"
        st.markdown(f"**Цели:** {tp_txt}")
    with cols[2]:
        st.markdown(f"**Стоп:** {_fmt_num(stop)}")

    # --- Уверенность/размер ---
    cols2 = st.columns(2)
    with cols2[0]:
        st.markdown(f"**Уверенность:** {pct_to_str(sig.get('confidence'))}")
    with cols2[1]:
        sz = sig.get("position_size_pct_nav")
        st.markdown(f"**Размер позиции:** {pct_to_str(sz)} NAV")

    # --- Шкала ---
    if show_g:
        show_gauge(sig)

    # --- Наратив/заметка о ML ---
    ml_note = sig.get("ml_note")
    if not ml_note:
        # если модель нашлась — покажем ON, иначе OFF
        ml_note = "[ML ON] Модель найдена." if sig.get("p_succ") is not None else "[ML OFF] Модель не найдена — используется rule-based логика."
    st.info(ml_note)

    # Если стратегия не проставила narrative_ru — попробуем сгенерировать тонко
    nar = sig.get("narrative_ru")
    if not nar and _narr:
        try:
            nar = _narr(sig)
        except Exception:
            nar = None
    if nar:
        st.info(nar)

    # --- Альтернативный план ---
    alts = sig.get("alternatives") or []
    if isinstance(alts, list) and alts:
        st.subheader("Альтернативный план")
        for a in alts:
            a_act = (a.get("action") or "—").upper()
            a_entry = _fmt_num(a.get("entry"))
            a_tps = a.get("take_profit") or []
            a_stop = _fmt_num(a.get("stop"))
            a_tp_txt = " • ".join([f"TP{i+1} {_fmt_num(tp)}" for i, tp in enumerate(a_tps[:2])]) if a_tps else "—"
            cond = a.get("if_condition")
            bullet = f"**{a_act}** от {pill(a_entry)} → {pill(a_tp_txt)}, стоп {pill(a_stop)}"
            if cond:
                bullet = f"{cond}: " + bullet
            st.markdown("• " + bullet, unsafe_allow_html=True)

    # --- Служебные поля (для дебага/аудита) ---
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.caption(f"Создан: {sig.get('created_at','—')}")
    with meta_cols[1]:
        st.caption(f"Истекает: {sig.get('expires_at','—')}")
    with meta_cols[2]:
        st.caption(f"ID: {sig.get('id','—')}")

    st.caption("Не инвестиционный совет. Торговля сопряжена с риском.")

    with st.expander("JSON"):
        st.code(json.dumps(sig, ensure_ascii=False, indent=2))

else:
    st.info("Выбери параметры слева и нажми **Сгенерировать сигнал**.")
