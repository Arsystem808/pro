# -*- coding: utf-8 -*-
def render_gauge_svg(score: float = 0.0, title: str = "Оценка") -> str:
    """
    score: -2..+2 -> шкала. 0 — нейтрально
    """
    s = max(-2.0, min(2.0, float(score)))
    # 180° дуга: позиция стрелки
    angle = 180 * (s + 2) / 4
    x = 150 + 110 * __import__("math").cos(__import__("math").radians(180 - angle))
    y = 180 - 110 * __import__("math").sin(__import__("math").radians(180 - angle))
    return f"""
<div style="width:320px">
<svg viewBox="0 0 300 200">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#f39c12"/>
      <stop offset="50%"  stop-color="#f1c40f"/>
      <stop offset="100%" stop-color="#2ecc71"/>
    </linearGradient>
  </defs>
  <path d="M30,180 A120,120 0 0,1 270,180" fill="none" stroke="url(#g)" stroke-width="18"/>
  <line x1="150" y1="180" x2="{x:.1f}" y2="{y:.1f}" stroke="#fff" stroke-width="4"/>
  <circle cx="150" cy="180" r="6" fill="#fff"/>
  <text x="150" y="105" text-anchor="middle" fill="#fff" font-size="14">{title}</text>
  <text x="150" y="195" text-anchor="middle" fill="#bbb" font-size="12">Нейтрально</text>
</svg>
</div>
"""
