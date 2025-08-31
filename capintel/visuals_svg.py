# -*- coding: utf-8 -*-
import math

def _arc_path(cx, cy, r, start_deg, end_deg):
    s = math.radians(start_deg)
    e = math.radians(end_deg)
    x1, y1 = cx + r*math.cos(s), cy + r*math.sin(s)
    x2, y2 = cx + r*math.cos(e), cy + r*math.sin(e)
    large = 1 if abs(end_deg - start_deg) > 180 else 0
    return "M {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 {large} 1 {x2:.2f},{y2:.2f}".format(
        x1=x1, y1=y1, r=r, large=large, x2=x2, y2=y2
    )

def render_gauge_svg(
    score, prev_score=None, max_width=660, scale=0.85, font_scale=0.9,
    dark_bg="#0E1117", animate=False, duration_ms=900
):
    score = max(-2.0, min(2.0, float(score)))
    if prev_score is None:
        prev_score = score
        animate = False

    W = int(max_width * scale)
    H = int(W * 0.60)
    cx, cy, R = W/2.0, H*0.87, W*0.40

    def to_angle(s):
        return -180 + 180 * (s + 2.0) / 4.0

    end_ang   = to_angle(score)
    arc_bg  = _arc_path(cx, cy, R, -180, 0)
    outline = _arc_path(cx, cy, R+2, -180, 0)

    ticks = [(-180, "−2"), (-135, "−1"), (-90, "0"), (-45, "+1"), (0, "+2")]
    tick_lines, tick_texts = [], []
    for a, lab in ticks:
        rad = math.radians(a)
        x1 = cx + (R-10)*math.cos(rad); y1 = cy + (R-10)*math.sin(rad)
        x2 = cx + (R+3 )*math.cos(rad); y2 = cy + (R+3 )*math.sin(rad)
        xt = cx + (R+38)*math.cos(rad); yt = cy + (R+38)*math.sin(rad)
        tick_lines.append(
            '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#ffffff" stroke-width="1.4"/>'
            .format(x1,y1,x2,y2)
        )
        tick_texts.append(
            '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" class="tick t">{}</text>'
            .format(xt,yt,lab)
        )

    fs_title  = int(W*0.048*font_scale)
    fs_status = int(W*0.036*font_scale)
    fs_tick   = int(W*0.028*font_scale)

    status = "Нейтрально"
    if score > 1.0:
        status = "Активно покупать"
    elif score > 0.15:
        status = "Покупать"
    elif score < -1.0:
        status = "Активно продавать"
    elif score < -0.15:
        status = "Продавать"

    ax = cx + (R-6)*math.cos(math.radians(end_ang))
    ay = cy + (R-6)*math.sin(math.radians(end_ang))

    parts = []
    parts.append('<div style="max-width:{}px;width:100%;margin:0 auto;">'.format(W))
    parts.append(
        '<svg viewBox="0 0 {W} {H}" width="100%" height="auto" '
        'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
        'style="background:{bg}; border-radius:12px">'.format(W=W, H=H, bg=dark_bg)
    )

    # без форматирования с %, чтобы не ловить ошибки
    parts.append(
        '<defs>'
        '  <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">'
        '    <stop offset="0%"   stop-color="#FF8C00"/>'
        '    <stop offset="40%"  stop-color="#FFE066"/>'
        '    <stop offset="70%"  stop-color="#40E0D0"/>'
        '    <stop offset="100%" stop-color="#32CD32"/>'
        '  </linearGradient>'
        '</defs>'
        '<style>'
        '  .t { fill:#ffffff; font-family:-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }'
        '  .h1 { font-weight:700; font-size:' + str(fs_title)  + 'px; }'
        '  .h2 { font-weight:700; font-size:' + str(fs_status) + 'px; }'
        '  .tick { opacity:0.98; font-size:' + str(fs_tick)   + 'px; }'
        '</style>'
    )

    parts.append('<path d="{d}" stroke="url(#gaugeGradient)" stroke-width="{w}" '
                 'stroke-linecap="round" fill="none"/>'.format(d=arc_bg, w=max(2, int(W*0.036))))
    parts.append('<path d="{d}" stroke="#FFFFFF" stroke-opacity="0.92" stroke-width="2" fill="none"/>'
                 .format(d=outline))

    parts.append("".join(tick_lines))
    parts.append("".join(tick_texts))

    parts.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" class="t h1">Общая оценка</text>'
                 .format(W/2.0, H*0.06))

    parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#ffffff" '
                 'stroke-width="6" stroke-linecap="round"/>'.format(cx, cy, ax, ay))
    parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="7" fill="#ffffff" />'.format(cx, cy))

    parts.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" class="t h2">{}</text>'
                 .format(W/2.0, H*0.95, status))

    parts.append("</svg></div>")
    return "".join(parts)
