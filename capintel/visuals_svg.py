import math
def _arc_path(cx, cy, r, start_deg, end_deg):
    import math
    s, e = math.radians(start_deg), math.radians(end_deg)
    x1, y1 = cx + r*math.cos(s), cy + r*math.sin(s)
    x2, y2 = cx + r*math.cos(e), cy + r*math.sin(e)
    large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
    return f"M {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f}"
def render_gauge_svg(score: float, prev_score: float | None = None, max_width: int = 660,
                     scale: float = 0.85, font_scale: float = 0.9, dark_bg: str = "#0E1117",
                     animate: bool = True, duration_ms: int = 900) -> str:
    score = max(-2.0, min(2.0, float(score)))
    if prev_score is None: animate=False; prev_score=score
    W = int(max_width*scale); H = int(W*0.60); cx,cy,R = W/2, H*0.87, W*0.40
    def to_angle(s: float): return -180 + 180*(s+2.0)/4.0
    start_ang, end_ang = to_angle(prev_score), to_angle(score)
    status = "Нейтрально"
    if score>1.0: status="Активно покупать"
    elif score>0.15: status="Покупать"
    elif score<-1.0: status="Активно продавать"
    elif score<-0.15: status="Продавать"
    arc=_arc_path(cx,cy,R,-180,0)
    outline=_arc_path(cx,cy,R+2,-180,0)
    ticks=[(-180,"−2"),(-135,"−1"),(-90,"0"),(-45,"+1"),(0,"+2")]
    lines=[]; texts=[]
    for a,lab in ticks:
        import math
        ax1,ay1 = cx + (R-10)*math.cos(math.radians(a)), cy + (R-10)*math.sin(math.radians(a))
        ax2,ay2 = cx + (R+3)*math.cos(math.radians(a)),  cy + (R+3)*math.sin(math.radians(a))
        tx,ty   = cx + (R+38)*math.cos(math.radians(a)), cy + (R+38)*math.sin(math.radians(a))
        lines.append(f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" stroke="#fff" stroke-width="1.4"/>
')
        texts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" class="t tick">{lab}</text>')
    fs_title=int(W*0.048*font_scale); fs_status=int(W*0.036*font_scale); fs_tick=int(W*0.028*font_scale)
    return f"""
<div style="max-width:{W}px;width:100%;margin:0 auto;">
  <svg viewBox="0 0 {W} {H}" width="100%" height="auto" preserveAspectRatio="xMidYMid meet"
       xmlns="http://www.w3.org/2000/svg" style="background:{dark_bg}; border-radius:12px">
    <style>
      .t {{ fill:#fff; font-family:-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
      .h1 {{ font-weight:700; font-size:{fs_title}px; }}
      .h2 {{ font-weight:700; font-size:{fs_status}px; }}
      .tick {{ opacity:0.98; font-size:{fs_tick}px; }}
    </style>
    <path d="{arc}" stroke="#7CFC00" stroke-width="{max(2,int(W*0.036))}" stroke-linecap="round" fill="none"/>
    <path d="{outline}" stroke="#FFFFFF" stroke-opacity="0.92" stroke-width="2" fill="none"/>
    {''.join(lines)}{''.join(texts)}
    <text x="{W/2}" y="{H*0.06}" text-anchor="middle" class="t h1">Общая оценка</text>
    <text x="{W/2}" y="{H*0.95}" text-anchor="middle" class="t h2">{status}</text>
  </svg>
</div>
"""
