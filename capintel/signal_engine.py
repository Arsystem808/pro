import os, importlib
from typing import Callable, Dict, Any
from datetime import datetime, timedelta
from capintel.schemas import Signal, SignalAlternative, AssetClass, Horizon
from capintel.risk import target_vol_position_size, sanitize_levels
from capintel.narrator import trader_tone_narrative_ru
from capintel.providers.polygon_client import get_agg_bars
def _load_custom_strategy() -> Callable | None:
    path=os.getenv("STRATEGY_PATH","").strip()
    if not path: return None
    try:
        module_path, func_name = path.split(":"); mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    except Exception: return None
def _horizon_params(h: Horizon):
    return {"intraday":(25,8),"swing":(60,48),"position":(200,7*24)}[h]
def build_signal(ticker: str, asset_class: AssetClass, horizon: Horizon, last_price: float) -> Signal:
    custom = _load_custom_strategy(); now=datetime.utcnow(); buffer_bp, expire_h = _horizon_params(horizon)
    if custom is not None:
        try: bars=get_agg_bars(asset_class,ticker,horizon,limit=400)
        except Exception: bars=None
        spec: Dict[str,Any] = custom(ticker, asset_class, horizon, last_price, bars)
        action=spec["action"]; entry=float(spec["entry"])
        tp1,tp2=[float(x) for x in spec["take_profit"][:2]]; stop=float(spec["stop"]); conf=float(spec["confidence"])
        tp1,tp2,stop=sanitize_levels(action, entry, tp1, tp2, stop); size=target_vol_position_size(conf, asset_class, horizon)
        alt_d=spec.get("alt") or {}; alt=SignalAlternative(if_condition=alt_d.get("if_condition","n/a"),
                    action=alt_d.get("action",action), entry=float(alt_d.get("entry", entry)),
                    take_profit=[float(x) for x in (alt_d.get("take_profit") or [tp1,tp2])],
                    stop=float(alt_d.get("stop", stop)))
        narrative=str(spec.get("narrative_ru") or trader_tone_narrative_ru(action, horizon, last_price))
        return Signal(id=f"{ticker}-{now.strftime('%Y%m%d%H%M%S')}-{horizon}", ticker=ticker.upper(), asset_class=asset_class, horizon=horizon,
                      action=action, entry=entry, take_profit=[tp1,tp2], stop=stop, confidence=conf, position_size_pct_nav=size,
                      created_at=now, expires_at=now+timedelta(hours=expire_h), narrative_ru=narrative, alternatives=[alt])
    entry=float(last_price); tp1=entry*1.002; tp2=entry*1.004; stop=entry*0.998; conf=0.55
    size=target_vol_position_size(conf, asset_class, horizon)
    alt=SignalAlternative(if_condition="если цена подтвердит сценарий", action="WAIT", entry=entry, take_profit=[tp1,tp2], stop=stop)
    return Signal(id=f"{ticker}-{now.strftime('%Y%m%d%H%M%S')}-{horizon}", ticker=ticker.upper(), asset_class=asset_class, horizon=horizon,
                  action="WAIT", entry=entry, take_profit=[tp1,tp2], stop=stop, confidence=conf, position_size_pct_nav=size,
                  created_at=now, expires_at=now+timedelta(hours=expire_h), narrative_ru=trader_tone_narrative_ru("WAIT", horizon, last_price), alternatives=[alt])
