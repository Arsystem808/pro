from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime
Action = Literal["BUY","SHORT","CLOSE","WAIT"]
Horizon = Literal["intraday","swing","position"]
AssetClass = Literal["crypto","equity"]
class SignalAlternative(BaseModel):
    if_condition: str; action: Action; entry: float; take_profit: List[float]; stop: float
class Signal(BaseModel):
    id: str; ticker: str; asset_class: AssetClass; horizon: Horizon; action: Action; entry: float
    take_profit: List[float] = Field(..., min_items=1, max_items=2); stop: float
    confidence: float = Field(..., ge=0, le=1); position_size_pct_nav: float = Field(..., ge=0, le=100)
    created_at: datetime; expires_at: datetime; narrative_ru: str; alternatives: List[SignalAlternative] = []
    disclaimer: str = "Не инвестиционный совет. Торговля сопряжена с риском."
