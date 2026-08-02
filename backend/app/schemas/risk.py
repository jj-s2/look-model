"""风险Schema"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class RiskBase(BaseModel):
    device_id: str
    person_id: Optional[str] = None
    risk_type: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    status: str
    reasons: list[str] = []


class RiskCreate(RiskBase):
    recorded_at: datetime
    snapshot_url: Optional[str] = None


class RiskResponse(RiskBase):
    id: int
    recorded_at: datetime
    created_at: datetime
    snapshot_url: Optional[str] = None
    
    class Config:
        from_attributes = True


class RiskSummary(BaseModel):
    device_id: str
    current_risk_score: float
    current_risk_level: str
    last_update: datetime
    recent_reasons: list[str]
