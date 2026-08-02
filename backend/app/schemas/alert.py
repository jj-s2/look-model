"""告警Schema"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class AlertBase(BaseModel):
    device_id: str
    alert_type: str
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    message: str


class AlertCreate(AlertBase):
    snapshot_url: Optional[str] = None
    risk_record_id: Optional[int] = None


class AlertUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern="^(CONFIRMED|RESOLVED|FALSE_ALARM|EXPIRED)$")
    handler: Optional[str] = None
    handler_note: Optional[str] = None


class AlertResponse(AlertBase):
    id: int
    status: str
    snapshot_url: Optional[str] = None
    created_at: datetime
    confirmed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    handler: Optional[str] = None
    handler_note: Optional[str] = None
    
    class Config:
        from_attributes = True


class AlertConfirm(BaseModel):
    handler: str
    note: Optional[str] = None


class AlertResolve(BaseModel):
    handler: str
    note: str
