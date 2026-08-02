"""健康数据Schema"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class HealthBase(BaseModel):
    device_id: str
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    body_temperature: Optional[float] = None
    in_bed: Optional[bool] = None
    sleep_duration: Optional[float] = None
    leave_bed_count: Optional[int] = None


class HealthCreate(HealthBase):
    recorded_at: Optional[datetime] = None


class HealthResponse(HealthBase):
    id: int
    recorded_at: datetime
    created_at: datetime
    sleep_quality_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class HealthTrend(BaseModel):
    device_id: str
    period: str  # 24h, 7d, 30d
    avg_heart_rate: Optional[float] = None
    avg_respiratory_rate: Optional[float] = None
    avg_sleep_duration: Optional[float] = None
    total_leave_bed_count: Optional[int] = None
