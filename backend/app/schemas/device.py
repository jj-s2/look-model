"""设备Schema"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DeviceBase(BaseModel):
    device_serial: str
    device_type: str = Field(..., pattern="^(camera|radar)$")
    device_name: Optional[str] = None
    location: Optional[str] = None


class DeviceCreate(DeviceBase):
    pass


class DeviceUpdate(BaseModel):
    device_name: Optional[str] = None
    location: Optional[str] = None
    online_status: Optional[bool] = None


class DeviceResponse(DeviceBase):
    id: str
    online_status: bool
    capabilities: dict
    last_seen_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class DeviceStatus(BaseModel):
    device_id: str
    online: bool
    last_seen: Optional[datetime]
    capabilities: dict
