"""系统日志Schema"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SystemLogBase(BaseModel):
    """系统日志基础Schema"""
    source: str
    level: str
    message: str
    metadata: Optional[dict] = {}


class SystemLogCreate(SystemLogBase):
    """创建系统日志"""
    pass


class SystemLogResponse(SystemLogBase):
    """系统日志响应"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True
