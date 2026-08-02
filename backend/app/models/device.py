"""设备模型"""
from sqlalchemy import Column, String, Boolean, DateTime, JSON, Text
from sqlalchemy.sql import func
from app.db.session import Base


class Device(Base):
    """设备表"""
    __tablename__ = "devices"
    
    id = Column(String(50), primary_key=True, index=True)
    device_serial = Column(String(100), unique=True, nullable=False, index=True)
    device_type = Column(String(20), nullable=False)  # camera, radar
    device_name = Column(String(100))
    online_status = Column(Boolean, default=False)
    capabilities = Column(JSON, default={})
    location = Column(String(200))
    last_seen_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata_ = Column("metadata", JSON, default={})
    
    def __repr__(self):
        return f"<Device {self.device_serial} ({self.device_type})>"
