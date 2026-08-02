"""健康数据模型"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from app.db.session import Base


class HealthRecord(Base):
    """健康记录表（来自SDNL1雷达）"""
    __tablename__ = "health_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(50), nullable=False, index=True)
    
    # 生理数据
    heart_rate = Column(Integer)  # 心率 bpm
    respiratory_rate = Column(Integer)  # 呼吸率 次/分钟
    
    # 状态数据
    in_bed = Column(Boolean)  # 是否在床
    sleep_duration = Column(Float)  # 睡眠时长（小时）
    leave_bed_count = Column(Integer)  # 离床次数
    
    # 睡眠质量评分（可选）
    sleep_quality_score = Column(Float)
    
    # 时间
    recorded_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    metadata_ = Column("metadata", JSON, default={})
    
    def __repr__(self):
        return f"<HealthRecord {self.device_id} HR:{self.heart_rate} RR:{self.respiratory_rate}>"
