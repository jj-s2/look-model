"""风险记录模型"""
from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Integer
from sqlalchemy.sql import func
from app.db.session import Base


class RiskRecord(Base):
    """风险记录表"""
    __tablename__ = "risk_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(50), nullable=False, index=True)
    person_id = Column(String(50))
    risk_type = Column(String(50), nullable=False)  # fall_risk, health_abnormal
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    status = Column(String(50))  # walking_unstable, sitting, lying, etc.
    reasons = Column(JSON, default=[])
    snapshot_url = Column(Text)
    recorded_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata_ = Column("metadata", JSON, default={})
    
    def __repr__(self):
        return f"<RiskRecord {self.device_id} {self.risk_level} {self.risk_score}>"
