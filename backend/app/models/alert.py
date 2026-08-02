"""告警模型"""
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.db.session import Base


class Alert(Base):
    """告警表"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(50), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # fall_risk, health_alert
    risk_level = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    message = Column(Text, nullable=False)
    snapshot_url = Column(Text)
    
    # 告警状态: NEW, NOTIFIED, CONFIRMED, RESOLVED, FALSE_ALARM, EXPIRED, UNHANDLED_ESCALATED
    status = Column(String(30), default="NEW", nullable=False, index=True)
    
    # 升级机制
    escalation_level = Column(Integer, default=0)  # 0-正常, 1-首次升级, 2-二次升级
    escalated_at = Column(DateTime(timezone=True))
    
    # 时间记录
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    notified_at = Column(DateTime(timezone=True))
    confirmed_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    
    # 处理人和备注
    handler = Column(String(100))
    handler_note = Column(Text)
    
    # 关联的风险记录ID
    risk_record_id = Column(Integer)
    
    metadata_ = Column("metadata", JSON, default={})
    
    def __repr__(self):
        return f"<Alert {self.id} {self.alert_type} {self.status}>"
