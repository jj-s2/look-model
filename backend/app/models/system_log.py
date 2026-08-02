"""系统日志模型"""
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Index
from sqlalchemy.sql import func
from app.db.session import Base


class SystemLog(Base):
    """系统日志表"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 日志来源: EZVIZ_SDK, AI_ENGINE, WEBSOCKET, USER_ACTION, SYSTEM
    source = Column(String(50), nullable=False, index=True)
    
    # 日志级别: DEBUG, INFO, WARN, ERROR, CRITICAL
    level = Column(String(10), nullable=False, index=True)
    
    # 日志消息
    message = Column(Text, nullable=False)
    
    # 元数据（JSON格式）
    metadata_ = Column("metadata", JSON, default={})
    
    # 创建时间
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_logs_source_level', 'source', 'level'),
        Index('idx_logs_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SystemLog {self.id} [{self.level}] {self.source}>"
    
    @classmethod
    def create(cls, source: str, level: str, message: str, metadata: dict = None):
        """便捷创建方法"""
        from app.db.session import SessionLocal
        
        log = cls(
            source=source,
            level=level,
            message=message,
            metadata_=metadata or {}
        )
        
        db = SessionLocal()
        try:
            db.add(log)
            db.commit()
            db.refresh(log)
            return log
        except Exception as e:
            db.rollback()
            print(f"❌ 创建系统日志失败: {e}")
            return None
        finally:
            db.close()
