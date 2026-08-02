"""告警超时升级服务"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.alert import Alert
from app.services.websocket_manager import websocket_manager
from app.db.session import SessionLocal


class AlertEscalationService:
    """告警超时升级服务
    
    监控高危告警，若在指定时间内未处理，自动升级并强提醒
    """
    
    def __init__(self, escalation_timeout_minutes: int = 5):
        """
        Args:
            escalation_timeout_minutes: 升级超时时间（分钟），默认5分钟
        """
        self.escalation_timeout_minutes = escalation_timeout_minutes
    
    async def check_and_escalate(self):
        """检查并升级未处理的高危告警"""
        db: Session = SessionLocal()
        try:
            escalated_count = 0
            
            # 查询需要升级的告警
            timeout_threshold = datetime.now() - timedelta(minutes=self.escalation_timeout_minutes)
            
            # 查询高危告警：状态为NEW或NOTIFIED，且超时未处理
            unhandled_alerts = db.query(Alert).filter(
                Alert.risk_level.in_(["HIGH", "CRITICAL"]),
                Alert.status.in_(["NEW", "NOTIFIED"]),
                Alert.created_at < timeout_threshold,
                Alert.escalation_level < 2  # 最多升级2次
            ).all()
            
            for alert in unhandled_alerts:
                # 升级告警
                alert.escalation_level += 1
                alert.escalated_at = datetime.now()
                
                if alert.escalation_level >= 2:
                    alert.status = "UNHANDLED_ESCALATED"
                
                db.commit()
                
                # 推送升级通知
                await self._broadcast_escalation(alert)
                
                escalated_count += 1
                print(f"⚠️ 告警升级: Alert#{alert.id} 升级到 Level {alert.escalation_level}")
            
            if escalated_count > 0:
                print(f"📢 共升级 {escalated_count} 条未处理告警")
            
            return escalated_count
            
        except Exception as e:
            print(f"❌ 告警升级检查失败: {e}")
            db.rollback()
            return 0
        finally:
            db.close()
    
    async def _broadcast_escalation(self, alert: Alert):
        """广播告警升级消息"""
        escalation_messages = {
            1: "⚠️ 紧急提醒：告警已超时5分钟未处理！",
            2: "🚨 严重警告：告警已超时10分钟未处理，请立即处置！"
        }
        
        message = escalation_messages.get(alert.escalation_level, "告警需要处理")
        
        await websocket_manager.broadcast({
            "type": "alert_escalation",
            "data": {
                "id": alert.id,
                "device_id": alert.device_id,
                "alert_type": alert.alert_type,
                "risk_level": alert.risk_level,
                "escalation_level": alert.escalation_level,
                "message": message,
                "original_message": alert.message,
                "snapshot_url": alert.snapshot_url,
                "created_at": alert.created_at.isoformat(),
                "escalated_at": alert.escalated_at.isoformat() if alert.escalated_at else None,
            }
        })


# 全局实例
alert_escalation_service = AlertEscalationService(escalation_timeout_minutes=5)
