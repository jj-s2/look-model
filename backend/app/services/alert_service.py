"""告警服务"""
from sqlalchemy.orm import Session
from app.models.alert import Alert
from app.models.risk import RiskRecord
from app.schemas.alert import AlertCreate
from app.services.websocket_manager import websocket_manager
from app.services.event_deduplicator import event_deduplicator
from app.services.ezviz_service import ezviz_service
from app.config import get_settings


class AlertService:
    """告警业务逻辑"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def check_and_create_alert(self, risk: RiskRecord, db: Session) -> Alert | None:
        """检查风险是否需要创建告警"""
        # 根据风险等级判断是否需要告警
        if risk.risk_level == "LOW":
            return None
        
        # 事件防抖去重检查
        if not event_deduplicator.should_process(risk.device_id, risk.risk_type, risk.risk_level):
            print(f"⏭️ 跳过重复事件: {risk.device_id} - {risk.risk_type} - {risk.risk_level}")
            return None
        
        # 检查是否有未处理的同类告警（防止重复）
        existing_alert = db.query(Alert).filter(
            Alert.device_id == risk.device_id,
            Alert.alert_type == risk.risk_type,
            Alert.status.in_(["NEW", "NOTIFIED", "CONFIRMED"])
        ).first()
        
        if existing_alert:
            print(f"设备 {risk.device_id} 已有未处理告警，跳过创建")
            return None
        
        # 高风险自动抓拍
        snapshot_url = risk.snapshot_url
        if risk.risk_level in ["HIGH", "CRITICAL"] and not snapshot_url:
            print(f"📸 高风险告警，正在抓拍...")
            snapshot_url = await ezviz_service.capture_and_save(risk.device_id)
        
        # 创建告警
        alert_data = AlertCreate(
            device_id=risk.device_id,
            alert_type=risk.risk_type,
            risk_level=risk.risk_level,
            message=self._generate_alert_message(risk),
            snapshot_url=snapshot_url,
            risk_record_id=risk.id,
        )
        
        alert = Alert(**alert_data.model_dump())
        db.add(alert)
        db.commit()
        db.refresh(alert)
        
        # 推送WebSocket
        await self._broadcast_alert(alert)
        
        # TODO: 发送短信、电话等通知
        
        return alert
    
    def _generate_alert_message(self, risk: RiskRecord) -> str:
        """生成告警消息"""
        reasons_text = "、".join(risk.reasons) if risk.reasons else "未知原因"
        
        level_text = {
            "HIGH": "高风险",
            "MEDIUM": "中风险",
            "LOW": "低风险",
        }.get(risk.risk_level, "未知")
        
        return f"{level_text}告警：{risk.status}，原因：{reasons_text}"
    
    async def _broadcast_alert(self, alert: Alert):
        """广播告警"""
        await websocket_manager.broadcast({
            "type": "alert",
            "data": {
                "id": alert.id,
                "device_id": alert.device_id,
                "alert_type": alert.alert_type,
                "risk_level": alert.risk_level,
                "message": alert.message,
                "snapshot_url": alert.snapshot_url,
                "created_at": alert.created_at.isoformat(),
            }
        })


# 全局实例
alert_service = AlertService()
