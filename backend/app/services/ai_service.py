"""AI服务接口"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.session import get_db
from app.models.risk import RiskRecord
from app.models.device import Device
from app.schemas.risk import RiskCreate
from app.services.alert_service import alert_service
from app.services.websocket_manager import websocket_manager

router = APIRouter()


@router.post("/ai/events", status_code=status.HTTP_201_CREATED)
async def receive_ai_event(event: RiskCreate, db: Session = Depends(get_db)):
    """接收AI分析结果
    
    成员A的AI服务将结果推送到这个接口
    """
    # 验证设备是否存在
    device = db.query(Device).filter(Device.id == event.device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {event.device_id} 不存在"
        )
    
    # 验证风险分数范围
    if not (0 <= event.risk_score <= 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="风险分数必须在0-1之间"
        )
    
    # 保存风险记录
    risk = RiskRecord(**event.model_dump())
    db.add(risk)
    db.commit()
    db.refresh(risk)
    
    # 推送到前端
    await websocket_manager.broadcast({
        "type": "risk_update",
        "data": {
            "device_id": risk.device_id,
            "person_id": risk.person_id,
            "risk_score": risk.risk_score,
            "risk_level": risk.risk_level,
            "status": risk.status,
            "reasons": risk.reasons,
            "recorded_at": risk.recorded_at.isoformat(),
        }
    })
    
    # 检查是否需要创建告警
    alert = await alert_service.check_and_create_alert(risk, db)
    
    return {
        "message": "AI事件已接收",
        "risk_id": risk.id,
        "alert_created": alert is not None,
        "alert_id": alert.id if alert else None,
    }
