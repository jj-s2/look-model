"""告警API路由"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from app.db.session import get_db
from app.schemas.alert import AlertCreate, AlertResponse, AlertConfirm, AlertResolve, AlertUpdate
from app.models.alert import Alert

router = APIRouter()


@router.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    status_filter: str = Query(None),
    device_id: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """获取告警列表"""
    query = db.query(Alert)
    
    if status_filter:
        query = query.filter(Alert.status == status_filter)
    
    if device_id:
        query = query.filter(Alert.device_id == device_id)
    
    alerts = query.order_by(desc(Alert.created_at)).limit(limit).all()
    return alerts


@router.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(alert_data: AlertCreate, db: Session = Depends(get_db)):
    """创建告警"""
    alert = Alert(**alert_data.model_dump())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    
    # TODO: WebSocket推送告警
    # TODO: 发送通知（短信、电话等）
    
    return alert


@router.post("/alerts/{alert_id}/confirm", response_model=AlertResponse)
async def confirm_alert(
    alert_id: int,
    confirm_data: AlertConfirm,
    db: Session = Depends(get_db)
):
    """确认告警"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"告警 {alert_id} 不存在"
        )
    
    if alert.status not in ["NEW", "NOTIFIED"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"告警状态为 {alert.status}，无法确认"
        )
    
    alert.status = "CONFIRMED"
    alert.confirmed_at = datetime.utcnow()
    alert.handler = confirm_data.handler
    alert.handler_note = confirm_data.note
    
    db.commit()
    db.refresh(alert)
    return alert


@router.post("/alerts/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    resolve_data: AlertResolve,
    db: Session = Depends(get_db)
):
    """处理完成告警"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"告警 {alert_id} 不存在"
        )
    
    if alert.status == "RESOLVED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="告警已处理完成"
        )
    
    alert.status = "RESOLVED"
    alert.resolved_at = datetime.utcnow()
    alert.handler = resolve_data.handler
    alert.handler_note = resolve_data.note
    
    db.commit()
    db.refresh(alert)
    return alert


@router.patch("/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int,
    update_data: AlertUpdate,
    db: Session = Depends(get_db)
):
    """更新告警（标记误报等）"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"告警 {alert_id} 不存在"
        )
    
    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(alert, field, value)
    
    if update_data.status == "FALSE_ALARM":
        alert.resolved_at = datetime.utcnow()
    
    db.commit()
    db.refresh(alert)
    return alert
