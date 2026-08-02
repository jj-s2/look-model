"""风险API路由"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from app.db.session import get_db
from app.schemas.risk import RiskCreate, RiskResponse, RiskSummary
from app.models.risk import RiskRecord

router = APIRouter()


@router.get("/risks/current", response_model=list[RiskSummary])
async def get_current_risks(db: Session = Depends(get_db)):
    """获取所有设备当前风险"""
    # 获取每个设备最新的风险记录
    from sqlalchemy import func
    
    subquery = db.query(
        RiskRecord.device_id,
        func.max(RiskRecord.recorded_at).label("max_time")
    ).group_by(RiskRecord.device_id).subquery()
    
    latest_risks = db.query(RiskRecord).join(
        subquery,
        (RiskRecord.device_id == subquery.c.device_id) &
        (RiskRecord.recorded_at == subquery.c.max_time)
    ).all()
    
    result = []
    for risk in latest_risks:
        result.append(RiskSummary(
            device_id=risk.device_id,
            current_risk_score=risk.risk_score,
            current_risk_level=risk.risk_level,
            last_update=risk.recorded_at,
            recent_reasons=risk.reasons or []
        ))
    
    return result


@router.get("/risks/history", response_model=list[RiskResponse])
async def get_risk_history(
    device_id: str = Query(None),
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """获取风险历史记录"""
    query = db.query(RiskRecord)
    
    if device_id:
        query = query.filter(RiskRecord.device_id == device_id)
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    query = query.filter(RiskRecord.recorded_at >= start_time)
    
    risks = query.order_by(desc(RiskRecord.recorded_at)).limit(1000).all()
    return risks


@router.post("/risks/report", response_model=RiskResponse, status_code=status.HTTP_201_CREATED)
async def report_risk(risk_data: RiskCreate, db: Session = Depends(get_db)):
    """上报风险记录（AI服务调用）"""
    risk = RiskRecord(**risk_data.model_dump())
    db.add(risk)
    db.commit()
    db.refresh(risk)
    
    # TODO: 检查是否需要创建告警
    # TODO: WebSocket推送给前端
    
    return risk
