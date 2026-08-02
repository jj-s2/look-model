"""健康数据API路由"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from app.db.session import get_db
from app.schemas.health import HealthCreate, HealthResponse
from app.models.health import HealthRecord
from app.services.health_validator import health_validator
from app.services.websocket_manager import websocket_manager

router = APIRouter()


@router.get("/health/latest", response_model=list[HealthResponse])
async def get_latest_health(
    device_id: str = Query(None),
    db: Session = Depends(get_db)
):
    """获取最新健康数据"""
    query = db.query(HealthRecord)
    
    if device_id:
        query = query.filter(HealthRecord.device_id == device_id)
        # 单个设备只返回最新一条
        latest = query.order_by(desc(HealthRecord.recorded_at)).first()
        return [latest] if latest else []
    
    # 获取每个设备最新的记录
    from sqlalchemy import func
    
    subquery = db.query(
        HealthRecord.device_id,
        func.max(HealthRecord.recorded_at).label("max_time")
    ).group_by(HealthRecord.device_id).subquery()
    
    latest_records = db.query(HealthRecord).join(
        subquery,
        (HealthRecord.device_id == subquery.c.device_id) &
        (HealthRecord.recorded_at == subquery.c.max_time)
    ).all()
    
    return latest_records


@router.get("/health/trends")
async def get_health_trends(
    device_id: str = Query(...),
    period: str = Query("24h", pattern="^(24h|7d|30d)$"),
    db: Session = Depends(get_db)
):
    """获取健康趋势"""
    hours_map = {"24h": 24, "7d": 168, "30d": 720}
    hours = hours_map[period]
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    records = db.query(HealthRecord).filter(
        HealthRecord.device_id == device_id,
        HealthRecord.recorded_at >= start_time
    ).order_by(HealthRecord.recorded_at).all()
    
    return {
        "device_id": device_id,
        "period": period,
        "data": [
            {
                "time": r.recorded_at.isoformat(),
                "heart_rate": r.heart_rate,
                "respiratory_rate": r.respiratory_rate,
                "in_bed": r.in_bed,
                "sleep_duration": r.sleep_duration,
            }
            for r in records
        ]
    }


@router.post("/health/report", response_model=HealthResponse, status_code=status.HTTP_201_CREATED)
async def report_health(health_data: HealthCreate, db: Session = Depends(get_db)):
    """上报健康数据（雷达调用）"""
    # 数据验证与清洗
    cleaned_data = health_validator.validate_and_filter(health_data.device_id, health_data)
    
    # 如果所有关键数据都被过滤掉，返回错误
    if cleaned_data.heart_rate is None and cleaned_data.respiratory_rate is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="健康数据异常，所有指标均被过滤"
        )
    
    health = HealthRecord(**cleaned_data.model_dump())
    db.add(health)
    db.commit()
    db.refresh(health)
    
    # WebSocket推送给前端
    await websocket_manager.broadcast({
        "type": "health_update",
        "data": {
            "device_id": health.device_id,
            "heart_rate": health.heart_rate,
            "respiratory_rate": health.respiratory_rate,
            "body_temperature": health.body_temperature,
            "in_bed": health.in_bed,
            "recorded_at": health.recorded_at.isoformat(),
        }
    })
    
    return health
