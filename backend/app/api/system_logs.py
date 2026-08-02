"""系统日志API路由"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from app.db.session import get_db
from app.models.system_log import SystemLog
from app.schemas.system_log import SystemLogResponse

router = APIRouter()


@router.get("/system-logs", response_model=list[SystemLogResponse])
async def get_system_logs(
    source: str = Query(None, description="日志来源过滤"),
    level: str = Query(None, description="日志级别过滤"),
    limit: int = Query(100, ge=1, le=500, description="返回数量"),
    db: Session = Depends(get_db)
):
    """获取系统日志列表"""
    query = db.query(SystemLog)
    
    # 过滤条件
    if source:
        query = query.filter(SystemLog.source == source)
    if level:
        query = query.filter(SystemLog.level == level)
    
    # 按时间倒序，限制数量
    logs = query.order_by(desc(SystemLog.created_at)).limit(limit).all()
    
    return logs


@router.get("/system-logs/stats")
async def get_system_logs_stats(
    hours: int = Query(24, ge=1, le=168, description="统计时间范围（小时）"),
    db: Session = Depends(get_db)
):
    """获取系统日志统计"""
    from sqlalchemy import func
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    # 按来源统计
    source_stats = db.query(
        SystemLog.source,
        func.count(SystemLog.id).label('count')
    ).filter(
        SystemLog.created_at >= start_time
    ).group_by(SystemLog.source).all()
    
    # 按级别统计
    level_stats = db.query(
        SystemLog.level,
        func.count(SystemLog.id).label('count')
    ).filter(
        SystemLog.created_at >= start_time
    ).group_by(SystemLog.level).all()
    
    return {
        "period_hours": hours,
        "by_source": {row.source: row.count for row in source_stats},
        "by_level": {row.level: row.count for row in level_stats},
        "total": sum(row.count for row in level_stats),
    }
