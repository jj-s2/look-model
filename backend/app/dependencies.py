"""依赖注入"""
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.config import get_settings


def get_current_settings():
    """获取当前配置"""
    return get_settings()


async def validate_device_id(device_id: str, db: Session = Depends(get_db)):
    """验证设备ID是否存在"""
    # TODO: 实现设备验证逻辑
    return device_id


async def check_ezviz_credentials(settings = Depends(get_current_settings)):
    """检查萤石平台凭证"""
    if not settings.EZVIZ_APP_KEY or not settings.EZVIZ_APP_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="萤石平台凭证未配置",
        )
    return settings
