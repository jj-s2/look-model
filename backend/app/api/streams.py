"""视频流API路由"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.device import Device

router = APIRouter()


@router.get("/streams/{device_id}/play-url")
async def get_play_url(device_id: str, db: Session = Depends(get_db)):
    """获取视频播放地址"""
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    if device.device_type != "camera":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该设备不支持视频播放"
        )
    
    # TODO: 调用萤石API获取播放地址
    play_url = f"ezopen://open.ys7.com/{device.device_serial}/1.hd.live"
    
    return {
        "device_id": device_id,
        "play_url": play_url,
        "expires_in": 7200,  # 2小时
    }


@router.post("/streams/{device_id}/snapshot")
async def take_snapshot(device_id: str, db: Session = Depends(get_db)):
    """抓拍图片"""
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    if device.device_type != "camera":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该设备不支持抓拍"
        )
    
    # TODO: 调用萤石API抓拍
    snapshot_url = f"https://example.com/snapshots/{device_id}/latest.jpg"
    
    return {
        "device_id": device_id,
        "snapshot_url": snapshot_url,
        "timestamp": "2026-07-25T10:30:00+09:00",
    }


@router.post("/streams/{device_id}/start")
async def start_stream(device_id: str, db: Session = Depends(get_db)):
    """开始视频流"""
    # TODO: 实现视频流开启逻辑
    return {"message": "视频流已开启", "device_id": device_id}


@router.post("/streams/{device_id}/stop")
async def stop_stream(device_id: str, db: Session = Depends(get_db)):
    """停止视频流"""
    # TODO: 实现视频流停止逻辑
    return {"message": "视频流已停止", "device_id": device_id}
