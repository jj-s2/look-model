"""设备API路由"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.device import DeviceCreate, DeviceResponse, DeviceStatus
from app.models.device import Device

router = APIRouter()


@router.get("/devices", response_model=list[DeviceResponse])
async def get_devices(db: Session = Depends(get_db)):
    """获取设备列表"""
    devices = db.query(Device).all()
    return devices


@router.get("/devices/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str, db: Session = Depends(get_db)):
    """获取单个设备详情"""
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    return device


@router.post("/devices", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def create_device(device_data: DeviceCreate, db: Session = Depends(get_db)):
    """创建设备"""
    # 检查设备序列号是否已存在
    existing = db.query(Device).filter(Device.device_serial == device_data.device_serial).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="设备序列号已存在"
        )
    
    device = Device(
        id=f"{device_data.device_type}_{device_data.device_serial}",
        **device_data.model_dump()
    )
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


@router.get("/devices/{device_id}/status", response_model=DeviceStatus)
async def get_device_status(device_id: str, db: Session = Depends(get_db)):
    """获取设备状态"""
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    return DeviceStatus(
        device_id=device.id,
        online=device.online_status,
        last_seen=device.last_seen_at,
        capabilities=device.capabilities or {}
    )


@router.post("/devices/{device_id}/bind")
async def bind_device(device_id: str, db: Session = Depends(get_db)):
    """绑定设备（占位接口，后续对接萤石）"""
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # TODO: 调用萤石API绑定设备
    
    return {"message": "设备绑定成功", "device_id": device_id}


@router.get("/devices/{device_id}/capabilities")
async def get_device_capabilities(device_id: str, db: Session = Depends(get_db)):
    """获取设备能力"""
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # TODO: 从萤石API获取设备能力
    capabilities = device.capabilities or {
        "supports_live_view": True,
        "supports_ptz": False,
        "supports_talk": False,
        "supports_snapshot": True,
        "supports_alarm_message": True,
    }
    
    return {"device_id": device_id, "capabilities": capabilities}
