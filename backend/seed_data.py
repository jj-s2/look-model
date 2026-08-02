"""填充测试数据"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.db.session import SessionLocal
from app.models.device import Device
from app.models.risk import RiskRecord
from app.models.alert import Alert
from app.models.health import HealthRecord


def create_sample_devices(db):
    """创建示例设备"""
    devices = [
        Device(
            id="camera_001",
            device_serial="C6C123456789",
            device_type="camera",
            device_name="客厅摄像头",
            online_status=True,
            capabilities={
                "supports_live_view": True,
                "supports_ptz": False,
                "supports_snapshot": True,
            },
            location="客厅",
        ),
        Device(
            id="camera_002",
            device_serial="C6C987654321",
            device_type="camera",
            device_name="卧室摄像头",
            online_status=True,
            capabilities={
                "supports_live_view": True,
                "supports_ptz": True,
                "supports_snapshot": True,
            },
            location="卧室",
        ),
        Device(
            id="radar_001",
            device_serial="SDNL1-001",
            device_type="radar",
            device_name="卧室雷达",
            online_status=True,
            capabilities={
                "supports_heart_rate": True,
                "supports_respiratory_rate": True,
                "supports_sleep_monitor": True,
            },
            location="卧室",
        ),
    ]
    
    for device in devices:
        existing = db.query(Device).filter(Device.id == device.id).first()
        if not existing:
            db.add(device)
            print(f"✓ 创建设备: {device.device_name}")
        else:
            print(f"  跳过已存在的设备: {device.device_name}")
    
    db.commit()
    return devices


def create_sample_risk_records(db, devices):
    """创建示例风险记录"""
    camera_devices = [d for d in devices if d.device_type == "camera"]
    
    # 创建最近24小时的风险记录
    records = []
    for i in range(48):  # 每30分钟一条
        for device in camera_devices:
            # 生成随机风险数据
            risk_score = random.uniform(0.1, 0.9)
            if risk_score >= 0.8:
                risk_level = "HIGH"
                status = random.choice(["walking_unstable", "lying_unusual", "fall_detected"])
                reasons = ["步态不稳", "活动减少"]
            elif risk_score >= 0.6:
                risk_level = "MEDIUM"
                status = random.choice(["walking_slow", "sitting_long"])
                reasons = ["活动缓慢"]
            else:
                risk_level = "LOW"
                status = random.choice(["walking_normal", "sitting", "standing"])
                reasons = []
            
            record = RiskRecord(
                device_id=device.id,
                person_id="person_001",
                risk_type="fall_risk",
                risk_score=risk_score,
                risk_level=risk_level,
                status=status,
                reasons=reasons,
                recorded_at=datetime.utcnow() - timedelta(minutes=30*i),
            )
            records.append(record)
    
    for record in records:
        db.add(record)
    
    db.commit()
    print(f"✓ 创建 {len(records)} 条风险记录")
    return records


def create_sample_alerts(db, devices):
    """创建示例告警"""
    camera_devices = [d for d in devices if d.device_type == "camera"]
    
    alerts = [
        Alert(
            device_id=camera_devices[0].id,
            alert_type="fall_risk",
            risk_level="HIGH",
            message="高风险告警：检测到步态不稳，可能有跌倒风险",
            status="NEW",
            created_at=datetime.utcnow() - timedelta(hours=2),
        ),
        Alert(
            device_id=camera_devices[1].id if len(camera_devices) > 1 else camera_devices[0].id,
            alert_type="fall_risk",
            risk_level="MEDIUM",
            message="中风险告警：长时间无活动",
            status="CONFIRMED",
            created_at=datetime.utcnow() - timedelta(hours=5),
            confirmed_at=datetime.utcnow() - timedelta(hours=4),
            handler="张护士",
            handler_note="已电话确认，老人正在休息",
        ),
        Alert(
            device_id=camera_devices[0].id,
            alert_type="fall_risk",
            risk_level="HIGH",
            message="高风险告警：疑似跌倒",
            status="RESOLVED",
            created_at=datetime.utcnow() - timedelta(days=1),
            confirmed_at=datetime.utcnow() - timedelta(days=1, hours=-1),
            resolved_at=datetime.utcnow() - timedelta(hours=22),
            handler="李医生",
            handler_note="误报，老人蹲下捡东西",
        ),
    ]
    
    for alert in alerts:
        db.add(alert)
    
    db.commit()
    print(f"✓ 创建 {len(alerts)} 条告警记录")
    return alerts


def create_sample_health_records(db, devices):
    """创建示例健康记录"""
    radar_devices = [d for d in devices if d.device_type == "radar"]
    
    if not radar_devices:
        print("  没有雷达设备，跳过健康记录")
        return []
    
    records = []
    # 创建最近24小时的健康记录
    for i in range(48):  # 每30分钟一条
        for device in radar_devices:
            record = HealthRecord(
                device_id=device.id,
                heart_rate=random.randint(60, 80),
                respiratory_rate=random.randint(14, 20),
                in_bed=i % 3 == 0,  # 模拟在床/不在床
                sleep_duration=random.uniform(0, 8) if i > 24 else None,
                leave_bed_count=random.randint(0, 3) if i > 24 else None,
                sleep_quality_score=random.uniform(0.6, 0.9) if i > 24 else None,
                recorded_at=datetime.utcnow() - timedelta(minutes=30*i),
            )
            records.append(record)
    
    for record in records:
        db.add(record)
    
    db.commit()
    print(f"✓ 创建 {len(records)} 条健康记录")
    return records


def seed_all():
    """填充所有测试数据"""
    db = SessionLocal()
    
    try:
        print("\n开始填充测试数据...")
        print("=" * 60)
        
        devices = create_sample_devices(db)
        create_sample_risk_records(db, devices)
        create_sample_alerts(db, devices)
        create_sample_health_records(db, devices)
        
        print("=" * 60)
        print("✓ 测试数据填充完成！\n")
        
    except Exception as e:
        print(f"\n✗ 填充数据失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_all()
