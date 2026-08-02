"""添加演示数据到数据库"""
from app.db.session import SessionLocal
from app.models.device import Device
from app.models.health import HealthRecord
from app.models.risk import RiskRecord
from datetime import datetime, timedelta
import random

def seed_devices():
    """添加演示设备"""
    db = SessionLocal()
    
    devices = [
        Device(
            device_id="camera_001",
            device_name="客厅摄像头",
            device_type="camera",
            location="101室-客厅",
            status="online"
        ),
        Device(
            device_id="camera_002",
            device_name="卧室摄像头", 
            device_type="camera",
            location="101室-卧室",
            status="online"
        ),
        Device(
            device_id="radar_001",
            device_name="卧室雷达",
            device_type="radar",
            location="101室-卧室",
            status="online"
        )
    ]
    
    for device in devices:
        existing = db.query(Device).filter_by(device_id=device.device_id).first()
        if not existing:
            db.add(device)
            print(f"✅ 添加设备: {device.device_name}")
        else:
            print(f"⚠️ 设备已存在: {device.device_name}")
    
    db.commit()
    db.close()

def seed_health_data():
    """添加演示健康数据"""
    db = SessionLocal()
    
    # 生成最近24小时的数据
    now = datetime.now()
    for i in range(24):
        time = now - timedelta(hours=23-i)
        
        health = HealthRecord(
            device_id="radar_001",
            heart_rate=random.randint(60, 85) + random.randint(-5, 5),
            respiratory_rate=random.randint(14, 18),
            in_bed=True if i >= 22 or i <= 6 else False,
            sleep_duration=7.5 if i == 6 else None,
            recorded_at=time,
            created_at=time
        )
        db.add(health)
    
    db.commit()
    db.close()
    print(f"✅ 添加了24小时健康数据")

def seed_risk_data():
    """添加演示风险数据"""
    db = SessionLocal()
    
    # 添加几条最近的风险记录
    risk_levels = ["LOW", "MEDIUM", "LOW", "MEDIUM", "HIGH"]
    risk_scores = [0.2, 0.5, 0.3, 0.6, 0.85]
    statuses = ["standing", "sitting", "walking", "standing", "falling"]
    
    for i, (level, score, status) in enumerate(zip(risk_levels, risk_scores, statuses)):
        risk = RiskRecord(
            device_id="camera_001",
            risk_type="fall_risk",
            risk_level=level,
            risk_score=score,
            status=status,
            reasons=["AI检测分析", "姿态识别"],
            detected_at=datetime.now() - timedelta(minutes=30-i*5),
            created_at=datetime.now() - timedelta(minutes=30-i*5)
        )
        db.add(risk)
    
    db.commit()
    db.close()
    print(f"✅ 添加了{len(risk_levels)}条风险记录")

if __name__ == "__main__":
    print("="*60)
    print("   添加演示数据到数据库")
    print("="*60)
    print()
    
    print("[1/3] 添加设备...")
    seed_devices()
    print()
    
    print("[2/3] 添加健康数据...")
    seed_health_data()
    print()
    
    print("[3/3] 添加风险数据...")
    seed_risk_data()
    print()
    
    print("="*60)
    print("✅ 演示数据添加完成！")
    print("💡 刷新浏览器页面（F5）即可看到数据")
    print("="*60)
