"""生成演示数据脚本"""
import requests
import random
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"

def add_demo_devices():
    """添加演示设备"""
    devices = [
        {
            "device_id": "camera_001",
            "device_name": "客厅摄像头",
            "device_type": "camera",
            "location": "101室-客厅",
            "status": "online"
        },
        {
            "device_id": "camera_002", 
            "device_name": "卧室摄像头",
            "device_type": "camera",
            "location": "101室-卧室",
            "status": "online"
        },
        {
            "device_id": "radar_001",
            "device_name": "卧室雷达",
            "device_type": "radar",
            "location": "101室-卧室",
            "status": "online"
        }
    ]
    
    for device in devices:
        try:
            response = requests.post(f"{BASE_URL}/devices", json=device)
            if response.status_code == 200:
                print(f"✅ 添加设备成功: {device['device_name']}")
            else:
                print(f"⚠️ 设备可能已存在: {device['device_name']}")
        except Exception as e:
            print(f"❌ 添加设备失败: {e}")

def add_demo_health_data():
    """添加演示健康数据"""
    health_data = {
        "device_id": "radar_001",
        "heart_rate": random.randint(60, 90),
        "respiratory_rate": random.randint(12, 20),
        "in_bed": True,
        "sleep_duration": 6.5,
        "recorded_at": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/health/report", json=health_data)
        if response.status_code == 201:
            print(f"✅ 添加健康数据成功: 心率={health_data['heart_rate']}, 呼吸={health_data['respiratory_rate']}")
        else:
            print(f"⚠️ 健康数据添加响应: {response.status_code}")
    except Exception as e:
        print(f"❌ 添加健康数据失败: {e}")

def add_demo_risk():
    """添加演示风险数据"""
    risk_data = {
        "device_id": "camera_001",
        "risk_type": "fall_risk",
        "risk_level": "MEDIUM",
        "risk_score": 0.65,
        "status": "standing",
        "reasons": ["姿态不稳", "活动频繁"],
        "snapshot_url": None
    }
    
    try:
        response = requests.post(f"{BASE_URL}/risks", json=risk_data)
        if response.status_code == 201:
            print(f"✅ 添加风险数据成功: {risk_data['risk_level']}")
        else:
            print(f"⚠️ 风险数据添加响应: {response.status_code}")
    except Exception as e:
        print(f"❌ 添加风险数据失败: {e}")

if __name__ == "__main__":
    print("="*60)
    print("   生成演示数据")
    print("="*60)
    print()
    
    print("[1/3] 添加演示设备...")
    add_demo_devices()
    print()
    
    print("[2/3] 添加健康数据...")
    add_demo_health_data()
    print()
    
    print("[3/3] 添加风险数据...")
    add_demo_risk()
    print()
    
    print("="*60)
    print("✅ 演示数据生成完成！")
    print("💡 刷新浏览器页面即可看到数据")
    print("="*60)
