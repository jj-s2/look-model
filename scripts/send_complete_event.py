"""发送完整的模拟事件（风险+告警）"""
import requests
from datetime import datetime
import json

BACKEND_URL = "http://localhost:8000"

def send_complete_fall_risk_event(score=0.85, device_id="CAMERA_001"):
    """发送完整的跌倒风险事件（风险记录 + 告警）"""
    
    # 1. 发送风险记录
    print("=" * 60)
    print("📊 步骤 1: 发送风险记录")
    print("=" * 60)
    
    risk_level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
    
    risk_data = {
        "device_id": device_id,
        "person_id": "PERSON_001",
        "risk_type": "fall_detection",
        "risk_score": score,
        "risk_level": risk_level,
        "status": "ACTIVE",
        "reasons": [
            "检测到不稳定姿态",
            f"跌倒风险评分: {score:.2f}",
            "建议立即关注"
        ],
        "recorded_at": datetime.now().isoformat(),
        "snapshot_url": None
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/risks/report", json=risk_data, timeout=5)
        response.raise_for_status()
        risk_result = response.json()
        print(f"✅ 风险记录创建成功 (ID: {risk_result['id']})")
        print(f"   设备: {device_id}")
        print(f"   风险评分: {score}")
        print(f"   风险等级: {risk_level}")
    except Exception as e:
        print(f"❌ 风险记录创建失败: {e}")
        return
    
    # 2. 创建告警
    print("\n" + "=" * 60)
    print("🚨 步骤 2: 创建告警")
    print("=" * 60)
    
    alert_data = {
        "device_id": device_id,
        "alert_type": "FALL_RISK",
        "severity": risk_level,
        "risk_level": risk_level,
        "risk_score": score,
        "message": f"检测到高危跌倒风险！风险评分: {score:.2f}",
        "details": {
            "risk_type": "fall_detection",
            "reasons": risk_data["reasons"],
            "timestamp": datetime.now().isoformat()
        },
        "status": "NEW"
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/alerts", json=alert_data, timeout=5)
        response.raise_for_status()
        alert_result = response.json()
        print(f"✅ 告警创建成功 (ID: {alert_result['id']})")
        print(f"   类型: {alert_result['alert_type']}")
        print(f"   严重级别: {alert_result['severity']}")
        print(f"   状态: {alert_result['status']}")
    except Exception as e:
        print(f"❌ 告警创建失败: {e}")
        return
    
    # 3. 查询验证
    print("\n" + "=" * 60)
    print("🔍 步骤 3: 验证数据")
    print("=" * 60)
    
    try:
        # 查询当前风险
        response = requests.get(f"{BACKEND_URL}/api/v1/risks/current", timeout=5)
        current_risks = response.json()
        device_risk = next((r for r in current_risks if r['device_id'] == device_id), None)
        if device_risk:
            print(f"✅ 当前风险状态:")
            print(f"   风险评分: {device_risk['current_risk_score']}")
            print(f"   风险等级: {device_risk['current_risk_level']}")
        
        # 查询告警列表
        response = requests.get(f"{BACKEND_URL}/api/v1/alerts?limit=5", timeout=5)
        alerts = response.json()
        print(f"\n✅ 最新告警数量: {len(alerts)}")
        if alerts:
            latest = alerts[0]
            print(f"   最新告警ID: {latest['id']}")
            print(f"   状态: {latest['status']}")
    except Exception as e:
        print(f"⚠️  验证失败: {e}")
    
    print("\n" + "=" * 60)
    print("✨ 完成！现在前端应该可以看到以下效果:")
    print("=" * 60)
    print("1. 📊 仪表板 - '当前风险等级' 显示 HIGH (红色)")
    print("2. 📈 风险趋势 - 选择 CAMERA_001，图表会显示新的风险点")
    print("3. 🚨 告警中心 - 会出现新的告警记录，状态为 NEW")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='发送完整的跌倒风险事件')
    parser.add_argument('--score', type=float, default=0.85, help='风险评分 (0-1)')
    parser.add_argument('--device', type=str, default='CAMERA_001', help='设备ID')
    args = parser.parse_args()
    
    send_complete_fall_risk_event(args.score, args.device)
