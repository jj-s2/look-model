"""发送卧室摄像头高风险告警 - 带WebSocket推送"""
import requests
from datetime import datetime
import json

BACKEND_URL = "http://localhost:8000"

def send_bedroom_fall_risk():
    """发送卧室摄像头跌倒高风险事件（7/25 16:33）"""
    
    # 设置特定时间 7/25 16:33
    timestamp = "2026-07-25T16:33:00"
    device_id = "camera_002"  # 卧室摄像头
    score = 0.85
    
    print("=" * 70)
    print("🏠 卧室摄像头高风险告警")
    print("=" * 70)
    print(f"设备: camera_002 (卧室摄像头)")
    print(f"时间: 2026-07-25 16:33:00")
    print(f"风险评分: {score}")
    print("=" * 70)
    
    # 1. 发送风险记录
    print("\n📊 步骤 1: 创建风险记录...")
    
    risk_data = {
        "device_id": device_id,
        "person_id": "PERSON_001",
        "risk_type": "fall_detection",
        "risk_score": score,
        "risk_level": "HIGH",
        "status": "ACTIVE",
        "reasons": [
            "卧室摄像头检测到不稳定姿态",
            f"跌倒风险评分: {score:.2f}",
            "老人可能有跌倒危险",
            "建议立即查看实时画面"
        ],
        "recorded_at": timestamp,
        "snapshot_url": None
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/risks/report", json=risk_data, timeout=5)
        response.raise_for_status()
        risk_result = response.json()
        print(f"✅ 风险记录已创建 (ID: {risk_result['id']})")
    except Exception as e:
        print(f"❌ 风险记录创建失败: {e}")
        return
    
    # 2. 创建告警
    print("\n🚨 步骤 2: 创建告警...")
    
    alert_data = {
        "device_id": device_id,
        "alert_type": "FALL_RISK",
        "risk_level": "HIGH",
        "message": f"【卧室】检测到高危跌倒风险！评分: {score:.2f}",
        "snapshot_url": None,
        "risk_record_id": risk_result['id']
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/alerts", json=alert_data, timeout=5)
        response.raise_for_status()
        alert_result = response.json()
        print(f"✅ 告警已创建 (ID: {alert_result['id']})")
        print(f"   状态: {alert_result['status']}")
        print(f"   风险等级: {alert_result['risk_level']}")
    except Exception as e:
        print(f"❌ 告警创建失败: {e}")
        return
    
    # 3. 通过AI服务推送（触发WebSocket）
    print("\n📡 步骤 3: 触发实时推送...")
    
    # 尝试通过AI服务的事件处理接口
    event_data = {
        "event_type": "fall_risk_detected",
        "device_id": device_id,
        "timestamp": timestamp,
        "data": {
            "risk_score": score,
            "risk_level": "HIGH",
            "alert_id": alert_result['id'],
            "message": alert_data['message']
        }
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/ai/events", json=event_data, timeout=5)
        if response.status_code in [200, 201]:
            print("✅ WebSocket推送已触发")
        else:
            print(f"⚠️  WebSocket推送可能未触发 (状态码: {response.status_code})")
    except Exception as e:
        print(f"⚠️  WebSocket推送失败: {e}")
        print("   (前端可能需要手动刷新)")
    
    # 4. 验证数据
    print("\n🔍 步骤 4: 验证数据已保存...")
    
    try:
        # 查询当前风险
        response = requests.get(f"{BACKEND_URL}/api/v1/risks/current", timeout=5)
        risks = response.json()
        bedroom_risk = next((r for r in risks if r['device_id'] == device_id), None)
        
        if bedroom_risk:
            print(f"✅ 卧室摄像头当前风险:")
            print(f"   风险评分: {bedroom_risk['current_risk_score']}")
            print(f"   风险等级: {bedroom_risk['current_risk_level']}")
            print(f"   更新时间: {bedroom_risk['last_update']}")
        
        # 查询告警
        response = requests.get(f"{BACKEND_URL}/api/v1/alerts?limit=5", timeout=5)
        alerts = response.json()
        bedroom_alert = next((a for a in alerts if a['device_id'] == device_id), None)
        
        if bedroom_alert:
            print(f"\n✅ 最新告警:")
            print(f"   ID: {bedroom_alert['id']}")
            print(f"   消息: {bedroom_alert['message']}")
            print(f"   状态: {bedroom_alert['status']}")
            
    except Exception as e:
        print(f"⚠️  验证失败: {e}")
    
    # 5. 前端查看指引
    print("\n" + "=" * 70)
    print("✨ 完成！前端查看效果:")
    print("=" * 70)
    print("📊 仪表板 (http://localhost:3000/)")
    print("   → 刷新页面 (F5)")
    print("   → 查看 '当前风险等级' 卡片是否显示 HIGH")
    print("   → 查看 '今日告警' 数量是否增加")
    print()
    print("📈 风险趋势 (点击导航栏 '风险趋势')")
    print("   → 选择设备: camera_002 (卧室摄像头)")
    print("   → 查看折线图是否有新数据点 (0.85)")
    print("   → 时间: 16:33")
    print()
    print("🚨 告警中心 (点击导航栏 '告警中心')")
    print("   → 应该看到新告警")
    print("   → 消息: '【卧室】检测到高危跌倒风险！评分: 0.85'")
    print("   → 状态: NEW (红色)")
    print("=" * 70)
    print()
    print("💡 提示:")
    print("   - 如果没有实时更新，请按 F5 刷新页面")
    print("   - 检查浏览器控制台是否有错误 (F12)")
    print("   - 确认 WebSocket 连接状态")
    print("=" * 70)


if __name__ == "__main__":
    send_bedroom_fall_risk()
