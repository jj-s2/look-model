"""模拟设备数据服务器"""
import asyncio
import json
import random
from datetime import datetime, timedelta
import httpx


class MockDeviceServer:
    """模拟摄像机和雷达数据生成"""
    
    def __init__(self, api_url="http://localhost:8000/api/v1"):
        self.api_url = api_url
        self.running = False
    
    def generate_camera_event(self, device_id="camera_001"):
        """生成模拟摄像机事件"""
        risk_score = random.uniform(0.1, 0.95)
        
        if risk_score >= 0.8:
            risk_level = "HIGH"
            status = random.choice(["walking_unstable", "stumbling", "about_to_fall"])
            reasons = ["身体左右摆动增加", "行走速度下降", "步幅不稳定"]
        elif risk_score >= 0.6:
            risk_level = "MEDIUM"
            status = random.choice(["standing_unsteady", "slow_movement", "sitting"])
            reasons = ["动作缓慢", "支撑不稳"]
        else:
            risk_level = "LOW"
            status = random.choice(["standing_stable", "sitting", "lying"])
            reasons = []
        
        return {
            "device_id": device_id,
            "person_id": "track_01",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "fall_risk",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "status": status,
            "reasons": reasons,
        }
    
    def generate_radar_data(self, device_id="radar_001"):
        """生成模拟雷达数据"""
        return {
            "device_id": device_id,
            "heart_rate": random.randint(60, 85),
            "respiratory_rate": random.randint(12, 20),
            "in_bed": random.choice([True, False]),
            "sleep_duration": round(random.uniform(0, 8), 1),
            "leave_bed_count": random.randint(0, 5),
            "recorded_at": datetime.utcnow().isoformat() + "Z",
        }
    
    async def send_camera_event(self):
        """发送摄像机事件到API"""
        event = self.generate_camera_event()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/ai/events",
                    json=event,
                    timeout=5.0
                )
                print(f"✅ 发送摄像机事件: risk_score={event['risk_score']}, status={response.status_code}")
            except Exception as e:
                print(f"❌ 发送失败: {e}")
    
    async def send_radar_data(self):
        """发送雷达数据到API"""
        data = self.generate_radar_data()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/health/report",
                    json=data,
                    timeout=5.0
                )
                print(f"✅ 发送雷达数据: HR={data['heart_rate']}, status={response.status_code}")
            except Exception as e:
                print(f"❌ 发送失败: {e}")
    
    async def run(self, camera_interval=10, radar_interval=30):
        """运行模拟服务器"""
        self.running = True
        print(f"🚀 模拟设备服务器启动")
        print(f"📹 摄像机事件间隔: {camera_interval}秒")
        print(f"📡 雷达数据间隔: {radar_interval}秒")
        
        last_camera_time = datetime.now()
        last_radar_time = datetime.now()
        
        while self.running:
            now = datetime.now()
            
            # 发送摄像机事件
            if (now - last_camera_time).seconds >= camera_interval:
                await self.send_camera_event()
                last_camera_time = now
            
            # 发送雷达数据
            if (now - last_radar_time).seconds >= radar_interval:
                await self.send_radar_data()
                last_radar_time = now
            
            await asyncio.sleep(1)
    
    def stop(self):
        """停止服务器"""
        self.running = False
        print("👋 模拟设备服务器停止")


if __name__ == "__main__":
    server = MockDeviceServer()
    try:
        asyncio.run(server.run(camera_interval=10, radar_interval=30))
    except KeyboardInterrupt:
        server.stop()
        print("\n退出")
