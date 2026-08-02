"""萤石开放平台服务"""
import httpx
import os
from pathlib import Path
from datetime import datetime
from app.config import get_settings


class EzvizService:
    """萤石平台API封装"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.EZVIZ_API_URL
        self.app_key = self.settings.EZVIZ_APP_KEY
        self.app_secret = self.settings.EZVIZ_APP_SECRET
        self.access_token = self.settings.EZVIZ_ACCESS_TOKEN
        
        # 创建快照存储目录
        self.snapshot_dir = Path("app/static/snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_access_token(self) -> str:
        """获取AccessToken"""
        # TODO: 实现AccessToken获取逻辑
        # POST /lapp/token/get
        # appKey, appSecret
        return self.access_token
    
    async def get_device_list(self) -> list:
        """获取设备列表"""
        # TODO: 实现设备列表获取
        # POST /lapp/device/list
        return []
    
    async def get_device_info(self, device_serial: str) -> dict:
        """获取设备详情"""
        # TODO: 实现设备详情获取
        # POST /lapp/device/info
        return {}
    
    async def get_live_address(self, device_serial: str, channel: int = 1) -> str:
        """获取直播地址"""
        # TODO: 实现直播地址获取
        # POST /lapp/live/address/get
        return f"ezopen://open.ys7.com/{device_serial}/{channel}.hd.live"
    
    async def capture_picture(self, device_serial: str, channel: int = 1) -> str:
        """设备抓图"""
        # TODO: 实现抓图功能
        # POST /lapp/device/capture
        return ""
    
    async def capture_and_save(self, device_id: str) -> str | None:
        """抓拍并保存图片
        
        Args:
            device_id: 设备ID
            
        Returns:
            保存的图片相对路径，失败返回None
        """
        try:
            # 1. 调用萤石抓拍接口
            image_url = await self.capture_picture(device_id)
            
            if not image_url:
                print(f"⚠️ 设备 {device_id} 抓拍失败：未获取到图片URL")
                return None
            
            # 2. 下载图片
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(image_url)
                if response.status_code != 200:
                    print(f"⚠️ 下载抓拍图失败: HTTP {response.status_code}")
                    return None
                
                # 3. 保存到本地
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{device_id}_{timestamp}.jpg"
                
                # 按设备ID创建子目录
                device_dir = self.snapshot_dir / device_id
                device_dir.mkdir(exist_ok=True)
                
                file_path = device_dir / filename
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                # 返回相对路径（用于数据库存储）
                relative_path = f"/static/snapshots/{device_id}/{filename}"
                print(f"📸 抓拍成功: {relative_path}")
                return relative_path
                
        except Exception as e:
            print(f"❌ 抓拍保存失败: {e}")
            return None
    
    async def ptz_control(self, device_serial: str, direction: str, speed: int = 1):
        """云台控制"""
        # TODO: 实现云台控制
        # POST /lapp/device/ptz/start
        pass


# 全局实例
ezviz_service = EzvizService()
