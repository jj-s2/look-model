"""WebSocket连接管理器"""
from typing import Dict
from fastapi import WebSocket
import time


class WebSocketManager:
    """WebSocket连接管理（增强版：支持心跳）"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_pong_time: Dict[str, float] = {}  # 记录最后一次pong时间
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """接受连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.last_pong_time[client_id] = time.time()
        print(f"✅ WebSocket客户端 {client_id} 已连接，当前连接数: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """断开连接"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.last_pong_time:
            del self.last_pong_time[client_id]
        print(f"❌ WebSocket客户端 {client_id} 已断开，当前连接数: {len(self.active_connections)}")
    
    def update_pong_time(self, client_id: str):
        """更新pong时间"""
        self.last_pong_time[client_id] = time.time()
    
    def check_timeout(self, timeout_seconds: int = 60) -> list:
        """检查超时连接
        
        Returns:
            超时的client_id列表
        """
        current_time = time.time()
        timeout_clients = []
        
        for client_id, last_pong in self.last_pong_time.items():
            if current_time - last_pong > timeout_seconds:
                timeout_clients.append(client_id)
        
        return timeout_clients
    
    async def send_personal_message(self, message: dict, client_id: str):
        """发送个人消息"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"发送消息失败: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """广播消息"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"广播失败给 {client_id}: {e}")
                disconnected.append(client_id)
        
        # 清理断开的连接
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def get_stats(self) -> dict:
        """获取连接统计"""
        return {
            "total_connections": len(self.active_connections),
            "active_clients": list(self.active_connections.keys()),
        }


# 全局单例
websocket_manager = WebSocketManager()
