"""后台定时任务调度器"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.services.alert_escalation import alert_escalation_service
from app.services.websocket_manager import websocket_manager


class TaskScheduler:
    """后台任务调度器"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
    
    def start(self):
        """启动调度器"""
        # 每分钟检查一次需要升级的告警
        self.scheduler.add_job(
            alert_escalation_service.check_and_escalate,
            trigger=IntervalTrigger(minutes=1),
            id="alert_escalation_check",
            name="检查告警超时升级",
            replace_existing=True,
        )
        
        # 每10分钟检查一次WebSocket连接超时
        self.scheduler.add_job(
            self._check_websocket_timeout,
            trigger=IntervalTrigger(minutes=10),
            id="websocket_timeout_check",
            name="检查WebSocket连接超时",
            replace_existing=True,
        )
        
        self.scheduler.start()
        print("✅ 后台任务调度器已启动")
    
    def shutdown(self):
        """关闭调度器"""
        self.scheduler.shutdown()
        print("👋 后台任务调度器已关闭")
    
    async def _check_websocket_timeout(self):
        """检查WebSocket连接超时"""
        timeout_clients = websocket_manager.check_timeout(timeout_seconds=60)
        if timeout_clients:
            print(f"⏰ 检测到 {len(timeout_clients)} 个超时连接: {timeout_clients}")
            for client_id in timeout_clients:
                websocket_manager.disconnect(client_id)


# 全局实例
task_scheduler = TaskScheduler()
