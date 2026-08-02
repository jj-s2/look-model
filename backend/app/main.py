"""FastAPI应用入口"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api import devices, streams, risks, alerts, health, system_logs
from app.services.ai_service import router as ai_router
from app.services.websocket_manager import websocket_manager
from app.services.scheduler import task_scheduler

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print(f"🚀 启动 {settings.APP_TITLE} v{settings.APP_VERSION}")
    print(f"📝 环境: {settings.APP_ENV}")
    print(f"🔗 数据库: {settings.DATABASE_URL.split('@')[-1]}")
    
    # 启动后台任务调度器
    task_scheduler.start()
    
    yield
    
    # 关闭时执行
    print("👋 关闭应用...")
    task_scheduler.shutdown()


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="老年人跌倒风险监测与健康预警系统",
    lifespan=lifespan,
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 健康检查
@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "name": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "running",
        "environment": settings.APP_ENV,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
    }


# 注册路由
app.include_router(devices.router, prefix="/api/v1", tags=["Devices"])
app.include_router(streams.router, prefix="/api/v1", tags=["Streams"])
app.include_router(risks.router, prefix="/api/v1", tags=["Risks"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(system_logs.router, prefix="/api/v1", tags=["System Logs"])
app.include_router(ai_router, prefix="/api/v1", tags=["AI"])


# WebSocket端点
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket连接端点（增强版：支持心跳）"""
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理客户端消息
            if data == "ping":
                await websocket.send_text("pong")
                websocket_manager.update_pong_time(client_id)
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        websocket_manager.disconnect(client_id)


# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "服务器内部错误",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
