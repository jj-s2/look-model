"""应用配置"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用配置
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    APP_TITLE: str = "老年人多模态AI监测预警平台"
    APP_VERSION: str = "0.1.0"
    
    # 数据库配置
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/elderly_monitor"
    
    # 萤石平台配置
    EZVIZ_APP_KEY: str = ""
    EZVIZ_APP_SECRET: str = ""
    EZVIZ_ACCESS_TOKEN: str = ""
    EZVIZ_API_URL: str = "https://open.ys7.com/api/lapp"
    
    # 告警阈值配置
    ALERT_HIGH_THRESHOLD: float = 0.8
    ALERT_MEDIUM_THRESHOLD: float = 0.6
    ALERT_LOW_THRESHOLD: float = 0.3
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # SDNL1雷达配置
    SDNL1_API_URL: str = ""
    SDNL1_API_KEY: str = ""
    
    # CORS配置
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
