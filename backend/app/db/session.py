"""数据库会话管理"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# 根据数据库类型配置引擎参数
engine_kwargs = {
    "pool_pre_ping": True,
}

# SQLite特殊配置
if "sqlite" in settings.DATABASE_URL:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL配置
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

# 创建数据库引擎
engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
