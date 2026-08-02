"""缓存服务抽象层"""
from abc import ABC, abstractmethod
from typing import Any, Optional
import json
import time


class CacheService(ABC):
    """缓存服务抽象基类
    
    提供统一的缓存接口，支持多种实现（内存、Redis等）
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），默认300秒
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """清空所有缓存"""
        pass


class MemoryCache(CacheService):
    """内存缓存实现（MVP阶段使用）
    
    适用于单机部署场景，数据存储在进程内存中
    """
    
    def __init__(self):
        self._cache = {}
        self._expiry = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # 检查是否过期
        if key in self._expiry:
            if time.time() > self._expiry[key]:
                # 过期，删除
                await self.delete(key)
                return None
        
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """设置缓存"""
        try:
            self._cache[key] = value
            self._expiry[key] = time.time() + ttl
            return True
        except Exception as e:
            print(f"❌ 内存缓存设置失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if key in self._cache:
            del self._cache[key]
        if key in self._expiry:
            del self._expiry[key]
        return True
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        value = await self.get(key)
        return value is not None
    
    async def clear(self) -> bool:
        """清空所有缓存"""
        self._cache.clear()
        self._expiry.clear()
        return True
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        # 清理过期键
        current_time = time.time()
        expired_keys = [k for k, exp_time in self._expiry.items() if current_time > exp_time]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
        
        return {
            "type": "memory",
            "total_keys": len(self._cache),
            "expired_cleaned": len(expired_keys),
        }


class RedisCache(CacheService):
    """Redis缓存实现（生产环境使用）
    
    适用于分布式部署场景，需要安装redis库
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Args:
            redis_url: Redis连接地址
        """
        try:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(redis_url, decode_responses=True)
            self._available = True
        except ImportError:
            print("⚠️ Redis库未安装，请使用 pip install redis")
            self._available = False
        except Exception as e:
            print(f"⚠️ Redis连接失败: {e}")
            self._available = False
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self._available:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"❌ Redis获取失败: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """设置缓存"""
        if not self._available:
            return False
        
        try:
            value_str = json.dumps(value, ensure_ascii=False)
            await self.redis.setex(key, ttl, value_str)
            return True
        except Exception as e:
            print(f"❌ Redis设置失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self._available:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            print(f"❌ Redis删除失败: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self._available:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            print(f"❌ Redis检查失败: {e}")
            return False
    
    async def clear(self) -> bool:
        """清空所有缓存"""
        if not self._available:
            return False
        
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            print(f"❌ Redis清空失败: {e}")
            return False


# 全局缓存实例（默认使用内存缓存）
# 生产环境可切换为 RedisCache
cache_service: CacheService = MemoryCache()


def get_cache() -> CacheService:
    """获取缓存服务实例"""
    return cache_service
