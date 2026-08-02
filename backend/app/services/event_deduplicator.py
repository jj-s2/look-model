"""事件去重与防抖服务"""
import time
from typing import Dict, Tuple
from datetime import datetime, timedelta


class EventDeduplicator:
    """事件防抖去重器
    
    使用滑动窗口算法，防止同一设备、同一风险类型在短时间内重复生成告警
    """
    
    def __init__(self, cooldown_seconds: int = 10):
        """
        Args:
            cooldown_seconds: 冷却时间（秒），默认10秒
        """
        self.cooldown_seconds = cooldown_seconds
        # 存储格式: {fingerprint: (last_trigger_time, count)}
        self._event_cache: Dict[str, Tuple[float, int]] = {}
        self._cleanup_interval = 60  # 每60秒清理一次过期数据
        self._last_cleanup = time.time()
    
    def _generate_fingerprint(self, device_id: str, risk_type: str, risk_level: str) -> str:
        """生成事件指纹"""
        return f"{device_id}:{risk_type}:{risk_level}"
    
    def should_process(self, device_id: str, risk_type: str, risk_level: str) -> bool:
        """判断事件是否应该被处理
        
        Args:
            device_id: 设备ID
            risk_type: 风险类型
            risk_level: 风险等级
            
        Returns:
            True: 应该处理该事件
            False: 应该忽略该事件（在冷却期内）
        """
        fingerprint = self._generate_fingerprint(device_id, risk_type, risk_level)
        current_time = time.time()
        
        # 定期清理过期数据
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
        
        # 检查是否在冷却期内
        if fingerprint in self._event_cache:
            last_time, count = self._event_cache[fingerprint]
            time_diff = current_time - last_time
            
            if time_diff < self.cooldown_seconds:
                # 在冷却期内，更新计数但不处理
                self._event_cache[fingerprint] = (current_time, count + 1)
                print(f"🔇 事件防抖: {fingerprint} 在冷却期内 ({time_diff:.1f}s < {self.cooldown_seconds}s), 已忽略 (第{count + 1}次)")
                return False
        
        # 不在冷却期，记录并允许处理
        self._event_cache[fingerprint] = (current_time, 1)
        return True
    
    def _cleanup_expired(self):
        """清理过期的缓存数据"""
        current_time = time.time()
        expired_keys = []
        
        for fingerprint, (last_time, _) in self._event_cache.items():
            if current_time - last_time > self.cooldown_seconds * 2:
                expired_keys.append(fingerprint)
        
        for key in expired_keys:
            del self._event_cache[key]
        
        if expired_keys:
            print(f"🧹 清理过期事件缓存: {len(expired_keys)} 条")
        
        self._last_cleanup = current_time
    
    def get_stats(self) -> dict:
        """获取去重统计信息"""
        return {
            "cached_events": len(self._event_cache),
            "cooldown_seconds": self.cooldown_seconds,
            "last_cleanup": datetime.fromtimestamp(self._last_cleanup).isoformat(),
        }


# 全局实例
event_deduplicator = EventDeduplicator(cooldown_seconds=10)
