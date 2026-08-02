"""健康数据验证与清洗服务"""
from typing import Optional
from collections import deque
from app.schemas.health import HealthCreate


class HealthDataValidator:
    """健康数据验证器
    
    对雷达等传感器数据进行异常值过滤和移动平均平滑处理
    """
    
    # 正常生理指标范围
    HEART_RATE_RANGE = (30, 180)         # 心率范围 (次/分钟)
    RESPIRATORY_RATE_RANGE = (8, 30)     # 呼吸率范围 (次/分钟)
    BODY_TEMP_RANGE = (35.0, 42.0)       # 体温范围 (摄氏度)
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: 移动平均窗口大小
        """
        self.window_size = window_size
        # 按设备存储历史数据窗口
        self._heart_rate_windows = {}
        self._respiratory_rate_windows = {}
    
    def validate_and_filter(self, device_id: str, data: HealthCreate) -> HealthCreate:
        """验证并过滤健康数据
        
        Args:
            device_id: 设备ID
            data: 原始健康数据
            
        Returns:
            清洗后的健康数据
        """
        # 心率验证与平滑
        if data.heart_rate is not None:
            data.heart_rate = self._validate_heart_rate(device_id, data.heart_rate)
        
        # 呼吸率验证与平滑
        if data.respiratory_rate is not None:
            data.respiratory_rate = self._validate_respiratory_rate(device_id, data.respiratory_rate)
        
        # 体温验证
        if data.body_temperature is not None:
            data.body_temperature = self._validate_body_temperature(data.body_temperature)
        
        return data
    
    def _validate_heart_rate(self, device_id: str, value: float) -> Optional[float]:
        """验证心率数据"""
        # 异常值检测
        if not (self.HEART_RATE_RANGE[0] <= value <= self.HEART_RATE_RANGE[1]):
            print(f"⚠️ 心率异常值被过滤: {value} (正常范围: {self.HEART_RATE_RANGE})")
            return None
        
        # 移动平均平滑
        if device_id not in self._heart_rate_windows:
            self._heart_rate_windows[device_id] = deque(maxlen=self.window_size)
        
        window = self._heart_rate_windows[device_id]
        
        # 突变检测：与最近平均值差异过大
        if len(window) >= 3:
            avg = sum(window) / len(window)
            if abs(value - avg) > 30:  # 突变超过30次/分
                print(f"⚠️ 心率突变被过滤: {value} (平均: {avg:.1f})")
                return None
        
        window.append(value)
        
        # 返回移动平均值
        smoothed = sum(window) / len(window)
        return round(smoothed, 1)
    
    def _validate_respiratory_rate(self, device_id: str, value: float) -> Optional[float]:
        """验证呼吸率数据"""
        # 异常值检测
        if not (self.RESPIRATORY_RATE_RANGE[0] <= value <= self.RESPIRATORY_RATE_RANGE[1]):
            print(f"⚠️ 呼吸率异常值被过滤: {value} (正常范围: {self.RESPIRATORY_RATE_RANGE})")
            return None
        
        # 移动平均平滑
        if device_id not in self._respiratory_rate_windows:
            self._respiratory_rate_windows[device_id] = deque(maxlen=self.window_size)
        
        window = self._respiratory_rate_windows[device_id]
        
        # 突变检测：与最近平均值差异过大
        if len(window) >= 3:
            avg = sum(window) / len(window)
            if abs(value - avg) > 10:  # 突变超过10次/分
                print(f"⚠️ 呼吸率突变被过滤: {value} (平均: {avg:.1f})")
                return None
        
        window.append(value)
        
        # 返回移动平均值
        smoothed = sum(window) / len(window)
        return round(smoothed, 1)
    
    def _validate_body_temperature(self, value: float) -> Optional[float]:
        """验证体温数据"""
        if not (self.BODY_TEMP_RANGE[0] <= value <= self.BODY_TEMP_RANGE[1]):
            print(f"⚠️ 体温异常值被过滤: {value} (正常范围: {self.BODY_TEMP_RANGE})")
            return None
        return round(value, 1)
    
    def clear_history(self, device_id: str):
        """清除指定设备的历史数据"""
        if device_id in self._heart_rate_windows:
            del self._heart_rate_windows[device_id]
        if device_id in self._respiratory_rate_windows:
            del self._respiratory_rate_windows[device_id]


# 全局实例
health_validator = HealthDataValidator(window_size=10)
