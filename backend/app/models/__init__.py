"""数据库模型"""
from app.models.device import Device
from app.models.risk import RiskRecord
from app.models.alert import Alert
from app.models.health import HealthRecord

__all__ = ["Device", "RiskRecord", "Alert", "HealthRecord"]
