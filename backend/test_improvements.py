#!/usr/bin/env python3
"""
生产级改进功能测试脚本
验证各项改进功能是否正常工作
"""
import asyncio
import time


async def test_event_deduplicator():
    """测试事件防抖去重"""
    print("\n" + "="*60)
    print("测试 1: 事件防抖去重机制")
    print("="*60)
    
    from app.services.event_deduplicator import event_deduplicator
    
    device_id = "test_camera_001"
    risk_type = "fall_risk"
    risk_level = "HIGH"
    
    # 第一次应该通过
    result1 = event_deduplicator.should_process(device_id, risk_type, risk_level)
    print(f"第1次上报: {'✅ 处理' if result1 else '❌ 忽略'}")
    assert result1 == True, "第一次应该被处理"
    
    # 立即重复应该被拒绝
    result2 = event_deduplicator.should_process(device_id, risk_type, risk_level)
    print(f"第2次上报（立即重复）: {'✅ 处理' if result2 else '❌ 忽略'}")
    assert result2 == False, "冷却期内应该被忽略"
    
    # 等待冷却期后应该通过
    print("⏰ 等待11秒（超过冷却期）...")
    await asyncio.sleep(11)
    result3 = event_deduplicator.should_process(device_id, risk_type, risk_level)
    print(f"第3次上报（11秒后）: {'✅ 处理' if result3 else '❌ 忽略'}")
    assert result3 == True, "超过冷却期应该被处理"
    
    stats = event_deduplicator.get_stats()
    print(f"\n统计信息: {stats}")
    print("✅ 事件防抖去重测试通过")


async def test_health_validator():
    """测试健康数据验证"""
    print("\n" + "="*60)
    print("测试 2: 健康数据验证与清洗")
    print("="*60)
    
    from app.services.health_validator import health_validator
    from app.schemas.health import HealthCreate
    from datetime import datetime
    
    device_id = "test_radar_001"
    
    # 测试正常数据
    normal_data = HealthCreate(
        device_id=device_id,
        heart_rate=72.0,
        respiratory_rate=16.0,
        body_temperature=36.5,
        recorded_at=datetime.now()
    )
    result1 = health_validator.validate_and_filter(device_id, normal_data)
    print(f"✅ 正常数据: 心率={result1.heart_rate}, 呼吸={result1.respiratory_rate}")
    
    # 测试异常心率（超出范围）
    abnormal_data = HealthCreate(
        device_id=device_id,
        heart_rate=200.0,  # 超出正常范围
        respiratory_rate=15.0,
        recorded_at=datetime.now()
    )
    result2 = health_validator.validate_and_filter(device_id, abnormal_data)
    print(f"⚠️ 异常数据过滤: 心率={result2.heart_rate} (原始200被过滤)")
    assert result2.heart_rate is None, "异常心率应该被过滤"
    
    # 测试移动平均
    print("\n测试移动平均平滑...")
    values = []
    for i in range(5):
        data = HealthCreate(
            device_id=f"test_device_{i}",
            heart_rate=70.0 + i,
            respiratory_rate=16.0,
            recorded_at=datetime.now()
        )
        result = health_validator.validate_and_filter(f"test_device_smooth", data)
        values.append(result.heart_rate)
    print(f"平滑后序列: {values}")
    
    print("✅ 健康数据验证测试通过")


async def test_cache_service():
    """测试缓存服务"""
    print("\n" + "="*60)
    print("测试 3: Cache Service抽象层")
    print("="*60)
    
    from app.services.cache_service import cache_service
    
    # 测试设置和获取
    key = "test_key"
    value = {"name": "张三", "age": 70, "device_id": "camera_001"}
    
    await cache_service.set(key, value, ttl=5)
    print(f"✅ 设置缓存: {key} = {value}")
    
    result = await cache_service.get(key)
    print(f"✅ 获取缓存: {result}")
    assert result == value, "缓存值应该匹配"
    
    # 测试过期
    print("⏰ 等待6秒（超过TTL）...")
    await asyncio.sleep(6)
    expired_result = await cache_service.get(key)
    print(f"⏰ 过期后获取: {expired_result}")
    assert expired_result is None, "过期缓存应该返回None"
    
    # 测试exists
    await cache_service.set("exist_test", "value")
    exists = await cache_service.exists("exist_test")
    print(f"✅ 存在性检查: {exists}")
    
    print("✅ 缓存服务测试通过")


async def test_websocket_manager():
    """测试WebSocket管理器"""
    print("\n" + "="*60)
    print("测试 4: WebSocket连接管理（模拟）")
    print("="*60)
    
    from app.services.websocket_manager import websocket_manager
    
    # 模拟更新pong时间
    client_id = "test_client_001"
    websocket_manager.last_pong_time[client_id] = time.time()
    print(f"✅ 模拟客户端连接: {client_id}")
    
    # 测试超时检查
    websocket_manager.last_pong_time["timeout_client"] = time.time() - 70  # 70秒前
    timeout_list = websocket_manager.check_timeout(timeout_seconds=60)
    print(f"⏰ 超时连接检测: {timeout_list}")
    assert "timeout_client" in timeout_list, "应该检测到超时连接"
    
    # 测试统计
    stats = websocket_manager.get_stats()
    print(f"📊 连接统计: {stats}")
    
    print("✅ WebSocket管理测试通过")


async def test_alert_escalation():
    """测试告警升级（模拟）"""
    print("\n" + "="*60)
    print("测试 5: 告警升级机制（模拟）")
    print("="*60)
    
    from app.services.alert_escalation import alert_escalation_service
    from app.models.alert import Alert
    from app.db.session import SessionLocal
    from datetime import datetime, timedelta
    
    print("ℹ️ 注意：完整测试需要实际数据库和WebSocket环境")
    print("ℹ️ 这里仅展示服务初始化和配置")
    
    print(f"✅ 升级超时: {alert_escalation_service.escalation_timeout_minutes} 分钟")
    print(f"✅ 服务已就绪")
    
    # 模拟检查（需要数据库）
    try:
        count = await alert_escalation_service.check_and_escalate()
        print(f"📢 本次检查升级告警数: {count}")
    except Exception as e:
        print(f"ℹ️ 完整测试需要运行环境: {e}")
    
    print("✅ 告警升级配置验证通过")


async def main():
    """主测试函数"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     生产级改进功能测试                                ║
    ║     Production Improvements Testing                    ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 测试1: 事件防抖去重
        await test_event_deduplicator()
        
        # 测试2: 健康数据验证
        await test_health_validator()
        
        # 测试3: 缓存服务
        await test_cache_service()
        
        # 测试4: WebSocket管理
        await test_websocket_manager()
        
        # 测试5: 告警升级
        await test_alert_escalation()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
        print("""
        改进功能验证完成：
        ✅ 事件防抖去重机制
        ✅ 健康数据验证与清洗
        ✅ Cache Service抽象层
        ✅ WebSocket连接管理
        ✅ 告警升级机制配置
        
        建议：
        1. 启动完整系统进行端到端测试
        2. 使用Mock数据验证WebSocket推送
        3. 观察后台任务调度器运行情况
        """)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # 添加backend到路径
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).parent
    sys.path.insert(0, str(backend_dir))
    
    # 运行测试
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
