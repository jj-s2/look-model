"""直接创建数据库表（不使用Alembic）"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.db.session import engine, Base
from app.models import Device, RiskRecord, Alert, HealthRecord


def create_tables():
    """创建所有数据库表"""
    print("开始创建数据库表...")
    
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        print("✓ 数据库表创建成功！")
        
        # 显示创建的表
        print("\n已创建的表:")
        for table in Base.metadata.sorted_tables:
            print(f"  - {table.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ 创建表失败: {e}")
        return False


def drop_tables():
    """删除所有数据库表（谨慎使用）"""
    confirm = input("⚠️  确定要删除所有表吗？这将清空所有数据！(yes/no): ")
    if confirm.lower() != 'yes':
        print("已取消")
        return False
    
    try:
        Base.metadata.drop_all(bind=engine)
        print("✓ 数据库表已删除")
        return True
    except Exception as e:
        print(f"✗ 删除表失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库表管理")
    parser.add_argument("action", choices=["create", "drop", "recreate"], 
                       help="操作: create(创建), drop(删除), recreate(重建)")
    
    args = parser.parse_args()
    
    if args.action == "create":
        create_tables()
    elif args.action == "drop":
        drop_tables()
    elif args.action == "recreate":
        if drop_tables():
            create_tables()
