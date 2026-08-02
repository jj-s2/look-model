"""
准备上传到 GitHub 的文件
自动排除不需要的文件和文件夹
"""

import os
import shutil
from pathlib import Path

# 需要排除的文件夹
EXCLUDE_DIRS = {
    '__pycache__',
    'node_modules',
    'venv',
    'env',
    'ENV',
    'dist',
    'build',
    '.pytest_cache',
    'htmlcov',
    '.vscode',
    '.idea',
    'logs',
}

# 需要排除的文件扩展名
EXCLUDE_EXTENSIONS = {
    '.pyc',
    '.pyo',
    '.pyd',
    '.db',
    '.sqlite3',
    '.log',
    '.swp',
    '.swo',
    '.DS_Store',
}

# 需要排除的具体文件
EXCLUDE_FILES = {
    '.env',
    '.env.local',
    'Thumbs.db',
    '.coverage',
}

def should_exclude(path: Path) -> bool:
    """判断是否应该排除该路径"""
    # 检查文件夹名
    if path.is_dir() and path.name in EXCLUDE_DIRS:
        return True
    
    # 检查文件扩展名
    if path.is_file() and path.suffix in EXCLUDE_EXTENSIONS:
        return True
    
    # 检查文件名
    if path.is_file() and path.name in EXCLUDE_FILES:
        return True
    
    return False

def count_files(directory: Path) -> dict:
    """统计文件数量和大小"""
    stats = {
        'total_files': 0,
        'total_size': 0,
        'by_type': {},
    }
    
    for item in directory.rglob('*'):
        if item.is_file() and not should_exclude(item):
            stats['total_files'] += 1
            size = item.stat().st_size
            stats['total_size'] += size
            
            ext = item.suffix or 'no_extension'
            if ext not in stats['by_type']:
                stats['by_type'][ext] = {'count': 0, 'size': 0}
            stats['by_type'][ext]['count'] += 1
            stats['by_type'][ext]['size'] += size
    
    return stats

def format_size(size: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def print_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """打印目录树"""
    if current_depth >= max_depth:
        return
    
    items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    for i, item in enumerate(items):
        if should_exclude(item):
            continue
        
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(item, next_prefix, max_depth, current_depth + 1)

def main():
    """主函数"""
    print("=" * 60)
    print("GitHub 上传准备工具")
    print("=" * 60)
    print()
    
    # 获取项目根目录
    root = Path.cwd()
    print(f"项目根目录: {root}")
    print()
    
    # 统计信息
    print("📊 文件统计:")
    print("-" * 60)
    stats = count_files(root)
    print(f"总文件数: {stats['total_files']}")
    print(f"总大小: {format_size(stats['total_size'])}")
    print()
    
    print("按类型统计:")
    for ext, info in sorted(stats['by_type'].items(), key=lambda x: x[1]['size'], reverse=True)[:10]:
        print(f"  {ext:15} {info['count']:4} 个文件, {format_size(info['size']):>10}")
    print()
    
    # 显示目录结构
    print("📁 将要上传的目录结构:")
    print("-" * 60)
    print_tree(root, max_depth=3)
    print()
    
    # 检查敏感文件
    print("🔒 敏感文件检查:")
    print("-" * 60)
    sensitive_found = False
    
    for pattern in ['.env', '*.db', '*.sqlite3']:
        for file in root.rglob(pattern):
            if not should_exclude(file):
                print(f"⚠️  发现: {file.relative_to(root)}")
                sensitive_found = True
    
    if not sensitive_found:
        print("✅ 未发现敏感文件")
    print()
    
    # 检查必需文件
    print("📋 必需文件检查:")
    print("-" * 60)
    required_files = [
        'README.md',
        '.gitignore',
        'backend/requirements.txt',
        'backend/.env.example',
        'frontend/package.json',
    ]
    
    all_present = True
    for file_path in required_files:
        file = root / file_path
        if file.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
            all_present = False
    print()
    
    # 大文件警告
    print("📦 大文件检查 (>10MB):")
    print("-" * 60)
    large_files_found = False
    
    for item in root.rglob('*'):
        if item.is_file() and not should_exclude(item):
            size = item.stat().st_size
            if size > 10 * 1024 * 1024:  # 10 MB
                print(f"⚠️  {item.relative_to(root)}: {format_size(size)}")
                large_files_found = True
    
    if not large_files_found:
        print("✅ 无大文件")
    print()
    
    # 建议
    print("💡 上传建议:")
    print("-" * 60)
    if sensitive_found:
        print("⚠️  请手动删除或移动敏感文件")
    if not all_present:
        print("⚠️  请确保所有必需文件都存在")
    if large_files_found:
        print("⚠️  考虑使用 Git LFS 处理大文件")
    
    print()
    print("✅ 可以使用以下命令上传到 GitHub:")
    print()
    print("  git init")
    print("  git add .")
    print("  git commit -m 'Initial commit: 智慧养老监护平台'")
    print("  git remote add origin https://github.com/jj-s2/look-model.git")
    print("  git branch -M main")
    print("  git push -u origin main")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
