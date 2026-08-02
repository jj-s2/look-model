# GitHub 上传指南

## 需要上传的文件和文件夹

### ✅ 必须上传

#### 根目录
- `README.md` - 项目说明文档
- `.gitignore` - Git 忽略配置

#### backend/ 后端代码
```
backend/
├── app/                    # 所有源代码
│   ├── api/               # API 路由
│   ├── db/                # 数据库配置
│   ├── models/            # 数据模型
│   ├── schemas/           # Pydantic 模型
│   ├── services/          # 业务服务
│   ├── static/            # 静态文件目录（空目录也要保留）
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   └── __init__.py
├── migrations/            # 数据库迁移脚本
├── tests/                 # 测试文件（即使为空也保留）
├── .env.example           # 环境变量示例
├── .gitignore
├── create_tables.py       # 数据库初始化
├── seed_data.py           # 种子数据
├── seed_demo_data.py      # 演示数据
├── requirements.txt       # Python 依赖
└── 生成演示数据.py        # 中文版数据生成
```

#### frontend/ 前端代码
```
frontend/
├── public/                # 公共资源
├── src/                   # 所有源代码
│   ├── components/       # 组件（如果有）
│   ├── hooks/            # React Hooks
│   ├── pages/            # 页面组件
│   ├── services/         # API 服务
│   ├── types/            # TypeScript 类型
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

#### deploy/ 部署配置
```
deploy/
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── nginx.conf
```

#### docs/ 文档
```
docs/
├── 快速启动指南.md
└── 生产级改进建议.md
```

#### mock/ 模拟设备
```
mock/
├── mock_camera_events.json
├── mock_device_server.py
└── mock_radar_data.json
```

#### scripts/ 脚本工具
```
scripts/
├── diagnose_frontend.py
├── send_bedroom_alert.py
└── send_complete_event.py
```

#### 启动脚本
- `_start_backend_direct.bat`
- `_start_frontend_direct.bat`
- `直接启动（无需安装）.bat`

#### 中文文档
- `测试指南_查看前端效果.md`

---

## ❌ 不要上传的文件和文件夹

### 自动生成的文件
- `backend/__pycache__/` - Python 字节码缓存
- `backend/app/**/__pycache__/` - 所有 __pycache__ 目录
- `frontend/node_modules/` - Node 依赖包
- `frontend/dist/` - 构建输出

### 虚拟环境
- `backend/venv/` - Python 虚拟环境
- `backend/env/` - Python 虚拟环境

### 数据库文件
- `backend/elderly_monitor.db` - SQLite 数据库文件
- `backend/*.sqlite3` - 其他数据库文件

### 敏感配置
- `backend/.env` - 实际环境变量（包含密钥）
- `.env.local` - 本地环境变量

### IDE 配置
- `.vscode/` - VS Code 配置
- `.idea/` - IntelliJ IDEA 配置
- `*.swp`, `*.swo` - Vim 临时文件

### 日志文件
- `*.log` - 所有日志文件
- `logs/` - 日志目录

### 临时文件
- `.DS_Store` - macOS 文件
- `Thumbs.db` - Windows 文件

### 测试覆盖率
- `.coverage` - 覆盖率数据
- `htmlcov/` - 覆盖率报告
- `.pytest_cache/` - Pytest 缓存

---

## 📋 上传前检查清单

### 1. 清理敏感信息
- [ ] 确保 `.env` 文件未被包含
- [ ] 检查代码中没有硬编码的密钥
- [ ] `.env.example` 中使用占位符

### 2. 确认 .gitignore 配置
```bash
# 检查 .gitignore 是否生效
cd backend
git status
```

应该看不到以下内容：
- `__pycache__/`
- `venv/`
- `.env`
- `*.db`

### 3. 测试文件完整性
- [ ] README.md 完整且清晰
- [ ] requirements.txt 包含所有依赖
- [ ] package.json 完整
- [ ] .env.example 包含所有必需变量

### 4. 验证代码可运行
```bash
# 测试后端
cd backend
pip install -r requirements.txt
python create_tables.py

# 测试前端
cd frontend
npm install
npm run build
```

---

## 🚀 上传步骤

### 方法一：通过 GitHub 网页上传

1. 访问 https://github.com/jj-s2/look-model/upload/web
2. 直接拖拽以下文件夹到上传区域：
   - `backend/app/`
   - `backend/migrations/`
   - `backend/tests/`
   - `frontend/src/`
   - `frontend/public/`
   - `deploy/`
   - `docs/`
   - `mock/`
   - `scripts/`
3. 逐个上传根目录文件：
   - `README.md`
   - `.gitignore`
   - `backend/.env.example`
   - `backend/.gitignore`
   - `backend/requirements.txt`
   - `backend/create_tables.py`
   - `backend/seed_data.py`
   - `backend/seed_demo_data.py`
   - `frontend/package.json`
   - `frontend/package-lock.json`
   - `frontend/tsconfig.json`
   - `frontend/vite.config.ts`
   - `frontend/index.html`
   - 所有 `.bat` 和 `.md` 文件

### 方法二：使用自动上传脚本（最快）

**直接双击** `upload_to_github.bat` 文件

或在命令行中运行：
```cmd
# 如果使用 CMD
upload_to_github.bat

# 如果使用 PowerShell
.\upload_to_github.bat
```

### 方法三：通过 Git 命令行（推荐有经验用户）

```bash
# 1. 初始化 Git 仓库（如果还没有）
git init

# 2. 添加远程仓库
git remote add origin https://github.com/jj-s2/look-model.git

# 3. 添加所有文件（.gitignore 会自动排除不需要的）
git add .

# 4. 查看将要提交的文件
git status

# 5. 提交
git commit -m "Initial commit: 智慧养老监护平台"

# 6. 推送到 GitHub
git branch -M main
git push -u origin main
```

---

## 📦 文件大小建议

### 单个文件大小限制
- GitHub 单文件不要超过 100 MB
- 建议单文件不超过 10 MB

### 需要注意的大文件
如果有以下文件过大，考虑使用 Git LFS：
- 演示视频
- 大型图片资源
- 二进制文件

---

## ✨ 上传后的优化建议

### 1. 添加 GitHub Actions（可选）
创建 `.github/workflows/ci.yml` 实现自动测试和部署

### 2. 添加徽章（可选）
在 README.md 顶部添加项目状态徽章：
```markdown
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

### 3. 创建 Releases
为重要版本创建 Release 标签

### 4. 添加 Issues 模板
创建 `.github/ISSUE_TEMPLATE/` 规范问题反馈

---

## 🔍 上传后验证

1. 访问你的仓库页面
2. 检查 README.md 是否正确显示
3. 确认目录结构完整
4. 测试克隆后能否正常运行：
   ```bash
   git clone https://github.com/jj-s2/look-model.git
   cd look-model
   # 按 README.md 说明启动
   ```

---

## 📞 需要帮助？

如果上传过程中遇到问题：
1. 检查文件大小是否超限
2. 确认网络连接稳定
3. 验证 GitHub 账号权限
4. 查看 Git 错误信息

---

**提示**：首次上传建议使用 Git 命令行，可以更好地控制上传内容和解决冲突。
