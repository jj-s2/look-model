# 智慧养老监护平台

一个基于 AI 视觉分析和毫米波雷达的智能养老监护系统，实时监测老年人的安全状态，提供跌倒风险预警、健康数据监测和告警管理功能。

## 项目特点

- 🎥 **多设备接入**：支持萤石摄像头和 SDNL1 毫米波雷达
- 🤖 **AI 智能分析**：实时视觉分析，识别跌倒风险和异常行为
- 📊 **健康监测**：毫米波雷达监测心率、呼吸率等生命体征
- 🔔 **智能告警**：多级告警机制，WebSocket 实时推送
- 📈 **数据可视化**：风险趋势分析、设备状态监控
- 🏥 **适老化设计**：大字体、清晰配色、简洁界面

## 技术栈

### 后端
- **FastAPI**：高性能 Python Web 框架
- **SQLAlchemy**：ORM 数据库操作
- **PostgreSQL / SQLite**：关系型数据库
- **WebSocket**：实时双向通信
- **APScheduler**：定时任务调度

### 前端
- **React 18 + TypeScript**：现代化前端框架
- **Vite**：快速构建工具
- **Recharts**：数据可视化
- **TailwindCSS**：实用优先的 CSS 框架

### 部署
- **Docker + Docker Compose**：容器化部署
- **Nginx**：反向代理和负载均衡

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- PostgreSQL 15+ (可选，也可使用 SQLite)

### Windows 系统一键启动

项目提供了 Windows 批处理脚本，可以快速启动：

```cmd
直接启动（无需安装）.bat
```

### 手动启动

#### 1. 启动后端

```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

编辑 `.env` 文件，配置数据库和萤石 API 密钥：

```env
DATABASE_URL=sqlite:///./elderly_monitor.db
EZVIZ_APP_KEY=your_app_key_here
EZVIZ_APP_SECRET=your_app_secret_here
```

初始化数据库并启动：

```cmd
python create_tables.py
python seed_demo_data.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. 启动前端

新开命令行窗口：

```cmd
cd frontend
npm install
npm run dev
```

#### 3. 启动模拟设备（可选）

用于测试，模拟设备数据推送：

```cmd
cd mock
pip install httpx
python mock_device_server.py
```

### 访问系统

- 🌐 前端界面：http://localhost:3000
- 📚 API 文档：http://localhost:8000/docs
- 💓 健康检查：http://localhost:8000/health

## Docker 部署

```bash
cd deploy
docker-compose up -d
```

访问 http://localhost:80

## 项目结构

```
├── backend/                # 后端服务
│   ├── app/
│   │   ├── api/           # API 路由
│   │   ├── models/        # 数据库模型
│   │   ├── schemas/       # Pydantic 模型
│   │   ├── services/      # 业务逻辑服务
│   │   └── db/            # 数据库配置
│   ├── migrations/        # 数据库迁移
│   └── requirements.txt   # Python 依赖
│
├── frontend/              # 前端应用
│   ├── src/
│   │   ├── pages/        # 页面组件
│   │   ├── hooks/        # React Hooks
│   │   ├── services/     # API 服务
│   │   └── types/        # TypeScript 类型
│   └── package.json      # Node 依赖
│
├── deploy/               # 部署配置
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── mock/                 # 模拟设备
│   └── mock_device_server.py
│
└── docs/                 # 文档
    ├── 快速启动指南.md
    └── 生产级改进建议.md
```

## 核心功能

### 1. 设备管理
- 支持摄像头和雷达设备接入
- 设备在线状态监控
- 设备配置管理

### 2. 实时监控
- 视频流实时播放
- AI 风险分数实时计算
- WebSocket 实时数据推送

### 3. 告警中心
- 多级告警（低/中/高/紧急）
- 告警状态流转管理
- 自动抓拍存证
- 超时升级机制

### 4. 健康监测
- 心率、呼吸率监测
- 睡眠状态分析
- 历史数据趋势图

### 5. 风险分析
- 24 小时风险趋势
- 多维度风险统计
- 异常行为识别

## API 文档

启动后端后访问 http://localhost:8000/docs 查看完整 API 文档。

主要接口：

- `GET /api/v1/devices` - 获取设备列表
- `POST /api/v1/devices` - 创建设备
- `GET /api/v1/alerts` - 获取告警列表
- `POST /api/v1/alerts/{id}/confirm` - 确认告警
- `GET /api/v1/risks/trend` - 获取风险趋势
- `WS /ws/{client_id}` - WebSocket 连接

## 配置说明

### 环境变量配置

后端 `.env` 文件配置项：

```env
# 萤石开放平台
EZVIZ_APP_KEY=your_app_key_here
EZVIZ_APP_SECRET=your_app_secret_here

# 数据库
DATABASE_URL=sqlite:///./elderly_monitor.db

# 服务配置
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# 告警配置
ALERT_HIGH_THRESHOLD=0.8
ALERT_MEDIUM_THRESHOLD=0.6
ALERT_LOW_THRESHOLD=0.3
```

### 数据库配置

支持 PostgreSQL 和 SQLite：

```env
# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/elderly_monitor

# SQLite (开发环境)
DATABASE_URL=sqlite:///./elderly_monitor.db
```

## 开发指南

### 添加新的 API 接口

1. 在 `backend/app/models/` 创建数据模型
2. 在 `backend/app/schemas/` 创建 Pydantic 模型
3. 在 `backend/app/services/` 实现业务逻辑
4. 在 `backend/app/api/` 创建路由

### 添加新的前端页面

1. 在 `frontend/src/pages/` 创建页面组件
2. 在 `frontend/src/services/api.ts` 添加 API 调用
3. 在 `frontend/src/types/index.ts` 定义类型
4. 在 `frontend/src/App.tsx` 添加路由

## 测试

### 后端测试

```cmd
cd backend
pytest
```

### 前端测试

```cmd
cd frontend
npm test
```

## 生产部署建议

参考 `docs/生产级改进建议.md`，包含：

- 防抖去重机制
- 视频流低延迟优化
- 告警超时升级
- 缓存层设计
- 数据清洗与过滤
- WebSocket 心跳机制

## 常见问题

### Q1: 后端启动失败，提示数据库连接错误
确保 PostgreSQL 服务正在运行，或使用 SQLite 配置。

### Q2: 前端无法连接后端
检查后端是否在 8000 端口启动，查看浏览器控制台错误。

### Q3: WebSocket 连接失败
确保后端正常运行，防火墙未阻止 WebSocket 连接。

### Q4: 视频流无法播放
检查萤石 API 密钥是否正确，设备是否在线。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎联系项目维护者。

---

**注意**：本项目为演示系统，生产环境使用前请参考改进建议进行增强。
