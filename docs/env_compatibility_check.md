# 环境兼容性检查报告

> 生成日期：2026-07-28
> 检查人：AI 算法负责人
> 所有信息均通过实际执行命令或访问官方文档获取，未伪造任何字段。

---

## 1. 本机硬件与系统环境（实测）

| 项目 | 实测值 | 命令来源 |
|------|--------|---------|
| 操作系统 | Windows 11 | `Get-CimInstance Win32_VideoController` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8 GB VRAM，Ada Lovelace 架构，compute capability 8.9） | `nvidia-smi` |
| 驱动版本 | 592.82 | `nvidia-smi` |
| 驱动支持的 CUDA Runtime 上限 | 13.1（注：此为驱动支持的最高版本，**非已安装的 CUDA Toolkit 版本**；PyTorch 自带 CUDA Runtime，无需单独安装 Toolkit） | `nvidia-smi` |
| 当前 GPU 显存占用 | 1957 MiB / 8188 MiB（系统其他进程占用） | `nvidia-smi` |
| C 盘可用空间 | 约 79 GB | `Get-PSDrive C` |
| Git | 2.54.0.windows.1 ✅ | `git --version` |
| Python（系统级） | 3.12.10 ⚠️（**与 OpenMMLab 推荐版本不匹配**） | `python --version` |
| pip | 25.0.1 ✅ | `pip --version` |
| conda | **未安装** ❌ | `conda --version` 返回未识别 |

---

## 2. OpenMMLab 官方兼容性矩阵（来源：readthedocs + GitHub README 实际访问）

### 2.1 各项目最新稳定版本

| 项目 | 最新稳定版本 | 发布日期 | 来源 |
|------|-------------|---------|------|
| mmdetection | v3.3.0 | 2024-05-01 | GitHub README "What's New" |
| mmpose | v1.3.0 | 2024-01-04 | GitHub README "What's New" |
| mmaction2 | v1.2.0 | 2023-10-12 | GitHub README "What's New" |
| mmcv | 2.2.0 | 2024 | mmcv readthedocs |

### 2.2 各项目最低依赖要求

| 项目 | PyTorch | CUDA | Python | mmcv | mmengine |
|------|---------|------|--------|------|----------|
| mmdetection 3.3.0 | 1.8+ | 9.2+ | 3.7+ | >=2.0.0 | 通过 MIM 安装 |
| mmpose 1.3.0 | 1.8+ | 9.2+ | 3.7+ | >=2.0.1 | 通过 MIM 安装 |
| mmaction2 1.2.0 | 1.8+ | 10.2+ | 3.7+ | 2.x | >=0.7.2 |

### 2.3 mmcv 2.2.0 预编译包覆盖范围

| CUDA | PyTorch 版本 |
|------|-------------|
| 12.1 | 2.4.x / 2.3.x / 2.2.x / 2.1.x |
| 11.8 | 2.4.x / 2.3.x / 2.2.x / 2.1.x |
| 11.7 | 2.x 系列 |
| 11.6 / 11.5 / 11.3 / 11.1 / 11.0 / 10.2 / 10.1 / 9.2 | 1.x / 部分 2.x |
| **12.4** | **无预编译包** ❌（需源码编译） |

**关键约束**：
- mmcv 预编译包**最高只覆盖到 CUDA 12.1 + PyTorch 2.4.x**
- 若使用 CUDA 12.4+ 的 PyTorch build，mmcv 必须源码编译，Windows 上易失败
- 推荐组合落在 `CUDA 12.1 + PyTorch 2.4.x + mmcv 2.2.0` 区间

---

## 3. Python 版本问题

### 3.1 当前状况
- 系统已装 Python 3.12.10
- OpenMMLab 文档仅写 "Python 3.7+"，**未明确支持 3.12**
- 官方文档所有示例均使用 `python=3.8`
- mmcv 预编译 wheel 在 cp310 上覆盖最全，cp311/cp312 覆盖不全

### 3.2 处置方案
- **必须**新建独立的 Python 3.10 环境（不污染系统 Python 3.12）
- conda 未安装，方案二选一：
  - **方案 A（推荐）**：安装 Miniconda，使用 conda 创建 `elderly-ai` 环境
  - **方案 B**：使用 Python 3.10 官方安装包 + venv

### 3.3 推荐方案
采用**方案 A（Miniconda）**，理由：
1. conda 可精确控制 Python 版本与依赖
2. OpenMMLab 官方文档示例全部使用 conda
3. 后续团队成员复现环境更方便

---

## 4. 最终推荐环境组合

```text
Python      : 3.10
PyTorch     : 2.4.x（CUDA 12.1 build）
CUDA Runtime: 12.1（PyTorch 自带，无需单独安装 Toolkit）
mmcv        : 2.2.0（cu121/torch2.4 预编译包）
mmengine    : 最新稳定版
mmdetection : 3.3.0
mmpose      : 1.3.0
mmaction2   : 1.2.0
```

### 4.1 安装命令（待执行）

```bash
# 1. 创建 conda 环境
conda create -n elderly-ai python=3.10 -y
conda activate elderly-ai

# 2. 安装 PyTorch 2.4.x + CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 mmcv 2.2.0 预编译包
pip install -U openmim
mim install "mmengine>=0.7.2"
mim install "mmcv==2.2.0"

# 4. 安装 mmdetection / mmpose / mmaction2
pip install "mmdet==3.3.0"
pip install "mmpose==1.3.0"
pip install "mmaction2==1.2.0"

# 5. 其他依赖
pip install opencv-python decord numpy pandas scikit-learn matplotlib pytest pyyaml requests tqdm
```

### 4.2 待验证项
- [ ] Miniconda 安装
- [ ] conda 环境创建成功
- [ ] PyTorch 能调用 CUDA（`torch.cuda.is_available()` 返回 True）
- [ ] mmcv 预编译包安装成功（不触发源码编译）
- [ ] mmdet/mmpose/mmaction2 import 成功

---

## 5. 风险与已知问题

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| mmcv 在 Windows + Python 3.10 + cu121 + torch2.4 上的预编译 wheel 实际可用性未实测 | 中 | 安装失败时回退到 torch2.1.x 或 cu118 |
| RTX 4060 Laptop 显存仅 8 GB | 中 | PoseC3D 训练需控制 batch_size；优先选 STGCN++（参数 1.39M） |
| mmaction2 v1.2.0 已 2 年未发布新版本 | 低 | 主分支仍活跃（最后提交 2026-03-18），可考虑用主分支但优先用 release |
| Python 3.12 与 OpenMMLab 兼容性未知 | 高 | 已规避：新建 Python 3.10 环境 |

---

## 6. 下一步动作

1. 安装 Miniconda（需要用户授权）
2. 创建 `elderly-ai` conda 环境（Python 3.10）
3. 按上述命令安装 PyTorch + OpenMMLab 全栈
4. 验证 `torch.cuda.is_available()` 与各包 import
5. 将环境导出为 `environment.yml` 入库
