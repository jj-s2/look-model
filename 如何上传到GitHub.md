# 如何上传项目到 GitHub

## 方法一：自动上传（推荐）⭐

### 前提条件
1. 已安装 Git：https://git-scm.com/download/win
2. 已登录 GitHub 账号

### 操作步骤

1. **双击运行批处理文件**
   
   直接双击 `upload_to_github.bat` 文件
   
   或在命令行中运行：
   ```cmd
   # 如果使用 CMD
   upload_to_github.bat
   
   # 如果使用 PowerShell
   .\upload_to_github.bat
   ```

2. **等待检查完成**
   - 工具会自动检查 Git 安装
   - 显示将要上传的文件列表

3. **确认上传**
   - 输入 `Y` 确认提交
   - 首次上传需要在浏览器中登录 GitHub

4. **完成**
   - 等待上传完成
   - 访问 https://github.com/jj-s2/look-model 查看

---

## 方法二：通过 GitHub Desktop（最简单）

### 步骤

1. **安装 GitHub Desktop**
   - 下载：https://desktop.github.com/
   - 安装并登录 GitHub 账号

2. **添加项目**
   - 打开 GitHub Desktop
   - File → Add Local Repository
   - 选择项目文件夹

3. **提交更改**
   - 查看左侧文件列表
   - 填写提交信息：`初始提交：智慧养老监护平台`
   - 点击 "Commit to main"

4. **发布到 GitHub**
   - 点击 "Publish repository"
   - 仓库名：`look-model`
   - 组织：`jj-s2`
   - 取消勾选 "Keep this code private"（如果要公开）
   - 点击 "Publish Repository"

---

## 方法三：命令行上传（适合熟悉 Git 的用户）

### 步骤

```bash
# 1. 初始化 Git 仓库
git init

# 2. 添加远程仓库
git remote add origin https://github.com/jj-s2/look-model.git

# 3. 查看将要上传的文件
git status

# 4. 添加所有文件
git add .

# 5. 提交
git commit -m "Initial commit: 智慧养老监护平台"

# 6. 设置主分支
git branch -M main

# 7. 推送到 GitHub
git push -u origin main
```

### 如果远程仓库已有内容

```bash
# 先拉取远程内容
git pull origin main --allow-unrelated-histories

# 解决冲突（如果有）

# 再次推送
git push -u origin main
```

---

## 方法四：通过 GitHub 网页上传（分批上传）

### 适用场景
- 无法安装 Git
- 网络环境限制
- 只想上传部分文件

### 步骤

1. **访问仓库**
   - 打开 https://github.com/jj-s2/look-model

2. **创建文件夹并上传**
   - 点击 "Add file" → "Upload files"
   - 拖拽文件夹到上传区域
   - 分批上传：
     - 第一批：backend/app/
     - 第二批：frontend/src/
     - 第三批：deploy/, docs/, mock/, scripts/
     - 第四批：根目录配置文件

3. **逐个上传根文件**
   - README.md
   - .gitignore
   - package.json 等

**⚠️ 注意**：
- 单次上传不要超过 100 个文件
- 每个文件不能超过 25 MB
- 上传 node_modules 会非常慢（不建议上传）

---

## 方法五：使用 VS Code（如果已安装）

### 步骤

1. **打开项目**
   - 在 VS Code 中打开项目文件夹

2. **初始化 Git**
   - 点击左侧 "Source Control" 图标
   - 点击 "Initialize Repository"

3. **暂存所有更改**
   - 点击 "Changes" 旁边的 "+"
   - 或右键选择 "Stage All Changes"

4. **提交**
   - 输入提交信息：`Initial commit: 智慧养老监护平台`
   - 按 Ctrl+Enter 或点击 ✓

5. **推送到 GitHub**
   - 点击 "..." → "Remote" → "Add Remote"
   - 输入 URL：`https://github.com/jj-s2/look-model.git`
   - 点击 "..." → "Push"
   - 登录 GitHub 账号

---

## 🔍 上传前检查清单

### 必须检查
- [ ] 已删除 `.env` 文件（包含敏感密钥）
- [ ] 已删除 `elderly_monitor.db` 数据库文件
- [ ] 已删除 `backend/venv/` 虚拟环境
- [ ] 已删除 `frontend/node_modules/` 依赖包
- [ ] 确认 `.gitignore` 配置正确

### 建议检查
- [ ] README.md 内容完整
- [ ] .env.example 使用占位符
- [ ] requirements.txt 完整
- [ ] package.json 完整

### 运行检查脚本

```bash
python prepare_upload.py
```

这个脚本会自动检查：
- ✅ 敏感文件是否已排除
- ✅ 必需文件是否存在
- ✅ 是否有超大文件
- ✅ 文件统计和目录结构

---

## ❓ 常见问题

### Q1: 提示 "git 不是内部或外部命令"

**解决方案：**
1. 下载安装 Git：https://git-scm.com/download/win
2. 安装时选择 "Git from the command line and also from 3rd-party software"
3. 重启命令行窗口

### Q2: 推送时提示需要登录

**解决方案：**
1. 第一次推送会弹出登录窗口
2. 使用 GitHub 账号登录
3. 如果使用两步验证，需要生成 Personal Access Token
   - 访问：https://github.com/settings/tokens
   - 生成 token 并复制
   - 使用 token 作为密码登录

### Q3: 提示 "remote origin already exists"

**解决方案：**
```bash
# 删除现有远程仓库
git remote remove origin

# 重新添加
git remote add origin https://github.com/jj-s2/look-model.git
```

### Q4: 提示 "Updates were rejected"

**解决方案：**
```bash
# 强制推送（谨慎使用，会覆盖远程内容）
git push -u origin main --force

# 或者先拉取再推送
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Q5: 上传很慢或失败

**解决方案：**
1. 检查网络连接
2. 确认已删除大文件（venv/, node_modules/）
3. 使用代理或 VPN
4. 分批上传（使用 GitHub 网页）

### Q6: 不小心上传了敏感文件

**解决方案：**
```bash
# 从 Git 历史中删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch backend/.env" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin --force --all
```

---

## 📚 相关资源

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub 使用指南](https://docs.github.com/cn)
- [Git 命令速查表](https://training.github.com/downloads/zh_CN/github-git-cheat-sheet/)
- [GitHub Desktop 文档](https://docs.github.com/cn/desktop)

---

## 🎯 推荐方案

### 新手推荐
1. 使用 **GitHub Desktop**（可视化界面，最简单）
2. 或使用 **自动上传脚本**（upload_to_github.bat）

### 有经验用户推荐
1. 使用 **命令行**（灵活、快速）
2. 或使用 **VS Code**（编辑器集成）

### 网络受限推荐
1. 使用 **GitHub 网页上传**（分批上传）
2. 考虑使用代理加速

---

## ✅ 上传成功后

1. **访问仓库**
   - https://github.com/jj-s2/look-model

2. **检查内容**
   - 确认所有文件都已上传
   - README.md 正确显示

3. **测试克隆**
   ```bash
   git clone https://github.com/jj-s2/look-model.git
   cd look-model
   ```

4. **添加说明**
   - 编辑仓库描述
   - 添加主题标签（tags）
   - 设置仓库可见性

5. **创建 Release**（可选）
   - 为重要版本创建 Release
   - 添加版本说明

---

**祝上传顺利！🚀**
