@echo off
chcp 65001 >nul
echo ========================================
echo GitHub 自动上传工具
echo ========================================
echo.

REM 检查是否安装 Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未安装 Git
    echo 请先下载并安装 Git: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo ✅ Git 已安装
echo.

REM 检查是否已经初始化
if exist .git (
    echo 📁 Git 仓库已存在
    echo.
) else (
    echo 📁 初始化 Git 仓库...
    git init
    echo.
)

REM 检查远程仓库
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 🔗 添加远程仓库...
    git remote add origin https://github.com/jj-s2/look-model.git
    echo.
) else (
    echo ✅ 远程仓库已配置
    echo.
)

REM 添加所有文件
echo 📦 添加文件到 Git...
git add .
echo.

REM 显示将要提交的文件
echo 📋 将要提交的文件:
git status --short
echo.

REM 确认提交
set /p confirm="是否继续提交? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo 已取消
    pause
    exit /b 0
)

REM 提交
echo.
echo 💾 提交更改...
git commit -m "Initial commit: 智慧养老监护平台"
if errorlevel 1 (
    echo.
    echo ⚠️  可能没有新的更改需要提交
    echo.
)

REM 设置主分支
echo.
echo 🌳 设置主分支为 main...
git branch -M main
echo.

REM 推送到 GitHub
echo 📤 推送到 GitHub...
echo.
echo ⚠️  首次推送需要登录 GitHub 账号
echo    请在弹出的窗口中完成登录
echo.
pause

git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ 推送失败
    echo.
    echo 可能的原因:
    echo 1. 网络连接问题
    echo 2. 需要 GitHub 登录验证
    echo 3. 远程仓库已存在内容
    echo.
    echo 如果远程仓库已有内容，可以尝试:
    echo   git pull origin main --allow-unrelated-histories
    echo   git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 上传成功！
echo ========================================
echo.
echo 访问你的仓库: https://github.com/jj-s2/look-model
echo.
pause
