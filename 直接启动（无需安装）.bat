@echo off
chcp 65001 >nul
title Direct Start System (Skip Installation)

echo.
echo ============================================================
echo   Direct Start System (Skip Dependency Check)
echo ============================================================
echo.

cd /d "%~dp0"

REM Check virtual environment
if not exist "backend\venv" (
    echo [ERROR] Virtual environment not found
    echo [TIP] Please run "Start-Simple.bat" first
    pause
    exit /b 1
)

echo [OK] Virtual environment found
echo.

REM Create backend startup script
echo @echo off > _start_backend_direct.bat
echo title Backend Service >> _start_backend_direct.bat
echo cd backend >> _start_backend_direct.bat
echo echo. >> _start_backend_direct.bat
echo echo ============================================================ >> _start_backend_direct.bat
echo echo   Backend Service Running >> _start_backend_direct.bat
echo echo   API: http://localhost:8000 >> _start_backend_direct.bat
echo echo   Docs: http://localhost:8000/docs >> _start_backend_direct.bat
echo echo ============================================================ >> _start_backend_direct.bat
echo echo. >> _start_backend_direct.bat
echo venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload >> _start_backend_direct.bat
echo pause >> _start_backend_direct.bat

REM Create frontend startup script (if Node.js exists)
node --version >nul 2>&1
if not errorlevel 1 (
    echo @echo off > _start_frontend_direct.bat
    echo title Frontend Service >> _start_frontend_direct.bat
    echo cd frontend >> _start_frontend_direct.bat
    echo echo. >> _start_frontend_direct.bat
    echo echo ============================================================ >> _start_frontend_direct.bat
    echo echo   Frontend Service Running >> _start_frontend_direct.bat
    echo echo   URL: http://localhost:3000 >> _start_frontend_direct.bat
    echo echo ============================================================ >> _start_frontend_direct.bat
    echo echo. >> _start_frontend_direct.bat
    echo npm run dev >> _start_frontend_direct.bat
    echo pause >> _start_frontend_direct.bat
    
    set HAS_NODE=1
) else (
    set HAS_NODE=0
)

echo ============================================================
echo   Starting Services...
echo ============================================================
echo.

REM Start backend
echo [OK] Starting backend service...
start "Backend Service" cmd /k _start_backend_direct.bat

REM Wait 2 seconds
timeout /t 2 /nobreak >nul

REM Start frontend
if "%HAS_NODE%"=="1" (
    echo [OK] Starting frontend service...
    timeout /t 1 /nobreak >nul
    start "Frontend Service" cmd /k _start_frontend_direct.bat
    
    echo.
    echo ============================================================
    echo   Services Started Successfully!
    echo ============================================================
    echo.
    echo   Backend: http://localhost:8000
    echo   Frontend: http://localhost:3000
    echo.
    echo   Browser will open in 10 seconds...
    echo.
    echo ============================================================
    
    timeout /t 10 /nobreak >nul
    start http://localhost:3000
) else (
    echo.
    echo ============================================================
    echo   Backend Service Started!
    echo ============================================================
    echo.
    echo   Backend: http://localhost:8000
    echo   Docs: http://localhost:8000/docs
    echo.
    echo   [WARNING] Frontend not started (Node.js not found)
    echo.
    echo ============================================================
    
    timeout /t 5 /nobreak >nul
    start http://localhost:8000/docs
)

echo.
echo [TIP] You can close this window, services run in separate windows
pause
