@echo off 
title Backend Service 
cd backend 
echo. 
echo ============================================================ 
echo   Backend Service Running 
echo   API: http://localhost:8000 
echo   Docs: http://localhost:8000/docs 
echo ============================================================ 
echo. 
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 
pause 
