@echo off
echo ==========================================
echo Project Environment Repair Script
echo ==========================================
cd /d "e:\CIS_Audit_Project"

if not exist ".venv" (
    echo Error: .venv folder not found! Please make sure you are in the project root.
    pause
    exit /b
)

echo Installing dependencies into .venv...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo Installation failed! Trying fallback with global pip...
    pip install -r requirements.txt
)

echo.
echo ==========================================
echo Verification...
".venv\Scripts\python.exe" -c "import streamlit, pandas, docx, langchain; print('Environment OK!')"
echo ==========================================
pause
