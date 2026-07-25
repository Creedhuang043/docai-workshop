@echo off
REM 次世代 DocAI 系統工作坊 - 快速啟動腳本 (Windows)

echo.
echo ==================================================
echo  DocAI 智能文檔分析系統 - 快速啟動
echo ==================================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 未找到 Python，請先安裝 Python 3.10+
    echo 下載地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ 已檢測到 Python

REM 檢查是否在正確的目錄
if not exist "demo_requirements.txt" (
    echo ❌ 錯誤: 請在項目根目錄運行此腳本
    pause
    exit /b 1
)

REM 創建虛擬環境（如果不存在）
if not exist "venv" (
    echo.
    echo 📦 正在創建虛擬環境...
    python -m venv venv
    echo ✅ 虛擬環境創建完成
)

REM 激活虛擬環境
echo.
echo 🔧 正在激活虛擬環境...
call venv\Scripts\activate.bat

REM 安裝依賴
echo.
echo 📚 正在安裝依賴...
pip install -r demo_requirements.txt --quiet
if errorlevel 1 (
    echo ❌ 依賴安裝失敗
    pause
    exit /b 1
)
echo ✅ 依賴安裝完成

REM 檢查 .env 是否存在（用來放 Gemini API Key）
if not exist ".env" (
    echo.
    echo ⚠️  尚未設定 .env，正在從 .env.example 複製...
    copy .env.example .env >nul
    echo ⚠️  請打開 .env 填入你的 GEMINI_API_KEY 後再重新執行本腳本
    echo    免費申請金鑰: https://aistudio.google.com/apikey
    pause
    exit /b 1
)

REM 啟動後端
echo.
echo ==================================================
echo  🚀 正在啟動後端服務器...
echo ==================================================
echo.
echo 服務器地址: http://localhost:8000
echo API 文檔:  http://localhost:8000/docs
echo.

python demo_backend_main.py

pause
