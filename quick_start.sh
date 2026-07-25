#!/bin/bash

# 次世代 DocAI 系統工作坊 - 快速啟動腳本 (Linux/Mac)

echo ""
echo "=================================================="
echo "  DocAI 智能文檔分析系統 - 快速啟動"
echo "=================================================="
echo ""

# 檢查 Python 是否安裝
if ! command -v python3 &> /dev/null
then
    echo "❌ 錯誤: 未找到 Python，請先安裝 Python 3.10+"
    echo "下載地址: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ 已檢測到 Python"
python3 --version

# 檢查是否在正確的目錄
if [ ! -f "demo_requirements.txt" ]; then
    echo "❌ 錯誤: 請在項目根目錄運行此腳本"
    exit 1
fi

# 創建虛擬環境（如果不存在）
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 正在創建虛擬環境..."
    python3 -m venv venv
    echo "✅ 虛擬環境創建完成"
fi

# 激活虛擬環境
echo ""
echo "🔧 正在激活虛擬環境..."
source venv/bin/activate

# 安裝依賴
echo ""
echo "📚 正在安裝依賴..."
pip install -r demo_requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "❌ 依賴安裝失敗"
    exit 1
fi
echo "✅ 依賴安裝完成"

# 檢查 .env 是否存在（用來放 Gemini API Key）
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  尚未設定 .env，正在從 .env.example 複製..."
    cp .env.example .env
    echo "⚠️  請打開 .env 填入你的 GEMINI_API_KEY 後再重新執行本腳本"
    echo "   免費申請金鑰: https://aistudio.google.com/apikey"
    exit 1
fi

# 啟動後端
echo ""
echo "=================================================="
echo "  🚀 正在啟動後端服務器..."
echo "=================================================="
echo ""
echo "服務器地址: http://localhost:8000"
echo "API 文檔:  http://localhost:8000/docs"
echo ""

python3 demo_backend_main.py
