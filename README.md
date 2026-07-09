# 🤖 次世代 DocAI 系統工作坊 - 完整 DEMO

> 綜合展示 6 天課程核心技術的智能文檔分析系統

## ✨ 功能展示

| 功能 | 技術 | 課程 |
|------|------|------|
| 📊 數據提取 | LangChain JSON 解析 | Day 2 |
| 🔄 工作流處理 | LangGraph 狀態機 | Day 3 |
| ⚡ 並行分析 | RunnableParallel | Day 4 |
| 🔍 RAG 檢索 | 檢索增強生成 | Day 5-6 |

## 🚀 快速啟動（本地）

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 安裝依賴
pip install -r demo_requirements.txt

# 啟動後端
python demo_backend_main.py
```

然後直接打開 `demo_docai_system_frontend.html`，或訪問 `http://localhost:8000/docs`

## 🌐 線上 DEMO

部署於 Render.com，訪問即可使用：  
👉 **[點此訪問 DEMO](#)** ← 部署後更新此連結

## 🛠️ 技術架構

```
前端 HTML (Vanilla JS)
       ↓ fetch API
FastAPI 後端 (Python)
       ↓ LangChain / LangGraph
Gemma-3-27B 模型 (課程伺服器)
```

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `demo_backend_main.py` | FastAPI 主程式 |
| `demo_llm_chain.py` | Day 2-4 LangChain 模組 |
| `demo_workflow.py` | Day 3 LangGraph 工作流 |
| `demo_rag_system.py` | Day 5-6 RAG 系統 |
| `demo_docai_system_frontend.html` | 前端介面 |
| `demo_requirements.txt` | 依賴套件列表 |

---
*次世代 DocAI 系統工作坊*
