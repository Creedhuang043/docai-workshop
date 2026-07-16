# 🤖 次世代 DocAI 系統工作坊 — 完整成果展示

> 6 天 AI 開發課程的完整實作成果，涵蓋 LangChain、LangGraph、RAG 等核心技術

## 🌐 線上 Demo

👉 **[點此體驗互動 Demo](https://your-username.github.io/your-repo-name/)** ← 推上 GitHub Pages 後更新此連結

## ✨ 課程成果

| 功能 | 技術 | 課程 |
|------|------|------|
| 📊 數據提取 | LangChain JSON Parser | Day 2 |
| 🔄 工作流處理 | LangGraph 狀態機 | Day 3 |
| ⚡ 並行分析 | RunnableParallel | Day 4 |
| 🔍 RAG 檢索 | 檢索增強生成 | Day 5-6 |

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `index.html` | 🌐 **靜態展示網站**（主要 Demo，無需後端） |
| `demo_backend_main.py` | FastAPI 後端主程式 |
| `demo_llm_chain.py` | Day 2-4 LangChain 模組 |
| `demo_workflow.py` | Day 3 LangGraph 工作流 |
| `demo_rag_system.py` | Day 5-6 RAG 系統 |
| `demo_docai_system_frontend.html` | 原始前端介面（需後端） |
| `demo_requirements.txt` | 依賴套件列表 |
| `EXECUTE_WORKFLOW.md` | 完整執行教學文件 |

## 🚀 快速開始

### 方法一：直接開啟靜態 Demo（推薦）

直接用瀏覽器開啟 `index.html`，無需任何安裝！

```
雙擊 index.html
```

### 方法二：本地後端（需要 LLM 伺服器）

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

然後開啟 `demo_docai_system_frontend.html`

## 🏗️ 技術架構

```
前端 (Vanilla HTML/CSS/JS)
       ↓ fetch API
FastAPI 後端 (Python)
       ↓ LangChain / LangGraph
Gemma-3-27B 大語言模型
```

## 🛠️ 技術棧

- **LangChain** — LLM 鏈式應用框架
- **LangGraph** — 圖形化 AI 工作流
- **FastAPI** — 高效能 Python API
- **Google Gemma-3-27B** — 開源大語言模型
- **RAG** — 檢索增強生成技術

---

*次世代 DocAI 系統工作坊 2026 — 學員實作成果*
