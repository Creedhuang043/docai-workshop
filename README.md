# 🤖 次世代 DocAI 系統工作坊 — 完整成果展示

> 6 天 AI 開發課程的完整實作成果，涵蓋 LangChain、LangGraph、RAG 等核心技術

## 🌐 線上 Demo

👉 **[點此體驗互動 Demo](https://Creedhuang043.github.io/docai-workshop/)**

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
| `.env.example` | 環境變數範本（複製為 `.env` 並填入 Gemini API Key） |
| `EXECUTE_WORKFLOW.md` | 完整執行教學文件 |

## 🚀 快速開始

### 方法一：直接開啟靜態 Demo（推薦）

直接用瀏覽器開啟 `index.html`，無需任何安裝！

```
雙擊 index.html
```

### 方法二：本地後端（需要 Gemini API Key）

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 安裝依賴
pip install -r demo_requirements.txt

# 設定 API Key
copy .env.example .env     # Windows（Mac/Linux 用 cp）
# 編輯 .env，填入你的 GEMINI_API_KEY（免費取得：https://aistudio.google.com/apikey）

# 啟動後端
python demo_backend_main.py
```

然後開啟 `demo_docai_system_frontend.html`

> 📌 課程原本使用的 `ws-02.wade0426.me` 臨時推論伺服器已隨課程結束關閉，
> 後端現在改用 **Google Gemini**（透過 OpenAI 相容端點），需要你自行申請一組免費的 Gemini API Key。

## 🏗️ 技術架構

```
前端 (Vanilla HTML/CSS/JS)
       ↓ fetch API
FastAPI 後端 (Python)
       ↓ LangChain / LangGraph
Google Gemini（OpenAI 相容端點）
```

## 🛠️ 技術棧

- **LangChain** — LLM 鏈式應用框架
- **LangGraph** — 圖形化 AI 工作流
- **FastAPI** — 高效能 Python API
- **Google Gemini** — 大語言模型（gemini-2.5-flash，可透過 `GEMINI_MODEL` 環境變數更換）
- **RAG** — 檢索增強生成技術

---

*次世代 DocAI 系統工作坊 2026 — 學員實作成果*

[![GitHub](https://img.shields.io/badge/GitHub-Creedhuang043-181717?logo=github)](https://github.com/Creedhuang043/docai-workshop)
[![Demo](https://img.shields.io/badge/Demo-線上體驗-7c3aed)](https://Creedhuang043.github.io/docai-workshop/)
