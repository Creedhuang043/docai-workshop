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

## 📊 實測評測結果（Day5–Day7 作業）

> 完整程式碼、原始 CSV 與報告：**[`HW/`](HW/)**　│　線上圖表：[Demo 網站的「實測數據」區塊](https://Creedhuang043.github.io/docai-workshop/#results)

以下分數皆由實際呼叫 Gemini API 產出，非模擬資料。評估框架為 DeepEval。

**Day5 — 三種切塊方法比較**（5 份文件 × 20 題 = 60 筆）

| 切塊方法 | 參數 | 塊數 | 平均分 |
|---|---|---|---|
| 🥇 滑動視窗 | window 500 / step 250（重疊 50%） | 71 | **0.905** |
| 固定大小 | chunk 500 / 不重疊 | 40 | 0.850 |
| 語意切塊 | 相鄰句相似度 < P25 處切開 | 87 | 0.850 |

**Day6 — 客服 RAG 優化前後對照**（同 10 題；基線 = 純向量檢索，優化版 = Query Rewrite + Hybrid Search + Rerank）

| 指標 | 基線 | 優化版 | 提升 |
|---|---|---|---|
| Contextual Relevancy | 0.5023 | 0.5552 | **+0.0529** |
| Contextual Precision | 0.9249 | 0.9583 | **+0.0334** |
| Answer Relevancy | 0.9633 | 0.9800 | +0.0167 |
| Contextual Recall | 0.8334 | 0.8500 | +0.0166 |
| Faithfulness | 1.0000 | 0.9833 | −0.0167 |

全部 30 題最終成績：忠實度 0.978、答案相關性 0.966、上下文精確度 0.986。

**Day7 — IDP 文件問答 + 注入偵測**（11 題）

| 指標 | 分數 |
|---|---|
| Answer Relevancy | 1.0000 |
| Faithfulness | 0.9818 |
| Contextual Recall | 0.9091 |
| Contextual Precision | 0.8030 |

🚨 以規則式比對與 LLM 判讀雙軌掃描 5 份文件，一致偵測出 **`2.pdf`、`5.docx`** 含惡意提示詞注入，
建索引前已將注入句子消毒移除。

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

### ⚠️ 已知限制：Day 3 工作流（多輪工具呼叫）

`demo_workflow.py` 的 LangGraph 訂單流程實測會在第二輪工具呼叫時出現：

```
400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts.
```

原因：Gemini 3.x 等「思考型」模型要求多輪工具呼叫時原樣帶回上一輪的 `thought_signature`
（內部推理簽章），但 LangChain 的 OpenAI 相容轉接層目前還不支援保留這個 Gemini 專屬欄位。
extraction / parallel / rag 三個端點（單輪對話，無工具呼叫）已實測正常。
若你的帳號還能存取 `gemini-2.0-flash` 這類舊一代模型（免費配額常見為 0，視帳號而定），
可在 `.env` 設定 `GEMINI_WORKFLOW_MODEL=gemini-2.0-flash` 繞過此問題。

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
- **Google Gemini** — 大語言模型（gemini-flash-latest，可透過 `GEMINI_MODEL` 環境變數更換）
- **RAG** — 檢索增強生成技術

---

*次世代 DocAI 系統工作坊 2026 — 學員實作成果*

[![GitHub](https://img.shields.io/badge/GitHub-Creedhuang043-181717?logo=github)](https://github.com/Creedhuang043/docai-workshop)
[![Demo](https://img.shields.io/badge/Demo-線上體驗-7c3aed)](https://Creedhuang043.github.io/docai-workshop/)
