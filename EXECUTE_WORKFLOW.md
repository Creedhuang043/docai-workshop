# 🚀 次世代 DocAI 系統工作坊 DEMO - 完整執行流程

> **本文檔提供逐步執行指令，照著步驟做就能順利運行整個系統！**

---

## 📋 目錄

- [第一階段：環境準備](#第一階段環境準備)
- [第二階段：項目設置](#第二階段項目設置)
- [第三階段：啟動服務](#第三階段啟動服務)
- [第四階段：測試驗證](#第四階段測試驗證)
- [第五階段：體驗 DEMO](#第五階段體驗-demo)

---

## 第一階段：環境準備

### 步驟 1.1：檢查 Python 版本

**執行命令：**
```bash
python --version
```

**預期輸出：**
```
Python 3.10.x 或更高版本
```

**如果失敗：**
- ❌ 找不到 Python → 下載安裝 [Python 3.10+](https://www.python.org/)
- ❌ 版本過舊 → 升級 Python

✅ **檢查點：** 確認 Python 版本 ≥ 3.10

---

### 步驟 1.2：檢查 pip

**執行命令：**
```bash
pip --version
```

**預期輸出：**
```
pip 23.x.x (或更高版本)
```

✅ **檢查點：** pip 能正常使用

---

### 步驟 1.3：檢查網絡連接

**執行命令：**
```bash
ping 8.8.8.8
```

**預期輸出：**
```
reply from 8.8.8.8: ...
```

**如果失敗：**
- 檢查網絡連接
- 確保防火牆未阻止

✅ **檢查點：** 網絡連接正常

---

## 第二階段：項目設置

### 步驟 2.1：進入工作目錄

**執行命令：**
```bash
cd D:\10_Creed\myself\次世代DocAI系統工作坊
```

**預期結果：**
```
當前目錄應為: D:\10_Creed\myself\次世代DocAI系統工作坊
```

✅ **檢查點：** 已進入正確目錄

---

### 步驟 2.2：驗證所有文件已複製

**執行命令：**
```bash
dir /s          # Windows
ls -la          # Mac/Linux
```

**預期輸出：** 應該看到以下檔案

**後端文件（4 個）：**
```
✓ demo_backend_main.py
✓ demo_llm_chain.py
✓ demo_workflow.py
✓ demo_rag_system.py
```

**前端文件（1 個）：**
```
✓ demo_docai_system_frontend.html
```

**配置文件（3 個）：**
```
✓ demo_requirements.txt
✓ quick_start.bat (Windows)
✓ quick_start.sh (Mac/Linux)
```

✅ **檢查點：** 所有文件已在正確位置

---

### 步驟 2.3：創建虛擬環境

**執行命令：**
```bash
python -m venv venv
```

**預期結果：**
```
創建 venv 文件夾（包含隔離的 Python 環境）
```

**驗證方法：**
```bash
dir venv          # Windows
ls venv           # Mac/Linux
```

✅ **檢查點：** 虛擬環境已創建

---

### 步驟 2.4：激活虛擬環境

#### Windows：
```bash
venv\Scripts\activate
```

**預期輸出：**
```
(venv) D:\10_Creed\myself\次世代DocAI系統工作坊>
```
*注意命令行前面出現 (venv) 表示激活成功*

#### Mac/Linux：
```bash
source venv/bin/activate
```

**預期輸出：**
```
(venv) $ 
```

✅ **檢查點：** 命令行前出現 (venv) 標誌

---

### 步驟 2.5：升級 pip

**執行命令：**
```bash
pip install --upgrade pip
```

**預期輸出：**
```
Successfully installed pip-...
```

✅ **檢查點：** pip 已升級

---

### 步驟 2.6：安裝依賴

**執行命令：**
```bash
pip install -r demo_requirements.txt
```

**預期輸出：**
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 ...
```

**安裝時間：** 約 2-5 分鐘

**如果安裝失敗：**
```bash
# 清除緩存並重試
pip install --no-cache-dir -r demo_requirements.txt
```

**驗證安裝：**
```bash
pip list
```

**預期看到：**
```
fastapi            0.104.1
uvicorn            0.24.0
langchain          0.0.340
langgraph          0.0.31
langchain-openai   0.0.8
...
```

✅ **檢查點：** 所有依賴已成功安裝

---

## 第三階段：啟動服務

### 步驟 3.1：啟動後端服務器

**執行命令：**
```bash
python demo_backend_main.py
```

**預期輸出：**
```
==================================================
🚀 DocAI 智能文檔分析系統 - 後端服務器
==================================================
📍 API Server 運行在: http://localhost:8000
📚 API 文檔: http://localhost:8000/docs
==================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**狀態檢查：**
- ✅ 看到上述信息 → 服務器啟動成功
- ❌ 看到 "Address already in use" → 端口被占用（見故障排除）
- ❌ 看到其他錯誤 → 檢查依賴是否正確安裝

**重要：** 
- ⚠️ **不要關閉此終端窗口**
- 服務器需要持續運行

✅ **檢查點：** 後端服務器成功運行

---

### 步驟 3.2：打開新的終端窗口

**操作步驟：**
1. 打開新的終端/命令提示符窗口
2. 進入同一目錄
3. 激活虛擬環境

**Windows：**
```bash
cd D:\10_Creed\myself\次世代DocAI系統工作坊
venv\Scripts\activate
```

**Mac/Linux：**
```bash
cd /path/to/次世代DocAI系統工作坊
source venv/bin/activate
```

✅ **檢查點：** 新終端已準備好

---

## 第四階段：測試驗證

### 步驟 4.1：測試後端健康狀態

**在新終端窗口中執行：**
```bash
curl http://localhost:8000/health
```

**預期輸出：**
```json
{
  "status": "healthy",
  "service": "DocAI System",
  "version": "1.0.0"
}
```

**如果失敗：**
- ❌ "Connection refused" → 後端沒有運行（檢查步驟 3.1）
- ❌ "timeout" → 網絡問題或防火牆

✅ **檢查點：** 後端健康檢查通過

---

### 步驟 4.2：查看 API 文檔

**操作步驟：**
1. 打開瀏覽器
2. 訪問：`http://localhost:8000/docs`

**預期結果：**
- 看到 Swagger UI 界面
- 列出所有 API 端點
- 可以進行交互式測試

✅ **檢查點：** API 文檔頁面正常顯示

---

### 步驟 4.3：測試數據提取 API

**執行命令：**
```bash
curl -X POST http://localhost:8000/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"doc_type\":\"extraction\",\"content\":\"我叫王小明，電話 0912-345-678，要買 5 台筆電\"}"
```

**Mac/Linux 版本：**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"doc_type":"extraction","content":"我叫王小明，電話 0912-345-678，要買 5 台筆電"}'
```

**預期輸出：**
```json
{
  "status": "success",
  "extraction": {
    "name": "王小明",
    "phone": "0912-345-678",
    "product": "筆電",
    "quantity": 5,
    "address": null
  },
  "metadata": {
    "doc_type": "extraction",
    "execution_time": 2345.67,
    "model": "gemma-3-27b-it"
  }
}
```

✅ **檢查點：** 數據提取 API 正常工作

---

## 第五階段：體驗 DEMO

### 步驟 5.1：打開前端界面

**方法 1：直接打開文件（推薦）**
```
雙擊 demo_docai_system_frontend.html
```

**方法 2：使用 HTTP 服務器（備選）**
```bash
# 在 次世代DocAI系統工作坊 目錄中新開終端
python -m http.server 8080

# 瀏覽器訪問
http://localhost:8080/demo_docai_system_frontend.html
```

**預期結果：**
- 看到紫色漸變背景的現代界面
- 左側有分析類型選擇和輸入框
- 右側有功能說明和課程內容

✅ **檢查點：** 前端界面正常加載

---

### 步驟 5.2：測試數據提取（Day 2）

**操作步驟：**
1. 分析類型：選擇 **「✨ 數據提取 (Day 2)」**
2. 輸入內容：
   ```
   我叫陳大明，電話是 0912-345-678，
   我想要訂購 3 台筆記型電腦，
   下週五送到台中市北區。
   ```
3. 點擊 **「🚀 開始分析」**

**預期結果：**
```
✅ 分析完成！

提取結果：
{
  "name": "陳大明",
  "phone": "0912-345-678",
  "product": "筆記型電腦",
  "quantity": 3,
  "address": "台中市北區"
}

執行時間: 2345.67 ms
模型: gemma-3-27b-it
```

✅ **檢查點：** Day 2 數據提取功能正常

---

### 步驟 5.3：測試工作流處理（Day 3）

**操作步驟：**
1. 分析類型：選擇 **「🔄 工作流處理 (Day 3)」**
2. 輸入內容：
   ```
   我要訂購 10 台筆電，
   客戶名稱是李美琪，
   電話 0988-888-888，
   地址台南市
   ```
3. 點擊 **「🚀 開始分析」**

**預期結果：**
```
✅ 分析完成！

工作流執行結果：
步驟 1: [agent] 分析用戶輸入
步驟 2: [tools] 執行提取工具
步驟 3: [agent] 評估結果並回應

流程摘要: 處理了訂單申請，涉及 3 個流程步驟

執行時間: 3456.78 ms
```

✅ **檢查點：** Day 3 工作流功能正常

---

### 步驟 5.4：測試 RAG 檢索（Day 5-6）

**操作步驟：**
1. 分析類型：選擇 **「🔍 RAG 檢索 (Day 5-6)」**
2. 輸入內容：
   ```
   什麼是 LangGraph？
   ```
3. 點擊 **「🚀 開始分析」**

**預期結果：**
```
✅ 分析完成！

檢索到的相關文檔:
1. LangGraph 是 LangChain 的圖形處理擴展。它使用狀態機...
2. RAG（檢索增強生成）是一種結合信息檢索和生成的技術...
3. 次世代 DocAI 系統工作坊是一個全面的 AI 開發課程...

AI 答案:
LangGraph 是 LangChain 框架的圖形工作流擴展，
它使用狀態機模型定義複雜的 AI 流程...

相關度: 0.95, 0.87, 0.78
```

✅ **檢查點：** Day 5-6 RAG 檢索功能正常

---

### 步驟 5.5：測試並行分析（Day 4）

**操作步驟：**
1. 分析類型：選擇 **「⚡ 並行分析 (Day 4)」**
2. 輸入內容：
   ```
   人工智能的未來發展
   ```
3. 點擊 **「🚀 開始分析」**

**預期結果：**
```
✅ 分析完成！

LinkedIn 專家觀點:
人工智能正在革新各行業，
企業需要評估 AI 帶來的風險和機遇...

Instagram 網紅觀點:
🚀 AI 來了！你準備好迎接未來了嗎？
💡 從今天開始學習 AI 技術吧！
#AI #FutureOfTech #Innovation

執行時間: 4567.89 ms
並行鏈數: 2
```

✅ **檢查點：** Day 4 並行分析功能正常

---

## 🔧 故障排除

### 問題 1：端口 8000 已被占用
```bash
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### 問題 2：依賴安裝失敗
```bash
pip install --no-cache-dir -r demo_requirements.txt
```

### 問題 3：CORS 跨域錯誤
- 確保後端運行在 `http://localhost:8000`
- 確保前端通過 HTTP 服務器訪問

---

## ✅ 成功完成！

當你看到以下結果時，表示完全成功：

1. ✅ 後端終端顯示：`Application startup complete`
2. ✅ 前端頁面加載成功
3. ✅ 4 種分析模式都能返回結果
4. ✅ 沒有紅色錯誤信息

---

**準備好了嗎？開始執行第一階段吧！** 🚀
