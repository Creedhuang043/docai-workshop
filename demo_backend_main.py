"""
DocAI 智能文檔分析系統 - 後端服務器
展示 Day 1-6 課程的核心概念
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import json
from typing import Dict, Any
import asyncio

# 本地模塊導入
from demo_llm_chain import (
    create_extraction_chain,
    create_parallel_chain,
    GEMINI_MODEL
)
from demo_workflow import create_order_workflow
from demo_rag_system import RAGSystem

# ==================== 初始化 ====================
app = FastAPI(title="DocAI System", version="1.0.0")

# CORS 配置（允許前端調用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化各個系統
# 若尚未設定 GEMINI_API_KEY，這裡不讓伺服器直接崩潰，而是延後到實際呼叫該功能時才回報錯誤，
# 這樣 /health 仍可用來確認伺服器本身有正常啟動。
_init_error = None
try:
    extraction_chain = create_extraction_chain()
    parallel_chain = create_parallel_chain()
    rag_system = RAGSystem()
except RuntimeError as e:
    _init_error = str(e)
    extraction_chain = None
    parallel_chain = None
    rag_system = None

order_workflow = create_order_workflow()  # 內部 LLM 呼叫是延遲執行，這裡建立 Graph 本身不需要金鑰

# ==================== 數據模型 ====================
class AnalysisRequest(BaseModel):
    doc_type: str  # extraction, workflow, rag, parallel
    content: str

class AnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

# ==================== API 端點 ====================

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": "DocAI System",
        "version": "1.0.0",
        "llm_ready": _init_error is None,
        "llm_error": _init_error
    }

@app.post("/analyze")
async def analyze_document(request: AnalysisRequest):
    """
    主分析端點 - 支持 4 種分析模式

    Day 2: 數據提取
    Day 3: 工作流處理
    Day 4: 並行分析
    Day 5-6: RAG 檢索
    """
    start_time = time.time()

    if _init_error and request.doc_type in ("extraction", "rag", "parallel"):
        raise HTTPException(status_code=503, detail=_init_error)

    try:
        doc_type = request.doc_type
        content = request.content

        if doc_type == "extraction":
            # ==================== Day 2: 數據提取 ====================
            # 使用 LangChain 的 JSON 提取器
            result = extraction_chain.invoke({"text": content})

            response_data = {
                "extraction": result,
                "metadata": {
                    "doc_type": "extraction",
                    "execution_time": (time.time() - start_time) * 1000,
                    "model": GEMINI_MODEL
                }
            }

        elif doc_type == "workflow":
            # ==================== Day 3: LangGraph 工作流 ====================
            # 使用 LangGraph 的訂單處理工作流
            from langchain_core.messages import HumanMessage

            workflow_result = []
            for event in order_workflow.stream(
                {"messages": [HumanMessage(content=content)]}
            ):
                for key, value in event.items():
                    workflow_result.append({
                        "node": key,
                        "message": str(value.get("messages", [])[-1])
                    })

            response_data = {
                "workflow": {
                    "steps": workflow_result,
                    "summary": f"處理了訂單申請，涉及 {len(workflow_result)} 個流程步驟"
                },
                "metadata": {
                    "doc_type": "workflow",
                    "execution_time": (time.time() - start_time) * 1000,
                    "model": GEMINI_MODEL
                }
            }

        elif doc_type == "rag":
            # ==================== Day 5-6: RAG 檢索 ====================
            # 使用 RAG 系統進行文檔檢索和問答
            retrieved_docs = rag_system.retrieve(content)
            answer = rag_system.generate_answer(content, retrieved_docs)

            response_data = {
                "rag": {
                    "query": content,
                    "retrieved_docs": [d["content"] for d in retrieved_docs],  # 前 3 個最相關文檔
                    "answer": answer,
                    "relevance_scores": [d["score"] for d in retrieved_docs]
                },
                "metadata": {
                    "doc_type": "rag",
                    "execution_time": (time.time() - start_time) * 1000,
                    "model": GEMINI_MODEL,
                    "documents_retrieved": len(retrieved_docs)
                }
            }

        elif doc_type == "parallel":
            # ==================== Day 4: 並行分析 ====================
            # 使用 RunnableParallel 同時生成多個視角
            result = parallel_chain.invoke({"topic": content})

            response_data = {
                "parallel": {
                    "linkedin": result.get("linkedin", "無法生成"),
                    "instagram": result.get("instagram", "無法生成")
                },
                "metadata": {
                    "doc_type": "parallel",
                    "execution_time": (time.time() - start_time) * 1000,
                    "model": GEMINI_MODEL,
                    "parallel_chains": 2
                }
            }

        else:
            raise HTTPException(status_code=400, detail=f"未知的分析類型: {doc_type}")

        return {
            "status": "success",
            **response_data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"分析失敗: {str(e)}"
        )

@app.get("/info")
async def get_system_info():
    """獲取系統信息"""
    return {
        "name": "智能文檔分析系統",
        "version": "1.0.0",
        "description": "次世代 DocAI 系統工作坊完整 DEMO",
        "features": [
            "Day 2: LangChain 數據提取",
            "Day 3: LangGraph 工作流",
            "Day 4: 並行鏈式處理",
            "Day 5-6: RAG 檢索增強"
        ],
        "models_supported": [
            GEMINI_MODEL
        ]
    }

@app.post("/demo")
async def run_demo():
    """運行完整示例"""
    demo_results = {
        "extractions": [
            {
                "input": "我叫張三，電話 0912-345-678，要買 3 台筆電，送台北",
                "output": {
                    "name": "張三",
                    "phone": "0912-345-678",
                    "product": "筆電",
                    "quantity": 3,
                    "location": "台北"
                }
            }
        ],
        "workflows": [
            {
                "name": "訂單處理流程",
                "steps": ["數據提取", "驗證", "處理", "回應"]
            }
        ],
        "parallel": {
            "linkedin": "AI 正在改變職場環境，專業人士需要持續學習新技術...",
            "instagram": "🚀 AI 來了！你準備好迎接未來了嗎？💡 #AI #Tech #Future"
        }
    }
    return demo_results

# ==================== 啟動服務 ====================
if __name__ == "__main__":
    import uvicorn
    import sys
    out = sys.stdout
    out.reconfigure(encoding="utf-8") if hasattr(out, "reconfigure") else None
    print("=" * 50)
    print("DocAI - Backend Server Starting")
    print("=" * 50)
    print("API Server: http://localhost:8000")
    print("API Docs  : http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
