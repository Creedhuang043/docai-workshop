"""
Day 5-6: RAG (檢索增強生成) 系統
展示：文檔檢索、向量化、問答系統
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 使用 Google Gemini 的 OpenAI 相容端點，金鑰請在 .env 中設定 GEMINI_API_KEY
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ==================== 樣本文檔庫 ====================
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "AI 基礎知識",
        "content": """
        人工智能（AI）是計算機科學的一個分支，旨在創建能夠執行通常需要人類智能的任務的機器。
        AI 的主要應用領域包括：
        1. 機器學習：訓練計算機從數據中學習模式
        2. 自然語言處理：理解和生成人類語言
        3. 計算機視覺：處理和分析圖像
        4. 機器人技術：設計自動化系統
        """
    },
    {
        "id": "doc_002",
        "title": "LangChain 框架",
        "content": """
        LangChain 是一個用於開發 LLM 應用程序的框架。
        核心概念：
        - Chains: 組合多個步驟的序列
        - Prompts: 用戶指令模板
        - Memory: 保存對話歷史
        - Tools: 外部函數集成
        - Agents: 自主決策系統

        LangChain 簡化了複雜的 AI 應用開發。
        """
    },
    {
        "id": "doc_003",
        "title": "LangGraph 工作流",
        "content": """
        LangGraph 是 LangChain 的圖形處理擴展。
        它使用狀態機模型定義複雜的工作流：
        - State: 系統的當前狀態
        - Node: 處理節點
        - Edge: 節點之間的連接
        - Conditional Edge: 條件分支

        LangGraph 適合構建多步驟的 AI 流程。
        """
    },
    {
        "id": "doc_004",
        "title": "RAG 技術",
        "content": """
        RAG（檢索增強生成）是一種結合信息檢索和生成的技術。
        工作流程：
        1. 用戶提出問題
        2. 系統從文檔庫中檢索相關信息
        3. 將檢索的信息作為上下文
        4. LLM 基於上下文生成答案

        RAG 能夠提供更準確和有根據的回答。
        """
    },
    {
        "id": "doc_005",
        "title": "次世代 DocAI 工作坊",
        "content": """
        次世代 DocAI 系統工作坊是一個全面的 AI 開發課程。
        課程內容：
        - Day 1: Git 和前端基礎
        - Day 2: LangChain 基礎
        - Day 3: Tool 和 LangGraph
        - Day 4: 高級主題
        - Day 5-6: RAG 系統

        通過 6 天的學習，掌握現代 AI 應用開發。
        """
    }
]

# ==================== RAG 系統類 ====================
class RAGSystem:
    """
    RAG 系統實現
    包括文檔檢索和問答生成
    """

    def __init__(self):
        """初始化 RAG 系統"""
        self.documents = SAMPLE_DOCUMENTS
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 GEMINI_API_KEY 環境變數。請複製 .env.example 為 .env，"
                "並填入你的 Gemini API Key（https://aistudio.google.com/apikey）。"
            )
        self.llm = ChatOpenAI(
            base_url=GEMINI_BASE_URL,
            api_key=api_key,
            model=GEMINI_MODEL,
            temperature=0.7,
            max_tokens=512
        )

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Day 5: 文檔檢索
        使用簡單的文本相似度進行檢索
        """

        scores = []

        for doc in self.documents:
            doc_text = (doc["title"] + " " + doc["content"]).lower()
            query_lower = query.lower()

            query_words = query_lower.split()
            overlap = sum(1 for word in query_words if word in doc_text)
            score = overlap / len(query_words) if query_words else 0

            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc for doc, score in scores[:top_k]]

        return [doc["content"][:200] for doc in retrieved_docs]

    def generate_answer(self, query: str, context_docs: list) -> str:
        """
        Day 6: 基於上下文生成答案
        """

        context = "\n\n".join(context_docs)

        prompt = ChatPromptTemplate.from_template("""
        基於以下文檔內容，請回答用戶的問題。
        如果文檔中沒有相關信息，請誠實地說明。

        相關文檔：
        {context}

        用戶問題：{query}

        請用繁體中文簡潔回答（2-3 句）：
        """)

        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": context,
                "query": query
            })
            return answer
        except Exception as e:
            return f"生成答案時出錯：{str(e)}"

    def qa_pipeline(self, query: str) -> dict:
        """
        Day 5-6: 完整的 RAG 流程
        """

        retrieved_docs = self.retrieve(query)
        answer = self.generate_answer(query, retrieved_docs)

        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "answer": answer
        }


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
