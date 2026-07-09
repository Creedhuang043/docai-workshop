"""
Day 3: LangGraph 工作流模組
展示：State、Node、Edge、Conditional Edge、Loop
"""

from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
import json

# ==================== 配置 ====================
def get_llm():
    """初始化 LLM"""
    return ChatOpenAI(
        base_url="https://ws-02.wade0426.me/v1",
        api_key="EMPTY",
        model="google/gemma-3-27b-it",
        temperature=0
    )

# ==================== Day 3-ch4-1: 定義工具 ====================
@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    訂單數據提取工具
    從非結構化文本中提取訂單相關資訊
    """
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }


@tool
def validate_order(order_data: dict):
    """
    訂單驗證工具
    檢查訂單數據的完整性和有效性
    """
    required_fields = ["name", "phone", "product", "quantity", "address"]
    valid = all(field in order_data and order_data[field] for field in required_fields)

    return {
        "is_valid": valid,
        "message": "訂單驗證通過" if valid else "缺少必要欄位"
    }


@tool
def process_order(order_data: dict):
    """
    訂單處理工具
    處理有效的訂單
    """
    return {
        "status": "processed",
        "order_id": f"ORD-{order_data['name']}-001",
        "message": f"訂單已提交，{order_data['name']} 的訂單將於 3-5 個工作日內送達"
    }

# ==================== Day 3-ch5-1: 定義 State ====================
class AgentState(TypedDict):
    """
    Agent 的狀態容器
    add_messages 確保訊息是「疊加」而非覆蓋
    """
    messages: Annotated[list[BaseMessage], add_messages]

# ==================== Day 3-ch5-1: 定義 Node ====================
def call_model(state: AgentState):
    """
    Node A: 思考節點
    負責呼叫 LLM，決定是否需要使用工具
    """
    llm = get_llm()

    # 綁定工具到 LLM
    tools = [extract_order_data, validate_order, process_order]
    llm_with_tools = llm.bind_tools(tools)

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# ==================== Day 3-ch5-2: 定義工具節點 ====================
tool_node = ToolNode([extract_order_data, validate_order, process_order])

# ==================== Day 3-ch5-1: 定義路由邏輯 ====================
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Edge: 條件邊
    判斷邏輯：LLM 有呼叫工具 -> 走工具節點；沒有 -> 結束
    """
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    else:
        return "end"

# ==================== Day 3-ch5-1: 組裝 Graph ====================
def create_order_workflow():
    """
    Day 3-ch5-1: 訂單處理工作流
    展示完整的 State → Node → Edge → Conditional Edge → Loop
    """

    workflow = StateGraph(AgentState)

    # (1) 添加節點
    workflow.add_node("agent", call_model)      # Node A: 思考節點
    workflow.add_node("tools", tool_node)       # Node B: 工具執行節點

    # (2) 設定入口點
    workflow.set_entry_point("agent")

    # (3) 設定條件邊（Conditional Edge）
    workflow.add_conditional_edges(
        "agent",                    # 從 agent 節點出發
        should_continue,            # 經過 should_continue 判斷
        {
            "tools": "tools",       # 如果回傳 "tools"，走向 tools 節點
            "end": END             # 如果回傳 "end"，結束流程
        }
    )

    # (4) 設定普通邊（Normal Edge）
    # Loop: 工具執行完後，將結果丟回給 Agent
    workflow.add_edge("tools", "agent")

    # (5) 編譯成可執行的應用
    app = workflow.compile()

    return app


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
