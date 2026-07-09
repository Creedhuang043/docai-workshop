"""
Day 2-4: LangChain 鏈式處理模組
展示：數據提取、溫度參數、JSON 解析、並行處理
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel
import json

# ==================== 配置 ====================
def get_llm(temperature: float = 0):
    """
    初始化 LLM 客戶端
    temperature: 0 = 確定性, 1 = 平衡, 1.5+ = 創意性
    """
    return ChatOpenAI(
        base_url="https://ws-02.wade0426.me/v1",
        api_key="EMPTY",
        model="google/gemma-3-27b-it",
        temperature=temperature,
        max_tokens=256
    )

# ==================== Day 2: 數據提取 ====================
def create_extraction_chain():
    """
    Day 2-ch2-4: JSON 提取鏈
    從非結構化文本中提取結構化數據
    """

    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    請從以下文本中提取訂單信息，並以 JSON 格式返回。
    必須包含以下欄位：name, phone, product, quantity, location
    如果缺少信息，使用 null

    文本: {text}

    JSON 格式:
    {{
        "name": "客戶名稱",
        "phone": "電話號碼",
        "product": "商品名稱",
        "quantity": 數量,
        "location": "送貨位置"
    }}
    """)

    parser = JsonOutputParser()

    chain = prompt | llm | parser

    return chain


# ==================== Day 4: 並行鏈處理 ====================
def create_parallel_chain():
    """
    Day 4-HW: 並行鏈式處理 (RunnableParallel)
    同時從多個角度生成內容
    """

    llm = get_llm(temperature=0.8)

    # 分支 A: LinkedIn 專家
    linkedin_prompt = ChatPromptTemplate.from_template("""
    你是 LinkedIn 上的專業職涯顧問。
    請針對主題：{topic}
    寫一段嚴肅、專業的商業評論（50字內）
    """)

    linkedin_chain = (
        linkedin_prompt | llm | StrOutputParser()
    )

    # 分支 B: Instagram 網紅
    ig_prompt = ChatPromptTemplate.from_template("""
    你是 Instagram 上的幽默網紅。
    請針對主題：{topic}
    寫一段活潑有趣的貼文，包含 Emoji 和 Hashtag（50字內）
    """)

    ig_chain = (
        ig_prompt | llm | StrOutputParser()
    )

    # 組合成並行鏈
    parallel_chain = RunnableParallel(
        linkedin=linkedin_chain,
        instagram=ig_chain
    )

    return parallel_chain


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
