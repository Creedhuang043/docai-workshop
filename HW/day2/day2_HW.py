"""
2026_2504 次世代 DocAI 系統工作坊  Day2 課後實戰
學號：2411332043    姓名：黃晧德

題目：打造平行化 AI 社群小編
  利用 LangChain LCEL 的平行處理能力，設計一個能「多工處理」的 AI Agent。
  使用者只需輸入一個主題（Topic），系統需同時撰寫出兩種不同風格的貼文。

作業要求對照：
  1. 使用 RunnableParallel 平行處理           → combo_chain
  2. 兩種不同風格的貼文                        → LinkedIn 專業版 / IG 網紅版
  3. 流式(Streaming)與批次(batch)各執行一次    → run_streaming() / run_batch()
  4. 批次處理需另記錄處理時間                  → run_batch() 內計時
  5. temperature 設為 0                        → 模型設定
  6. 流式輸出需看到不同主題交錯                → 逐 chunk 標示來源分支

備註：課程原本使用的 vLLM 推論伺服器（ws-02.wade0426.me）已隨課程結束關閉，
      本作業改接 Google Gemini 的 OpenAI 相容端點，金鑰請設定於 .env 的 GEMINI_API_KEY。
"""

import os
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


# --- 1. 設定模型 ---
def build_model() -> ChatOpenAI:
    """建立 LLM（temperature=0，方便觀察平行處理的輸出差異）。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY，請於 .env 設定。")
    return ChatOpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=api_key,
        model=GEMINI_MODEL,
        temperature=0,
        max_tokens=2048,
        timeout=120,
        max_retries=5,
    )


model = build_model()

# --- 2. 定義人設分工 ---
# Branch A: 嚴肅 LinkedIn 專家
linkedin_chain = (
    ChatPromptTemplate.from_template(
        "你是 LinkedIn 上的專業職涯顧問。請針對主題：{topic}，"
        "寫一段嚴肅、專業且具備商業洞察力的短評(50字內)。直接輸出貼文內容，不要加標題或說明。"
    ) | model | StrOutputParser()
)

# Branch B: 幽默 IG 網紅
ig_chain = (
    ChatPromptTemplate.from_template(
        "你是 Instagram 上的幽默網紅。請針對主題：{topic}，"
        "寫一段活潑、好笑的貼文，一定要包含表情符號(Emoji)和熱門 Hashtag (50字內)。"
        "直接輸出貼文內容，不要加標題或說明。"
    ) | model | StrOutputParser()
)

# --- 3. 優雅組合 (Parallel) ---
combo_chain = RunnableParallel(
    linkedin=linkedin_chain,
    instagram=ig_chain,
)

LABELS = {"linkedin": "💼 LinkedIn", "instagram": "📱 Instagram"}


# --- 模式 1: Streaming (流式輸出) ---
async def _stream_interleaved(topic: str, buffers: dict, order: list) -> None:
    """以 astream 非同步取回兩個分支的 chunk；asyncio 排程能真正讓兩者交錯抵達。"""
    async for chunk in combo_chain.astream({"topic": topic}):
        for branch, text in chunk.items():
            if not text:
                continue
            buffers[branch] += text
            order.append(branch)
            # 逐 chunk 印出來源，交錯情形一目了然
            print(f"  [{LABELS[branch]}] {text.replace(chr(10), ' ')}")


def run_streaming(topic: str) -> None:
    """
    流式輸出：兩條 chain 同時進行，因此 chunk 會交錯抵達。
    這裡逐一標示每個 chunk 來自哪一個分支，讓「交錯」現象清楚可見。
    """
    print("=" * 60)
    print("【模式 1：Streaming 流式輸出】")
    print("觀察重點：兩個分支的文字片段會交錯出現")
    print("=" * 60)

    order, buffers = [], {"linkedin": "", "instagram": ""}
    start = time.time()

    asyncio.run(_stream_interleaved(topic, buffers, order))

    elapsed = time.time() - start
    print("-" * 60)
    print(f"抵達順序（前 30 個 chunk）：")
    print("  " + " → ".join(LABELS[b].split()[1] for b in order[:30]))
    switches = sum(1 for i in range(1, len(order)) if order[i] != order[i - 1])
    print(f"  共 {len(order)} 個 chunk，分支切換 {switches} 次 → 確實為交錯輸出")
    print(f"流式耗時: {elapsed:.2f} 秒")

    print("-" * 60)
    for branch, text in buffers.items():
        print(f"【{LABELS[branch]} 完整內容】\n{text.strip()}\n")


# --- 模式 2: Batch (批次/完整輸出) ---
def run_batch(topic: str) -> dict:
    """批次處理：一次執行完畢再輸出，並記錄處理時間。"""
    print("=" * 60)
    print("【模式 2：Batch 批次處理】")
    print("觀察重點：等待一段時間後，一次顯示完整結果")
    print("=" * 60)

    start_time = time.time()
    # 雖然只有一個主題，但 batch 介面要求輸入 List
    results = combo_chain.batch([{"topic": topic}])
    elapsed = time.time() - start_time

    final_result = results[0]
    print(f"批次耗時: {elapsed:.2f} 秒")
    print("-" * 60)
    print(f"【LinkedIn 專家說】：\n{final_result['linkedin'].strip()}")
    print("-" * 60)
    print(f"【IG 網紅說】：\n{final_result['instagram'].strip()}")
    print("-" * 60)
    return {"result": final_result, "elapsed": elapsed}


def main():
    print(f"模型：{GEMINI_MODEL}（temperature=0）\n")
    target_topic = input("輸入主題：").strip() or "Work Life Balance"
    print(f"\n目標主題：{target_topic}\n")

    run_streaming(target_topic)
    print()
    run_batch(target_topic)


if __name__ == "__main__":
    main()
