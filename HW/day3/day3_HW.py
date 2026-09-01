"""
2026_2504 次世代 DocAI 系統工作坊  Day3 課後實戰
學號：2411332043    姓名：黃晧德

題目：智慧會議記錄助手
  利用 ASR 工具，將一段語音整理成兩種不同的內容：
    1. 詳細的逐字稿：需要按時間軸與對應台詞逐一列出
    2. 重點摘要：須整理出重點摘要
  規則：必須要使用 LangGraph 內的 node / edge 功能

圖結構（Fan-out / Fan-in）：
        ┌──────────────┐
        │     asr      │  呼叫 ASR API，取得 TXT 逐字稿與 SRT 時間軸
        └──────┬───────┘
        ┌──────┴───────┐        ← Fan-out：兩節點平行執行
   minutes_taker   summarizer
   （時間軸逐字稿） （重點摘要）
        └──────┬───────┘        ← Fan-in：匯聚
        ┌──────┴───────┐
        │    writer    │  整合成最終報告
        └──────────────┘

備註：課程原本的 vLLM 推論伺服器（ws-02.wade0426.me）已隨課程結束關閉，
      本作業改接 Google Gemini 的 OpenAI 相容端點。
      ASR API（3090api.huannago.com）經實測仍可正常使用，故保留原本的呼叫方式。
"""

import os
import time
from pathlib import Path
from typing import TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# ================= 配置區 =================
CONFIG = {
    "asr_api_url": "https://3090api.huannago.com",
    "asr_auth": ("nutc2504", "nutc2504"),   # 老師提供的帳密
}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

AUDIO_PATH = BASE_DIR / "audio" / "Podcast_EP14_20s.wav"
REPORT_PATH = BASE_DIR / "meeting_report.md"


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY，請於 .env 設定。")
    return ChatOpenAI(
        base_url=GEMINI_BASE_URL, api_key=api_key, model=GEMINI_MODEL,
        temperature=0, max_tokens=4096, timeout=180, max_retries=5,
    )


llm = build_llm()


# ================= 0. ASR API 工具函式 =================
def call_asr_api(audio_path: str):
    """上傳音檔並等待轉錄結果（TXT 逐字稿 + SRT 時間軸）。"""
    print(f"🎤 [ASR] 正在上傳音檔: {audio_path} ...", flush=True)

    create_url = f"{CONFIG['asr_api_url']}/api/v1/subtitle/tasks"
    with open(audio_path, "rb") as f:
        r = requests.post(create_url, files={"audio": f},
                          timeout=120, auth=CONFIG["asr_auth"])
    r.raise_for_status()
    task_id = r.json()["id"]
    print(f"⏳ [ASR] 任務 ID: {task_id}，等待轉錄中...", flush=True)

    base = f"{CONFIG['asr_api_url']}/api/v1/subtitle/tasks/{task_id}/subtitle"

    def wait_download(url: str, label: str, max_wait: int = 300):
        """輪詢直到該格式的字幕檔可下載。"""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=15, auth=CONFIG["asr_auth"])
                if resp.status_code == 200 and resp.text.strip():
                    print(f"   ✅ [ASR] {label} 已取得（{len(resp.text)} 字）", flush=True)
                    return resp.text
            except requests.RequestException:
                pass
            time.sleep(2)
        return None

    txt_content = wait_download(f"{base}?type=TXT", "TXT 逐字稿")
    srt_content = wait_download(f"{base}?type=SRT", "SRT 時間軸")

    if not txt_content or not srt_content:
        raise RuntimeError("ASR 轉錄逾時或失敗")

    print("✅ [ASR] 轉錄完成！", flush=True)
    return txt_content, srt_content


# ================= 1. 定義狀態 (State) =================
class MeetingState(TypedDict):
    audio_path: str         # 輸入：音檔路徑
    transcript_txt: str     # 中間產物：純文字稿
    transcript_srt: str     # 中間產物：時間軸稿
    minutes: str            # 輸出 A：詳細逐字稿（時間軸）
    summary: str            # 輸出 B：重點摘要
    final_report: str       # 最終產出：整合報告


# ================= 2. 定義節點 (Nodes) =================
def asr_node(state: MeetingState):
    """負責呼叫 ASR API 的節點。"""
    txt, srt = call_asr_api(state["audio_path"])
    return {"transcript_txt": txt, "transcript_srt": srt}


def minutes_node(state: MeetingState):
    """(平行節點 A) 紀錄員：讀取 SRT，按時間軸逐條列出台詞。"""
    print("📝 [Minutes] 正在整理詳細逐字稿（含時間軸）...", flush=True)

    prompt = f"""你是專業的會議記錄員。請根據以下 SRT 字幕內容，整理出「按時間軸逐一列出」的詳細逐字稿。

格式要求（務必嚴格遵守）：
- 每一條都以 `[開始時間 - 結束時間] 台詞內容` 的形式列出，時間取自 SRT
- 依時間先後排序，不可跳過任何一段
- 將口語贅字修飾為通順的書面語，但不可改變原意、不可省略內容
- 只輸出逐字稿本身，不要加任何前言或結語

SRT 內容：
{state['transcript_srt']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"minutes": response.content.strip()}


def summary_node(state: MeetingState):
    """(平行節點 B) 總結者：讀取 TXT，撰寫高階重點摘要。"""
    print("💡 [Summary] 正在撰寫重點摘要...", flush=True)

    prompt = f"""你是公司的高階特助。請根據以下逐字稿，寫一份 200 字以內的重點摘要。

請包含（若內容中有提到）：
- 核心主題
- 重要觀點或決策
- 待辦事項（Action Items）

若逐字稿內容不足以構成完整會議，請如實摘要其實際內容即可，不要杜撰。
只輸出摘要本身，不要加前言。

逐字稿內容：
{state['transcript_txt']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content.strip()}


def writer_node(state: MeetingState):
    """(匯聚節點) 整合兩個平行節點的產出成最終報告。"""
    print("🖨️ [Writer] 正在整合最終報告...", flush=True)

    report = f"""# 📄 智慧會議紀錄報告

> 音檔：`{Path(state['audio_path']).name}`　│　ASR：3090api.huannago.com　│　LLM：{GEMINI_MODEL}

## 🎯 重點摘要 (Executive Summary)

{state['summary']}

---

## ⏱️ 詳細逐字稿 (Detailed Minutes)

{state['minutes']}

---
*本報告由 LangGraph AI Agent 自動生成*
"""
    return {"final_report": report}


# ================= 3. 建構 Graph =================
workflow = StateGraph(MeetingState)

workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_node)
workflow.add_node("summarizer", summary_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("asr")

# --- 關鍵：平行處理 (Fan-out) ---
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

# --- 關鍵：匯聚 (Fan-in) ---
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

workflow.add_edge("writer", END)

app = workflow.compile()


# ================= 4. 執行 =================
def print_graph():
    """印出 LangGraph 圖結構（需 grandalf；未安裝時不影響主流程）。"""
    try:
        print(app.get_graph().draw_ascii())
    except ImportError:
        print("(未安裝 grandalf，略過圖結構繪製：pip install grandalf)")


def main():
    print_graph()

    if not AUDIO_PATH.exists():
        print(f"❌ 找不到音檔: {AUDIO_PATH}，請確認路徑。")
        return

    print("\n🚀 會議助手啟動中...\n")
    start = time.time()
    result = app.invoke({"audio_path": str(AUDIO_PATH)})
    print(f"\n⏱️ 總耗時 {time.time() - start:.1f} 秒")

    print("\n" + "=" * 60)
    print(result["final_report"])

    REPORT_PATH.write_text(result["final_report"], encoding="utf-8")
    print(f"✅ 報告已儲存至 {REPORT_PATH}")


if __name__ == "__main__":
    main()
