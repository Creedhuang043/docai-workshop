"""
2026_2504 次世代 DocAI 系統工作坊  Day6 作業
學號：2411332043    姓名：黃晧德

題目：根據 qa_data.txt（台灣自來水公司客服問答資料）建立一個 AI 客服助手，
      並使用 DeepEval 五項指標評估系統品質。

使用技術：
  1. Query Rewrite  ── 將口語化問題改寫成適合檢索的正式查詢
  2. Hybrid Search  ── BM25 關鍵字檢索 + 向量語意檢索，以 RRF 融合
  3. Rerank         ── 以 LLM 對候選段落重新排序，取前 K 筆
  4. LLM            ── 依據檢索結果生成客服回答

評估指標（DeepEval）：
  Faithfulness / Answer Relevancy / Contextual Recall /
  Contextual Precision / Contextual Relevancy

輸出：day6_HW_questions.csv（utf-8-sig）
"""

import os
import re
import csv
import json
import time
import pickle
import hashlib
from pathlib import Path

import jieba
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# --------------------------------------------------------------------------
# 基本設定
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768

KB_FILE = BASE_DIR / "qa_data.txt"
GROUND_TRUTH_CSV = BASE_DIR / "questions_answer.csv"
OUTPUT_CSV = BASE_DIR / "day6_HW_questions.csv"
EMBED_CACHE = BASE_DIR / "embedding_cache.pkl"
RAG_CACHE = BASE_DIR / "rag_results.json"
EVAL_CACHE = BASE_DIR / "eval_results.json"

# --- 檢索參數 ---
HYBRID_TOP_K = 10      # Hybrid Search 各路取回的候選數
RERANK_TOP_N = 3       # Rerank 後保留、送進 LLM 的段落數
RRF_K = 60             # Reciprocal Rank Fusion 常數

jieba.setLogLevel(60)


def get_client() -> OpenAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY，請於 .env 設定。")
    return OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)


_last_call_at = [0.0]
MIN_CALL_INTERVAL = 1.5   # 節流：兩次 API 呼叫之間至少間隔的秒數


def _throttle():
    wait = MIN_CALL_INTERVAL - (time.time() - _last_call_at[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_at[0] = time.time()


def _parse_retry_delay(message: str) -> float:
    """從 Google 的 429 錯誤訊息中取出建議的重試秒數。"""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", message)
    if match:
        return float(match.group(1))
    match = re.search(r"'retryDelay':\s*'(\d+)s'", message)
    return float(match.group(1)) if match else 0.0


def call_with_retry(fn, *args, retries: int = 12, **kwargs):
    """遇到 429 / 5xx 時退避重試，並尊重 API 建議的等待秒數。"""
    delay = 5.0
    for attempt in range(retries):
        try:
            _throttle()
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            message = str(exc)
            wait = max(delay, _parse_retry_delay(message) + 2.0)
            print(f"    [重試 {attempt + 1}/{retries}] {message[:70]} → {wait:.0f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120.0)


# --------------------------------------------------------------------------
# 一、知識庫載入與切塊
# --------------------------------------------------------------------------
def load_knowledge_base() -> list:
    """
    qa_data.txt 的結構為多組「問題 / 發布日期 / 答案 / 來源URL」，
    每組以『來源：<網址>』作結，因此以該行為邊界切塊，
    讓每個 chunk 剛好是一組完整的問答（語意最完整的切法）。
    """
    raw = KB_FILE.read_text(encoding="utf-8")
    parts = re.split(r"(來源：\S+)", raw)

    chunks, buffer = [], ""
    for part in parts:
        buffer += part
        if part.startswith("來源："):
            text = buffer.strip()
            if text:
                title = text.split("\n", 1)[0].strip()
                url = part.replace("來源：", "").strip()
                chunks.append({"text": text, "title": title, "source": url})
            buffer = ""
    if buffer.strip():
        chunks.append({"text": buffer.strip(), "title": buffer.strip()[:30], "source": ""})
    return chunks


def load_questions() -> list:
    """讀取題目與標準答案（標準答案供 DeepEval 的 Contextual 指標使用）。"""
    with open(GROUND_TRUTH_CSV, encoding="utf-8-sig") as f:
        return [
            {"q_id": r["q_id"], "question": r["questions"].strip(),
             "expected": r.get("answer", "").strip()}
            for r in csv.DictReader(f) if r.get("questions", "").strip()
        ]


# --------------------------------------------------------------------------
# 二、向量化（含本機快取）
# --------------------------------------------------------------------------
class Embedder:
    def __init__(self, client: OpenAI):
        self.client = client
        self.cache = pickle.load(open(EMBED_CACHE, "rb")) if EMBED_CACHE.exists() else {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _request(self, batch: list) -> list:
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM
        )
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def embed(self, texts: list, batch_size: int = 16) -> list:
        todo = list(dict.fromkeys(t for t in texts if self._key(t) not in self.cache))
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            for text, vec in zip(batch, call_with_retry(self._request, batch)):
                self.cache[self._key(text)] = vec
            print(f"    embedding {min(i + batch_size, len(todo))}/{len(todo)}", flush=True)
        if todo:
            pickle.dump(self.cache, open(EMBED_CACHE, "wb"))
        return [self.cache[self._key(t)] for t in texts]


def tokenize(text: str) -> list:
    """中文斷詞，供 BM25 使用。"""
    return [w for w in jieba.lcut(text) if w.strip()]


def parse_json_response(raw: str) -> dict:
    """
    穩健地從模型回覆中取出 JSON。
    （Gemini 的 response_format=json_object 目前會被導向另一組配額而容易觸發 429，
     因此改為在 prompt 中要求輸出 JSON，再於此處解析。）
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# --------------------------------------------------------------------------
# 三、AI 客服助手（Query Rewrite → Hybrid Search → Rerank → LLM）
# --------------------------------------------------------------------------
REWRITE_PROMPT = """你是台灣自來水公司客服系統的查詢改寫助手。
使用者的問題常是口語化的描述，請改寫成適合在「自來水公司官方問答知識庫」中檢索的查詢語句。

規則：
1. 保留原意，補上正式用語與專業術語（例如「紅色線蟲」→「顫蚓 紅蟲 水質 生物」）。
2. 只輸出改寫後的查詢關鍵語句，不要解釋、不要加標點以外的符號。
3. 長度控制在 40 字以內。

使用者問題：{question}

改寫後查詢："""

RERANK_PROMPT = """你是檢索結果的重排序評分員。
請針對「使用者問題」，評估以下每一段「候選資料」的相關程度，給 0 到 10 分。

使用者問題：{question}

候選資料：
{candidates}

請只輸出 JSON：{{"scores": [第1段分數, 第2段分數, ...]}}
分數數量必須與候選資料數量相同。"""

ANSWER_PROMPT = """你是台灣自來水公司的 AI 客服助理。
請「只根據」以下參考資料回答使用者的問題。

規則：
1. 僅使用參考資料中的內容作答，不可自行編造或補充資料以外的資訊。
2. 若參考資料不足以回答，請明確說明「目前資料中查無相關資訊」。
3. 用親切、清楚的口吻，以繁體中文回答，長度控制在 150 字以內。
4. 直接給出答案，不要說「根據參考資料」之類的開場白。

參考資料：
{context}

使用者問題：{question}

回答："""


class WaterCustomerServiceRAG:
    """台水 AI 客服助手：Query Rewrite + Hybrid Search + Rerank + LLM。"""

    def __init__(self, client: OpenAI, embedder: Embedder):
        self.client = client
        self.embedder = embedder

        print("=== 建立知識庫索引 ===", flush=True)
        self.chunks = load_knowledge_base()
        print(f"知識庫切塊完成：{len(self.chunks)} 組問答", flush=True)

        # BM25 稀疏索引
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        # 向量稠密索引
        self.vectors = self.embedder.embed([c["text"] for c in self.chunks])
        print("索引建立完成（BM25 + 向量）", flush=True)

    # ---- 技術 1：Query Rewrite ----
    def query_rewrite(self, question: str) -> str:
        def _call():
            return self.client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                max_tokens=2000, temperature=0,
            )
        text = (call_with_retry(_call).choices[0].message.content or "").strip()
        return text or question

    # ---- 技術 2：Hybrid Search（BM25 + 向量，RRF 融合）----
    def hybrid_search(self, query: str, top_k: int = HYBRID_TOP_K) -> list:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = np.argsort(bm25_scores)[::-1][:top_k]

        q_vec = self.embedder.embed([query])[0]
        dense_scores = np.array([
            float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-10))
            for v in self.vectors
        ])
        dense_rank = np.argsort(dense_scores)[::-1][:top_k]

        # Reciprocal Rank Fusion：兩路排名以 1/(k+rank) 加總
        fused = {}
        for rank, idx in enumerate(bm25_rank):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dense_rank):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [self.chunks[i] for i in order]

    # ---- 技術 3：Rerank ----
    def rerank(self, question: str, candidates: list, top_n: int = RERANK_TOP_N) -> list:
        if not candidates:
            return []
        listing = "\n\n".join(
            f"[第{i + 1}段]\n{c['text'][:500]}" for i, c in enumerate(candidates)
        )

        def _call():
            return self.client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user",
                           "content": RERANK_PROMPT.format(question=question, candidates=listing)}],
                max_tokens=2000, temperature=0,
            )

        try:
            raw = (call_with_retry(_call).choices[0].message.content or "").strip()
            scores = parse_json_response(raw).get("scores", [])
            scores = [float(s) for s in scores][:len(candidates)]
        except Exception:
            scores = []
        if len(scores) != len(candidates):
            return candidates[:top_n]

        order = np.argsort(scores)[::-1][:top_n]
        return [candidates[i] for i in order]

    # ---- 技術 4：LLM 生成回答 ----
    def generate_answer(self, question: str, contexts: list) -> str:
        context_text = "\n\n---\n\n".join(c["text"] for c in contexts)

        def _call():
            return self.client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user",
                           "content": ANSWER_PROMPT.format(context=context_text, question=question)}],
                max_tokens=3000, temperature=0,
            )
        return (call_with_retry(_call).choices[0].message.content or "").strip()

    def answer(self, question: str) -> dict:
        """完整流程：改寫 → 混合檢索 → 重排序 → 生成。"""
        rewritten = self.query_rewrite(question)
        candidates = self.hybrid_search(rewritten)
        top_contexts = self.rerank(question, candidates)
        answer = self.generate_answer(question, top_contexts)
        return {
            "rewritten_query": rewritten,
            "answer": answer,
            "retrieval_context": [c["text"] for c in top_contexts],
            "source": ", ".join(dict.fromkeys(c["source"] for c in top_contexts if c["source"])),
        }


# --------------------------------------------------------------------------
# 四、DeepEval 五項指標評估
# --------------------------------------------------------------------------
def build_eval_model():
    """建立 DeepEval 使用的評審模型（Gemini）。"""
    from deepeval.models import GeminiModel
    return GeminiModel(model=GEMINI_MODEL, api_key=os.getenv("GEMINI_API_KEY"), temperature=0)


def evaluate_one(model, item: dict, result: dict) -> dict:
    """對單一題目跑 DeepEval 五項指標，回傳 {指標名: 分數}。"""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric,
        ContextualPrecisionMetric, ContextualRelevancyMetric,
    )

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected"],
        retrieval_context=result["retrieval_context"],
    )

    metrics = {
        "Faithfulness": FaithfulnessMetric(model=model, threshold=0.7),
        "Answer_Relevancy": AnswerRelevancyMetric(model=model, threshold=0.7),
        "Contextual_Recall": ContextualRecallMetric(model=model, threshold=0.7),
        "Contextual_Precision": ContextualPrecisionMetric(model=model, threshold=0.7),
        "Contextual_Relevancy": ContextualRelevancyMetric(model=model, threshold=0.7),
    }

    scores = {}
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores[name] = round(float(metric.score), 3)
        except Exception as exc:
            print(f"    [{name}] 評估失敗：{str(exc)[:70]}", flush=True)
            scores[name] = None
    return scores


# --------------------------------------------------------------------------
# 五、輸出 CSV
# --------------------------------------------------------------------------
def write_csv(rows: list, path: Path = OUTPUT_CSV):
    fields = ["q_id", "questions", "answer", "Faithfulness", "Answer_Relevancy",
              "Contextual_Recall", "Contextual_Precision", "Contextual_Relevancy"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已輸出：{path}（共 {len(rows)} 筆）", flush=True)


def print_summary(rows: list):
    metrics = ["Faithfulness", "Answer_Relevancy", "Contextual_Recall",
               "Contextual_Precision", "Contextual_Relevancy"]
    print("\n=== DeepEval 平均分數 ===", flush=True)
    summary = {}
    for m in metrics:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        summary[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
        print(f"  {m:<22}{summary[m]}", flush=True)
    (BASE_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    client = get_client()
    embedder = Embedder(client)
    questions = load_questions()
    print(f"題目數：{len(questions)}\n", flush=True)

    rag = WaterCustomerServiceRAG(client, embedder)

    # --- 階段一：RAG 產生答案（可續跑）---
    rag_results = json.loads(RAG_CACHE.read_text(encoding="utf-8")) if RAG_CACHE.exists() else {}
    print("\n=== 階段一：AI 客服回答 ===", flush=True)
    for item in questions:
        qid = item["q_id"]
        if qid in rag_results:
            continue
        print(f"[{qid}] {item['question']}", flush=True)
        rag_results[qid] = rag.answer(item["question"])
        print(f"     改寫→ {rag_results[qid]['rewritten_query']}", flush=True)
        print(f"     回答→ {rag_results[qid]['answer'][:60]}...", flush=True)
        RAG_CACHE.write_text(json.dumps(rag_results, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # --- 階段二：DeepEval 評估（可續跑）---
    print("\n=== 階段二：DeepEval 五項指標評估 ===", flush=True)
    model = build_eval_model()
    eval_results = json.loads(EVAL_CACHE.read_text(encoding="utf-8")) if EVAL_CACHE.exists() else {}
    for item in questions:
        qid = item["q_id"]
        if qid in eval_results:
            continue
        print(f"[{qid}] 評估中…", flush=True)
        eval_results[qid] = evaluate_one(model, item, rag_results[qid])
        print(f"     {eval_results[qid]}", flush=True)
        EVAL_CACHE.write_text(json.dumps(eval_results, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # --- 階段三：輸出 CSV ---
    rows = []
    for item in questions:
        qid = item["q_id"]
        row = {"q_id": qid, "questions": item["question"], "answer": rag_results[qid]["answer"]}
        row.update(eval_results.get(qid, {}))
        rows.append(row)

    write_csv(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
