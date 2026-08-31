"""
2026_2504 次世代 DocAI 系統工作坊  Day5 作業
學號：2411332043    姓名：黃晧德

題目：對 data_01~05.txt 分別使用三種切塊方法建立索引，針對 questions.csv 的 20 個問題
      進行檢索，並輸出 20 x 3 = 60 筆結果。

三種切塊方法：
  1. 固定大小切塊 (Fixed-size Chunking)
  2. 滑動視窗切塊 (Sliding Window)
  3. 語意切塊 (Semantic Chunking)

輸出：2411332043_RAG_HW_01.csv
      欄位 = id, q_id, method, retrieve_text, score, source（utf-8-sig 編碼）
"""

import os
import re
import csv
import json
import time
import hashlib
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

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

DATA_FILES = [f"data_0{i}.txt" for i in range(1, 6)]
QUESTIONS_CSV = BASE_DIR / "questions.csv"
OUTPUT_CSV = BASE_DIR / "2411332043_RAG_HW_01.csv"
CACHE_PATH = BASE_DIR / "embedding_cache.pkl"
SCORE_CACHE = BASE_DIR / "score_cache.json"

# --- 切塊參數設定（報告中需要說明的參數）---
FIXED_CHUNK_SIZE = 500          # 固定大小切塊：每塊 500 字，不重疊
SLIDING_WINDOW_SIZE = 500       # 滑動視窗切塊：視窗 500 字
SLIDING_STEP = 250              # 滑動視窗切塊：步長 250 字（等於重疊 250 字 / 50%）
SEMANTIC_PERCENTILE = 25        # 語意切塊：相鄰句相似度低於第 25 百分位即切開
SEMANTIC_MAX_CHARS = 800        # 語意切塊：單塊上限，避免產生過長的塊

METHODS = ["固定大小", "滑動視窗", "語意切塊"]


def get_client() -> OpenAI:
    """建立 Gemini（OpenAI 相容端點）用戶端。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY，請於 .env 設定。")
    return OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)


_last_call_at = [0.0]
MIN_CALL_INTERVAL = 1.5   # 節流：兩次 API 呼叫之間至少間隔的秒數


def _throttle():
    """簡易節流，避免瞬間打爆每分鐘配額。"""
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
    """統一的重試包裝：遇到 429 / 5xx 時退避重試，並尊重 API 建議的等待秒數。"""
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
            print(f"    [重試 {attempt + 1}/{retries}] {message[:70]} → {wait:.0f}s 後重試", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120.0)


# --------------------------------------------------------------------------
# 一、資料讀取
# --------------------------------------------------------------------------
def load_documents() -> dict:
    """讀取 data_01~05.txt，回傳 {檔名: 全文}。"""
    docs = {}
    for name in DATA_FILES:
        path = BASE_DIR / name
        docs[name] = path.read_text(encoding="utf-8").strip()
    return docs


def load_questions() -> list:
    """讀取 questions.csv，回傳 [{q_id, question}, ...]。"""
    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        return [
            {"q_id": row["q_id"], "question": row["questions"]}
            for row in csv.DictReader(f)
            if row.get("questions", "").strip()
        ]


# --------------------------------------------------------------------------
# 二、三種切塊方法
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """移除多餘空白，讓字數計算與切塊結果穩定。"""
    return re.sub(r"\s+", "", text)


def chunk_fixed(text: str, size: int = FIXED_CHUNK_SIZE) -> list:
    """方法一：固定大小切塊。每 size 字切一塊，塊與塊之間不重疊。"""
    body = clean_text(text)
    return [body[i:i + size] for i in range(0, len(body), size) if body[i:i + size].strip()]


def chunk_sliding(text: str, size: int = SLIDING_WINDOW_SIZE, step: int = SLIDING_STEP) -> list:
    """方法二：滑動視窗切塊。視窗大小 size，每次前進 step，形成 (size-step) 字的重疊。"""
    body = clean_text(text)
    chunks = []
    for i in range(0, len(body), step):
        piece = body[i:i + size]
        if not piece.strip():
            continue
        chunks.append(piece)
        if i + size >= len(body):
            break
    return chunks


def split_sentences(text: str) -> list:
    """以中文標點切句，作為語意切塊的最小單位。"""
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_semantic(text: str, embed_fn,
                   percentile: int = SEMANTIC_PERCENTILE,
                   max_chars: int = SEMANTIC_MAX_CHARS) -> list:
    """
    方法三：語意切塊。
    先切句 → 取得每句向量 → 計算相鄰句的餘弦相似度 →
    相似度低於 percentile 百分位處視為「語意轉折」，在該處切開。
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [clean_text(text)] if text.strip() else []

    vectors = embed_fn(sentences)
    sims = [
        float(np.dot(vectors[i], vectors[i + 1]) /
              (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i + 1]) + 1e-10))
        for i in range(len(sentences) - 1)
    ]
    threshold = float(np.percentile(sims, percentile))

    chunks, current = [], sentences[0]
    for i, sim in enumerate(sims):
        nxt = sentences[i + 1]
        # 語意轉折，或該塊已達長度上限 → 切開
        if sim < threshold or len(current) + len(nxt) > max_chars:
            chunks.append(clean_text(current))
            current = nxt
        else:
            current += nxt
    if current.strip():
        chunks.append(clean_text(current))
    return [c for c in chunks if c.strip()]


def build_chunks(method: str, docs: dict, embed_fn) -> list:
    """依指定方法，將所有文件切塊，回傳 [{text, source}, ...]。"""
    results = []
    for source, text in docs.items():
        if method == "固定大小":
            pieces = chunk_fixed(text)
        elif method == "滑動視窗":
            pieces = chunk_sliding(text)
        elif method == "語意切塊":
            pieces = chunk_semantic(text, embed_fn)
        else:
            raise ValueError(f"未知的切塊方法：{method}")
        results.extend({"text": p, "source": source} for p in pieces)
    return results


# --------------------------------------------------------------------------
# 三、向量化與檢索
# --------------------------------------------------------------------------
class Embedder:
    """封裝 embedding API，附帶本機快取，避免重跑時重複付費。"""

    def __init__(self, client: OpenAI):
        self.client = client
        self.cache = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as f:
                self.cache = pickle.load(f)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _request(self, batch: list) -> list:
        try:
            resp = self.client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM
            )
        except TypeError:
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def embed(self, texts: list, batch_size: int = 32) -> list:
        """取得一批文字的向量（有快取者直接取用）。"""
        todo = [t for t in texts if self._key(t) not in self.cache]
        todo = list(dict.fromkeys(todo))
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            vectors = call_with_retry(self._request, batch)
            for text, vec in zip(batch, vectors):
                self.cache[self._key(text)] = vec
            print(f"    embedding {min(i + batch_size, len(todo))}/{len(todo)}")
        if todo:
            self.save()
        return [self.cache[self._key(t)] for t in texts]

    def save(self):
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self.cache, f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve(query_vec: np.ndarray, chunks: list, chunk_vecs: list) -> dict:
    """回傳與問題向量最相似的一個文字塊。"""
    scores = [cosine_similarity(query_vec, v) for v in chunk_vecs]
    best = int(np.argmax(scores))
    return {
        "text": chunks[best]["text"],
        "source": chunks[best]["source"],
        "similarity": scores[best],
    }


# --------------------------------------------------------------------------
# 四、獲取分數函數（使用 API 評分）
# --------------------------------------------------------------------------
SCORE_PROMPT = """你是一位 RAG 檢索品質評分員。
請判斷「檢索段落」是否包含足以回答「問題」所需的資訊，並給出 0 到 1 之間的分數。

評分標準：
1.0 = 段落完整包含問題的答案
0.7 ~ 0.9 = 包含答案主要部分，但細節不全
0.4 ~ 0.6 = 主題相關，但沒有直接回答問題
0.1 ~ 0.3 = 僅有少量關聯
0.0 = 完全不相關

問題：{question}

檢索段落：
{context}

請只輸出 JSON 格式，不要加上任何說明文字：{{"score": 分數}}"""


def parse_json_response(raw: str) -> dict:
    """
    穩健地從模型回覆中取出 JSON。
    （注意：Gemini 的 response_format=json_object 目前會被導向另一組配額，
     容易觸發 429，因此改為在 prompt 中要求輸出 JSON，再於此處解析。）
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


def get_score(client: OpenAI, question: str, retrieve_text: str) -> float:
    """使用 API（LLM 評審）取得該次檢索的分數，範圍 0.0 ~ 1.0。"""

    def _call():
        return client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{
                "role": "user",
                "content": SCORE_PROMPT.format(question=question, context=retrieve_text),
            }],
            max_tokens=2000,
            temperature=0,
        )

    resp = call_with_retry(_call)
    raw = (resp.choices[0].message.content or "").strip()
    data = parse_json_response(raw)
    if "score" in data:
        try:
            score = float(data["score"])
        except (TypeError, ValueError):
            score = 0.0
    else:
        match = re.search(r"[01](?:\.\d+)?", raw)
        score = float(match.group()) if match else 0.0
    return round(max(0.0, min(1.0, score)), 3)


# --------------------------------------------------------------------------
# 五、建立 CSV 函數
# --------------------------------------------------------------------------
def write_csv(rows: list, path: Path = OUTPUT_CSV):
    """輸出結果 CSV（utf-8-sig 編碼）。"""
    fields = ["id", "q_id", "method", "retrieve_text", "score", "source"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已輸出：{path}（共 {len(rows)} 筆）")


def summarize(rows: list) -> dict:
    """統計各切塊方法的平均分數。"""
    summary = {}
    for method in METHODS:
        scores = [r["score"] for r in rows if r["method"] == method]
        summary[method] = round(sum(scores) / len(scores), 4) if scores else 0.0
    return summary


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    client = get_client()
    embedder = Embedder(client)

    print("=== 讀取資料 ===")
    docs = load_documents()
    questions = load_questions()
    print(f"文件 {len(docs)} 份、問題 {len(questions)} 題")

    print("\n=== 問題向量化 ===")
    question_vecs = embedder.embed([q["question"] for q in questions])

    # 分數快取：避免中途中斷後重跑時重複呼叫 API（key = 方法|題號）
    score_cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8")) if SCORE_CACHE.exists() else {}

    rows, row_id = [], 1
    for method in METHODS:
        print(f"\n=== 切塊方法：{method} ===", flush=True)
        chunks = build_chunks(method, docs, embedder.embed)
        print(f"  共 {len(chunks)} 塊，平均長度 {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} 字",
              flush=True)

        chunk_vecs = embedder.embed([c["text"] for c in chunks])

        for q, q_vec in zip(questions, question_vecs):
            hit = retrieve(q_vec, chunks, chunk_vecs)
            cache_key = f"{method}|{q['q_id']}"
            if cache_key in score_cache:
                score = score_cache[cache_key]
            else:
                score = get_score(client, q["question"], hit["text"])
                score_cache[cache_key] = score
                SCORE_CACHE.write_text(json.dumps(score_cache, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
            rows.append({
                "id": row_id,
                "q_id": q["q_id"],
                "method": method,
                "retrieve_text": hit["text"],
                "score": score,
                "source": hit["source"],
            })
            print(f"  [{q['q_id']:>2}] score={score:<5} source={hit['source']}", flush=True)
            row_id += 1

    write_csv(rows)

    print("\n=== 各方法平均分數 ===")
    summary = summarize(rows)
    for method, avg in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {method}：{avg}")
    best = max(summary, key=summary.get)
    print(f"\n效果最好的切塊方法：{best}（平均分數 {summary[best]}）")

    (BASE_DIR / "summary.json").write_text(
        json.dumps({"average_scores": summary, "best_method": best},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
