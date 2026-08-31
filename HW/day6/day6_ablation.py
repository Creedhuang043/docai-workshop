"""
Day6 消融實驗（Ablation Study）
學號：2411332043    姓名：黃晧德

目的：建立一個「未優化的陽春 RAG」作為基線（Baseline），
      與 day6_HW.py 的優化版（Query Rewrite + Hybrid Search + Rerank）在
      同一批問題、同一組 DeepEval 指標下比較，證明各項優化確實有效。

基線設計（刻意拿掉所有優化）：
  - 不做 Query Rewrite：直接拿使用者的口語化問題去檢索
  - 不做 Hybrid Search：只用向量檢索（Dense only），沒有 BM25
  - 不做 Rerank：直接取相似度前 3 名

輸出：baseline_results.json、ablation_compare.json
"""

import json
from pathlib import Path

import numpy as np

from day6_HW import (
    Embedder, get_client, load_questions, load_knowledge_base,
    build_eval_model, evaluate_one, call_with_retry,
    GEMINI_MODEL, ANSWER_PROMPT,
)

BASE_DIR = Path(__file__).parent
BASELINE_CACHE = BASE_DIR / "baseline_results.json"
BASELINE_EVAL_CACHE = BASE_DIR / "baseline_eval.json"
COMPARE_PATH = BASE_DIR / "ablation_compare.json"

# 為控制 API 花費，基線只跑前 N 題；優化版比較時也取同樣這 N 題，確保公平
SUBSET_N = 10
TOP_K = 3

METRIC_NAMES = ["Faithfulness", "Answer_Relevancy", "Contextual_Recall",
                "Contextual_Precision", "Contextual_Relevancy"]


class NaiveRAG:
    """基線系統：純向量檢索 + 直接生成，不做任何查詢改寫或重排序。"""

    def __init__(self, client, embedder: Embedder):
        self.client = client
        self.embedder = embedder
        self.chunks = load_knowledge_base()
        self.vectors = self.embedder.embed([c["text"] for c in self.chunks])

    def retrieve(self, question: str, top_k: int = TOP_K) -> list:
        q_vec = self.embedder.embed([question])[0]
        scores = [
            float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-10))
            for v in self.vectors
        ]
        order = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in order]

    def answer(self, question: str) -> dict:
        contexts = self.retrieve(question)
        context_text = "\n\n---\n\n".join(c["text"] for c in contexts)

        def _call():
            return self.client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user",
                           "content": ANSWER_PROMPT.format(context=context_text,
                                                           question=question)}],
                max_tokens=3000, temperature=0,
            )

        answer = (call_with_retry(_call).choices[0].message.content or "").strip()
        return {
            "answer": answer,
            "retrieval_context": [c["text"] for c in contexts],
            "source": ", ".join(dict.fromkeys(c["source"] for c in contexts if c["source"])),
        }


def average(results: dict, q_ids: list) -> dict:
    """計算指定題目集合的各指標平均。"""
    out = {}
    for name in METRIC_NAMES:
        vals = [results[q][name] for q in q_ids
                if q in results and isinstance(results[q].get(name), (int, float))]
        out[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out


def main():
    client = get_client()
    embedder = Embedder(client)
    questions = load_questions()[:SUBSET_N]
    q_ids = [q["q_id"] for q in questions]
    print(f"基線實驗題數：{len(questions)}（q_id: {', '.join(q_ids)}）\n", flush=True)

    print("=== 建立基線系統（Dense-only，無改寫、無 Rerank）===", flush=True)
    naive = NaiveRAG(client, embedder)

    # --- 基線回答 ---
    results = json.loads(BASELINE_CACHE.read_text(encoding="utf-8")) if BASELINE_CACHE.exists() else {}
    print("\n=== 基線系統回答 ===", flush=True)
    for item in questions:
        qid = item["q_id"]
        if qid in results:
            continue
        results[qid] = naive.answer(item["question"])
        print(f"[{qid}] {results[qid]['answer'][:60]}…", flush=True)
        BASELINE_CACHE.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # --- 基線評估 ---
    print("\n=== 基線系統 DeepEval 評估 ===", flush=True)
    model = build_eval_model()
    evals = json.loads(BASELINE_EVAL_CACHE.read_text(encoding="utf-8")) if BASELINE_EVAL_CACHE.exists() else {}
    for item in questions:
        qid = item["q_id"]
        if qid in evals:
            continue
        print(f"[{qid}] 評估中…", flush=True)
        evals[qid] = evaluate_one(model, item, results[qid])
        print(f"     {evals[qid]}", flush=True)
        BASELINE_EVAL_CACHE.write_text(json.dumps(evals, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    # --- 與優化版比較 ---
    optimized_eval_path = BASE_DIR / "eval_results.json"
    if not optimized_eval_path.exists():
        print("\n尚未有優化版評估結果（請先執行 day6_HW.py），略過比較。", flush=True)
        return

    optimized = json.loads(optimized_eval_path.read_text(encoding="utf-8"))
    base_avg = average(evals, q_ids)
    opt_avg = average(optimized, q_ids)

    compare = {
        "subset_q_ids": q_ids,
        "baseline": base_avg,
        "optimized": opt_avg,
        "improvement": {k: round(opt_avg[k] - base_avg[k], 4) for k in METRIC_NAMES},
    }
    COMPARE_PATH.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 基線 vs 優化版（同樣 %d 題）===" % len(q_ids), flush=True)
    print(f"{'指標':<24}{'基線':>10}{'優化版':>10}{'提升':>10}")
    for k in METRIC_NAMES:
        print(f"{k:<24}{base_avg[k]:>10.4f}{opt_avg[k]:>10.4f}{compare['improvement'][k]:>+10.4f}")
    print(f"\n已輸出：{COMPARE_PATH}", flush=True)


if __name__ == "__main__":
    main()
