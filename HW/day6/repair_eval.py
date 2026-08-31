"""
補跑 Day6 中因 API 暫時性錯誤（503 模型忙碌）而失敗的個別 DeepEval 指標。

只重跑 eval_results.json 裡值為 null 的指標，成功後更新該檔並重新產生 CSV。
"""

import csv
import json
import time
from pathlib import Path

from day6_HW import (
    load_questions, build_eval_model, write_csv, print_summary,
    EVAL_CACHE, RAG_CACHE,
)

BASE_DIR = Path(__file__).parent
MAX_ATTEMPTS = 5


def measure_one(model, metric_name: str, item: dict, result: dict):
    """重跑單一指標，回傳分數；失敗回傳 None。"""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric,
        ContextualPrecisionMetric, ContextualRelevancyMetric,
    )

    factory = {
        "Faithfulness": FaithfulnessMetric,
        "Answer_Relevancy": AnswerRelevancyMetric,
        "Contextual_Recall": ContextualRecallMetric,
        "Contextual_Precision": ContextualPrecisionMetric,
        "Contextual_Relevancy": ContextualRelevancyMetric,
    }[metric_name]

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected"],
        retrieval_context=result["retrieval_context"],
    )

    delay = 5.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            metric = factory(model=model, threshold=0.7)
            metric.measure(test_case)
            return round(float(metric.score), 3)
        except Exception as exc:
            print(f"      嘗試 {attempt}/{MAX_ATTEMPTS} 失敗：{str(exc)[:70]}", flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
    return None


def main():
    questions = {q["q_id"]: q for q in load_questions()}
    rag_results = json.loads(RAG_CACHE.read_text(encoding="utf-8"))
    evals = json.loads(EVAL_CACHE.read_text(encoding="utf-8"))

    todo = [(qid, name) for qid, scores in evals.items()
            for name, value in scores.items() if value is None]
    if not todo:
        print("沒有需要補跑的指標。")
        return

    print(f"需補跑 {len(todo)} 個指標：", flush=True)
    for qid, name in todo:
        print(f"  q{qid} - {name}", flush=True)

    model = build_eval_model()
    fixed = 0
    for qid, name in todo:
        print(f"\n[q{qid}] {name} 補跑中…", flush=True)
        score = measure_one(model, name, questions[qid], rag_results[qid])
        evals[qid][name] = score
        if score is not None:
            fixed += 1
            print(f"      ✔ 分數 = {score}", flush=True)
        else:
            print("      ✘ 仍然失敗", flush=True)
        EVAL_CACHE.write_text(json.dumps(evals, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n補跑完成：{fixed}/{len(todo)} 成功", flush=True)

    # --- 重新產生 CSV 與統計 ---
    rows = []
    for qid, item in questions.items():
        row = {"q_id": qid, "questions": item["question"], "answer": rag_results[qid]["answer"]}
        row.update(evals.get(qid, {}))
        rows.append(row)
    write_csv(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
