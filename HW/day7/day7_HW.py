"""
2026_2504 次世代 DocAI 系統工作坊  Day7 作業
學號：2411332043    姓名：黃晧德

題目：
  1. 使用 IDP 技術處理 data/ 底下 5 份檔案（1.pdf, 2.pdf, 3.pdf, 4.png, 5.docx）
  2. 辨識出哪些文檔被注入惡意提示詞（輸出報告供截圖）
  3. 使用 RAG 技術製作 AI 問答助手
  4. 使用 DeepEval 四個指標驗證系統
  5. 輸出 day7_hw_submit.csv（欄位：q_id, questions, answer, source）

模組分工：
  idp.py            ── IDP 文件處理（PDF 文字層 / Gemini Vision OCR / DOCX）
  injection_scan.py ── 惡意提示詞注入偵測（規則式 + LLM 判讀）
  rag_qa.py         ── RAG 問答系統（切塊 / 向量檢索 / 生成，含注入消毒）
  day7_HW.py        ── 本檔，主流程 + DeepEval 評估
"""

import os
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from injection_scan import run_scan
from rag_qa import RAGSystem

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
QUESTIONS_CSV = DATA_DIR / "questions.csv"
GROUND_TRUTH_CSV = DATA_DIR / "questions_answer.csv"
SUBMIT_CSV = BASE_DIR / "day7_hw_submit.csv"
RAG_CACHE = OUTPUT_DIR / "rag_results.json"
EVAL_CACHE = OUTPUT_DIR / "eval_results.json"
EVAL_CSV = BASE_DIR / "day7_deepeval_results.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# DeepEval 四項指標
METRIC_NAMES = ["Faithfulness", "Answer_Relevancy", "Contextual_Precision", "Contextual_Recall"]


# --------------------------------------------------------------------------
# 一、讀取題目與標準答案
# --------------------------------------------------------------------------
def load_questions() -> list:
    """讀取題目；若有標準答案檔則一併載入（供 Contextual 指標使用）。"""
    expected = {}
    if GROUND_TRUTH_CSV.exists():
        with open(GROUND_TRUTH_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    expected[row["id"]] = row.get("answer", "").strip()

    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        return [
            {
                "q_id": row["id"],
                "questions": row["questions"].strip(),
                "expected": expected.get(row["id"], ""),
            }
            for row in csv.DictReader(f)
            if row.get("questions", "").strip()
        ]


# --------------------------------------------------------------------------
# 二、DeepEval 評估
# --------------------------------------------------------------------------
def build_eval_model():
    """建立 DeepEval 評審模型（Gemini）。"""
    from deepeval.models import GeminiModel
    return GeminiModel(model=GEMINI_MODEL, api_key=os.getenv("GEMINI_API_KEY"), temperature=0)


def evaluate_one(model, item: dict, result: dict) -> dict:
    """對單一題目跑 DeepEval 四項指標。"""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric, AnswerRelevancyMetric,
        ContextualPrecisionMetric, ContextualRecallMetric,
    )

    test_case = LLMTestCase(
        input=item["questions"],
        actual_output=result["answer"],
        expected_output=item["expected"],
        retrieval_context=[r["text"] for r in result["retrieved"]],
    )

    metrics = {
        "Faithfulness": FaithfulnessMetric(model=model, threshold=0.7),
        "Answer_Relevancy": AnswerRelevancyMetric(model=model, threshold=0.7),
        "Contextual_Precision": ContextualPrecisionMetric(model=model, threshold=0.7),
        "Contextual_Recall": ContextualRecallMetric(model=model, threshold=0.7),
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
# 三、輸出
# --------------------------------------------------------------------------
def write_submit_csv(rows: list):
    """輸出作業指定的 day7_hw_submit.csv（q_id, questions, answer, source）。"""
    fields = ["q_id", "questions", "answer", "source"]
    with open(SUBMIT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])
    print(f"\n已輸出：{SUBMIT_CSV}（共 {len(rows)} 筆）", flush=True)


def write_eval_csv(rows: list):
    """另外輸出 DeepEval 評估結果，供報告與截圖使用。"""
    fields = ["q_id", "questions", "answer", "source"] + METRIC_NAMES
    with open(EVAL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已輸出：{EVAL_CSV}", flush=True)


def print_summary(rows: list) -> dict:
    print("\n=== DeepEval 四項指標平均分數 ===", flush=True)
    summary = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        summary[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
        print(f"  {name:<24}{summary[name]}", flush=True)
    (OUTPUT_DIR / "deepeval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    questions = load_questions()

    print("=== Step 1/4：IDP + 惡意提示詞注入偵測 ===", flush=True)
    report = run_scan()
    injected = report.get("injected_documents", [])
    print(f"偵測結果：被注入惡意提示詞的文件 = {injected}\n", flush=True)

    print("=== Step 2/4：建立 RAG 索引 ===", flush=True)
    rag = RAGSystem()
    print(f"索引完成，共 {len(rag.chunks)} 個文字塊\n", flush=True)

    print("=== Step 3/4：RAG 問答 ===", flush=True)
    rag_results = json.loads(RAG_CACHE.read_text(encoding="utf-8")) if RAG_CACHE.exists() else {}
    for item in questions:
        qid = item["q_id"]
        if qid in rag_results:
            continue
        rag_results[qid] = rag.answer(item["questions"])
        print(f"[{qid}] {item['questions']}", flush=True)
        print(f"     → {rag_results[qid]['answer'][:70]}… (source={rag_results[qid]['source']})",
              flush=True)
        RAG_CACHE.write_text(json.dumps(rag_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Step 4/4：DeepEval 四項指標評估 ===", flush=True)
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

    rows = []
    for item in questions:
        qid = item["q_id"]
        row = {
            "q_id": qid,
            "questions": item["questions"],
            "answer": rag_results[qid]["answer"],
            "source": rag_results[qid]["source"],
        }
        row.update(eval_results.get(qid, {}))
        rows.append(row)

    write_submit_csv(rows)
    write_eval_csv(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
