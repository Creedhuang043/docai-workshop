"""
Day5 作業報告產生器
學號：2411332043    姓名：黃晧德

讀取 2411332043_RAG_HW_01.csv 與 summary.json，產生繳交用的 PDF 報告：
2411332043_RAG_HW_01.pdf
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "2411332043_RAG_HW_01.csv"
SUMMARY_PATH = BASE_DIR / "summary.json"
PDF_PATH = BASE_DIR / "2411332043_RAG_HW_01.pdf"

FONT = "MSung-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

# 與主程式一致的參數（報告中需說明）
PARAMS = {
    "固定大小": "chunk_size = 500 字，overlap = 0（不重疊）",
    "滑動視窗": "window_size = 500 字，step = 250 字（重疊 250 字，重疊率 50%）",
    "語意切塊": "以句子為單位，相鄰句餘弦相似度低於第 25 百分位處切開，單塊上限 800 字",
}

S_TITLE = ParagraphStyle("t", fontName=FONT, fontSize=20, leading=28, spaceAfter=6)
S_SUB = ParagraphStyle("s", fontName=FONT, fontSize=11, leading=16,
                       textColor=colors.HexColor("#555555"), spaceAfter=14)
S_H1 = ParagraphStyle("h1", fontName=FONT, fontSize=14, leading=20, spaceBefore=14,
                      spaceAfter=8, textColor=colors.HexColor("#1a4f8b"))
S_H2 = ParagraphStyle("h2", fontName=FONT, fontSize=12, leading=18, spaceBefore=10,
                      spaceAfter=6, textColor=colors.HexColor("#333333"))
S_BODY = ParagraphStyle("b", fontName=FONT, fontSize=10.5, leading=17, alignment=TA_LEFT,
                        spaceAfter=6)
S_CELL = ParagraphStyle("c", fontName=FONT, fontSize=9, leading=13)
S_SMALL = ParagraphStyle("sm", fontName=FONT, fontSize=9, leading=14,
                         textColor=colors.HexColor("#666666"))


def load_rows() -> list:
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
    return rows


def count_chunks() -> dict:
    """重新執行三種切塊，取得各方法實際產生的區塊數（embedding 走本機快取，不呼叫 API）。"""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "hw01", BASE_DIR / "2411332043_RAG_HW_01.py")
    hw01 = importlib.util.module_from_spec(spec)
    sys.modules["hw01"] = hw01
    spec.loader.exec_module(hw01)

    docs = hw01.load_documents()
    embedder = hw01.Embedder(hw01.get_client())
    return {m: len(hw01.build_chunks(m, docs, embedder.embed))
            for m in ["固定大小", "滑動視窗", "語意切塊"]}


def make_table(data, col_widths, header_bg="#1a4f8b"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build():
    rows = load_rows()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    avg = summary["average_scores"]
    best = summary["best_method"]

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    story = []

    # ---------------- 封面標題 ----------------
    story.append(Paragraph("Day5 作業報告：三種切塊方法之 RAG 檢索效果比較", S_TITLE))
    story.append(Paragraph(
        "2026_2504 次世代 DocAI 系統工作坊　│　學號：2411332043　│　姓名：黃晧德", S_SUB))

    # ---------------- 一、實驗設定 ----------------
    story.append(Paragraph("一、實驗設定", S_H1))
    story.append(Paragraph(
        "本次作業以 data_01.txt ~ data_05.txt 共 5 份文件作為知識庫，針對 questions.csv 中的 "
        "20 個問題，分別以三種切塊方法建立索引並進行檢索，每題取回相似度最高的 1 個文字塊，"
        "共產生 20 × 3 = 60 筆結果。", S_BODY))
    story.append(Paragraph(
        "向量模型使用 Google <b>gemini-embedding-001</b>（維度 768），"
        "檢索方式為餘弦相似度（Cosine Similarity）。", S_BODY))

    # ---------------- 二、問題一：參數設定 ----------------
    story.append(Paragraph("二、問題一：固定大小切塊與滑動視窗切塊的參數設定", S_H1))
    story.append(Paragraph(
        "三種切塊方法的參數設定如下表。其中固定大小與滑動視窗的差別在於「是否重疊」："
        "滑動視窗以 250 字為步長前進，因此相鄰兩塊會有 250 字（50%）的重疊，"
        "目的是避免答案剛好被切斷在兩塊交界處。", S_BODY))

    chunk_counts = count_chunks()
    param_rows = [[Paragraph("<b>切塊方法</b>", S_CELL), Paragraph("<b>參數設定</b>", S_CELL),
                   Paragraph("<b>實際產生塊數</b>", S_CELL)]]
    for method in ["固定大小", "滑動視窗", "語意切塊"]:
        param_rows.append([
            Paragraph(method, S_CELL),
            Paragraph(PARAMS[method], S_CELL),
            Paragraph(f"{chunk_counts[method]} 塊", S_CELL),
        ])
    story.append(make_table(param_rows, [22 * mm, 100 * mm, 40 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "設定理由：500 字約可容納一個完整段落的語意，過小容易切斷語句、過大則會稀釋向量的語意焦點；"
        "重疊率設為 50% 是常見的折衷，能明顯降低邊界斷句的風險，代價是索引量約增加一倍。", S_SMALL))

    # ---------------- 三、問題二：效果比較 ----------------
    story.append(Paragraph("三、問題二：哪一種切塊方法效果最好？", S_H1))
    story.append(Paragraph(
        "評分方式：將每一題檢索到的文字塊與問題一併送入 LLM 評審（Gemini），"
        "依「該段落是否包含足以回答問題的資訊」給予 0 ~ 1 分，再取每種方法 20 題的平均值。", S_BODY))

    score_rows = [[Paragraph("<b>排名</b>", S_CELL), Paragraph("<b>切塊方法</b>", S_CELL),
                   Paragraph("<b>平均分數</b>", S_CELL), Paragraph("<b>滿分(1.0)題數</b>", S_CELL),
                   Paragraph("<b>低分(&lt;0.5)題數</b>", S_CELL)]]
    for rank, (method, score) in enumerate(sorted(avg.items(), key=lambda x: -x[1]), 1):
        full = sum(1 for r in by_method[method] if r["score"] >= 1.0)
        low = sum(1 for r in by_method[method] if r["score"] < 0.5)
        score_rows.append([
            Paragraph(f"{rank}", S_CELL), Paragraph(method, S_CELL),
            Paragraph(f"<b>{score:.4f}</b>", S_CELL),
            Paragraph(f"{full} / 20", S_CELL), Paragraph(f"{low} / 20", S_CELL),
        ])
    story.append(make_table(score_rows, [15 * mm, 35 * mm, 30 * mm, 35 * mm, 35 * mm]))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph(
        f"<b>結論：平均分數最高的切塊方法為「{best}」，平均分數 {avg[best]:.4f}。</b>", S_BODY))

    # ---------------- 四、結果分析 ----------------
    story.append(Paragraph("四、結果分析", S_H1))
    ordered = sorted(avg.items(), key=lambda x: -x[1])
    gap = ordered[0][1] - ordered[-1][1]
    story.append(Paragraph(
        f"三種方法的平均分數分別為："
        f"{'、'.join(f'{m} {s:.4f}' for m, s in ordered)}，"
        f"最高與最低相差 {gap:.4f}。", S_BODY))
    story.append(Paragraph("各方法的特性觀察：", S_H2))
    story.append(Paragraph(
        "• <b>固定大小切塊</b>：實作最簡單、索引量最小，但切點是機械式的字數切分，"
        "若答案剛好橫跨兩塊交界，就會導致檢索到的段落資訊不完整。", S_BODY))
    story.append(Paragraph(
        "• <b>滑動視窗切塊</b>：以重疊換取邊界的容錯能力，同一段內容會出現在多個塊中，"
        "因此答案被切斷的機率大幅降低；缺點是索引量約為固定大小的兩倍，成本與檢索時間增加。", S_BODY))
    story.append(Paragraph(
        "• <b>語意切塊</b>：依相鄰句子的向量相似度找出語意轉折點才切塊，"
        "每一塊在語意上最完整、雜訊最少；但需要額外對每個句子做 embedding，前置成本最高，"
        "且切塊品質會受句子切分規則與相似度門檻的影響。", S_BODY))

    story.append(PageBreak())

    # ---------------- 五、逐題明細 ----------------
    story.append(Paragraph("五、逐題檢索分數明細", S_H1))
    story.append(Paragraph("下表列出 20 題在三種切塊方法下的檢索分數與命中來源檔案。", S_BODY))

    detail = [[Paragraph("<b>題號</b>", S_CELL)] +
              [Paragraph(f"<b>{m}</b>", S_CELL) for m in ["固定大小", "滑動視窗", "語意切塊"]]]
    q_ids = sorted({r["q_id"] for r in rows}, key=lambda x: int(x))
    for qid in q_ids:
        line = [Paragraph(qid, S_CELL)]
        for method in ["固定大小", "滑動視窗", "語意切塊"]:
            hit = next((r for r in by_method[method] if r["q_id"] == qid), None)
            line.append(Paragraph(
                f"{hit['score']:.2f}<br/><font size=7 color='#888888'>{hit['source']}</font>"
                if hit else "-", S_CELL))
        detail.append(line)
    avg_line = [Paragraph("<b>平均</b>", S_CELL)] + [
        Paragraph(f"<b>{avg[m]:.4f}</b>", S_CELL) for m in ["固定大小", "滑動視窗", "語意切塊"]]
    detail.append(avg_line)

    t = make_table(detail, [18 * mm, 45 * mm, 45 * mm, 45 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, len(detail) - 1), (-1, len(detail) - 1),
                           colors.HexColor("#dce6f2"))]))
    story.append(t)

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("六、附註", S_H1))
    story.append(Paragraph(
        "課程原提供的評分 API（ws-02.wade0426.me）已於課程結束後關閉，"
        "因此本作業改以 Gemini LLM 作為評審模型（LLM-as-a-Judge）取得分數，"
        "評分函數為程式中的 <b>get_score()</b>，評分標準已固定於 prompt 中並設定 temperature = 0 以確保可重現性。",
        S_SMALL))
    story.append(Paragraph(
        "完整程式碼見 2411332043_RAG_HW_01.py，完整結果見 2411332043_RAG_HW_01.csv（60 筆，utf-8-sig 編碼）。",
        S_SMALL))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Day5 作業報告 - 2411332043 黃晧德", author="黃晧德",
    )
    doc.build(story)
    print(f"已產生報告：{PDF_PATH}")


if __name__ == "__main__":
    build()
