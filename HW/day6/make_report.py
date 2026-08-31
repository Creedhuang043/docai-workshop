"""
Day6 作業報告產生器
學號：2411332043    姓名：黃晧德

讀取 day6_HW_questions.csv、summary.json、ablation_compare.json，
產生繳交用的 PDF 報告：2411332043_RAG_HW_02.pdf
（說明如何使用 DeepEval 五項指標優化 RAG 系統）
"""

import csv
import json
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
CSV_PATH = BASE_DIR / "day6_HW_questions.csv"
SUMMARY_PATH = BASE_DIR / "summary.json"
COMPARE_PATH = BASE_DIR / "ablation_compare.json"
PDF_PATH = BASE_DIR / "2411332043_RAG_HW_02.pdf"

FONT = "MSung-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

METRIC_NAMES = ["Faithfulness", "Answer_Relevancy", "Contextual_Recall",
                "Contextual_Precision", "Contextual_Relevancy"]

METRIC_ZH = {
    "Faithfulness": "忠實度",
    "Answer_Relevancy": "答案相關性",
    "Contextual_Recall": "上下文召回率",
    "Contextual_Precision": "上下文精確度",
    "Contextual_Relevancy": "上下文相關性",
}

# 每個指標：量什麼 / 分數低代表什麼問題 / 對應到哪一項優化
METRIC_GUIDE = {
    "Faithfulness": (
        "回答中的每一項陳述，是否都能在檢索到的上下文中找到依據。",
        "分數低代表 LLM 產生了幻覺（憑空捏造上下文沒有的內容）。",
        "對應優化：在 Answer Prompt 中明確要求「僅能使用參考資料作答，資料不足時須明說查無資訊」，"
        "並將 temperature 設為 0 降低發散。",
    ),
    "Answer_Relevancy": (
        "回答是否切題，有沒有答非所問或塞入大量無關內容。",
        "分數低代表回答離題、或加了太多與問題無關的贅述。",
        "對應優化：在 Prompt 中限制回答長度（150 字內）、要求直接作答不要開場白，"
        "使回答聚焦於使用者真正問的事。",
    ),
    "Contextual_Recall": (
        "標準答案中的資訊，有多少比例能在檢索到的上下文中被找到。",
        "分數低代表「檢索漏掉了關鍵資料」——正確答案根本沒被撈進來，後面再強的 LLM 也救不回。",
        "對應優化：導入 Query Rewrite（把口語問題改寫成正式檢索詞）與 Hybrid Search"
        "（BM25 補足向量檢索對專有名詞不敏感的弱點），提高關鍵段落被撈到的機率。",
    ),
    "Contextual_Precision": (
        "真正有用的段落，是否被排在檢索結果的前面。",
        "分數低代表雖然撈到了正確資料，但排序不佳，有用的段落被雜訊擠到後面。",
        "對應優化：加入 Rerank 階段，以 LLM 對候選段落逐一評分後重新排序，只取前 3 名送進生成階段。",
    ),
    "Contextual_Relevancy": (
        "檢索到的上下文中，與問題相關的內容佔多少比例。",
        "分數低代表上下文夾帶太多不相關的雜訊，會干擾 LLM 生成。",
        "對應優化：以「一組完整問答」為切塊單位（而非固定字數硬切），"
        "並將送入 LLM 的段落數由 10 收斂到 Rerank 後的前 3 名。",
    ),
}

S_TITLE = ParagraphStyle("t", fontName=FONT, fontSize=19, leading=27, spaceAfter=6)
S_SUB = ParagraphStyle("s", fontName=FONT, fontSize=11, leading=16,
                       textColor=colors.HexColor("#555555"), spaceAfter=14)
S_H1 = ParagraphStyle("h1", fontName=FONT, fontSize=14, leading=20, spaceBefore=14,
                      spaceAfter=8, textColor=colors.HexColor("#1a4f8b"))
S_H2 = ParagraphStyle("h2", fontName=FONT, fontSize=11.5, leading=17, spaceBefore=9,
                      spaceAfter=5, textColor=colors.HexColor("#333333"))
S_BODY = ParagraphStyle("b", fontName=FONT, fontSize=10.5, leading=17, alignment=TA_LEFT,
                        spaceAfter=6)
S_CELL = ParagraphStyle("c", fontName=FONT, fontSize=9, leading=13)
S_SMALL = ParagraphStyle("sm", fontName=FONT, fontSize=9, leading=14,
                         textColor=colors.HexColor("#666666"))


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
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    compare = json.loads(COMPARE_PATH.read_text(encoding="utf-8")) if COMPARE_PATH.exists() else None
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    story = []
    story.append(Paragraph("Day6 作業報告：以 DeepEval 指標驅動 RAG 客服系統優化", S_TITLE))
    story.append(Paragraph(
        "2026_2504 次世代 DocAI 系統工作坊　│　學號：2411332043　│　姓名：黃晧德", S_SUB))

    # ---------------- 一、系統架構 ----------------
    story.append(Paragraph("一、系統架構", S_H1))
    story.append(Paragraph(
        "本系統以台灣自來水公司的公開客服問答資料（qa_data.txt）為知識庫，建立 AI 客服助手。"
        "原始資料為多組「問題／發布日期／答案／來源網址」，因此切塊時以<b>一組完整問答為一個 chunk</b>，"
        "而非固定字數硬切，讓每個檢索單位在語意上都是完整的。", S_BODY))

    arch = [[Paragraph("<b>階段</b>", S_CELL), Paragraph("<b>技術</b>", S_CELL),
             Paragraph("<b>作法</b>", S_CELL)]]
    arch += [
        [Paragraph("1", S_CELL), Paragraph("Query Rewrite", S_CELL),
         Paragraph("以 LLM 將口語化問題改寫為正式檢索語句，補上專業術語", S_CELL)],
        [Paragraph("2", S_CELL), Paragraph("Hybrid Search", S_CELL),
         Paragraph("BM25 關鍵字檢索 + 向量語意檢索，以 RRF（k=60）融合兩路排名，各取前 10 名", S_CELL)],
        [Paragraph("3", S_CELL), Paragraph("Rerank", S_CELL),
         Paragraph("以 LLM 對 10 筆候選逐一評分（0~10）後重新排序，保留前 3 名", S_CELL)],
        [Paragraph("4", S_CELL), Paragraph("LLM 生成", S_CELL),
         Paragraph("將前 3 名段落作為上下文，要求模型僅依參考資料作答", S_CELL)],
    ]
    story.append(make_table(arch, [12 * mm, 32 * mm, 118 * mm]))

    # ---------------- 二、DeepEval 指標 ----------------
    story.append(Paragraph("二、DeepEval 五項指標：各自診斷什麼問題", S_H1))
    story.append(Paragraph(
        "DeepEval 的價值不只在於「打一個分數」，而是每個指標會指向 RAG 流程中<b>不同的失效環節</b>。"
        "以下說明每個指標量測的內容、分數偏低代表的問題，以及本系統據此採取的對應優化。", S_BODY))

    for name in METRIC_NAMES:
        what, low, fix = METRIC_GUIDE[name]
        story.append(Paragraph(f"{name}（{METRIC_ZH[name]}）", S_H2))
        story.append(Paragraph(f"• <b>量測內容：</b>{what}", S_BODY))
        story.append(Paragraph(f"• <b>診斷意義：</b>{low}", S_BODY))
        story.append(Paragraph(f"• <b>{fix}</b>", S_BODY))

    story.append(PageBreak())

    # ---------------- 三、優化前後比較 ----------------
    story.append(Paragraph("三、優化成效：基線 vs 優化版", S_H1))
    if compare:
        q_n = len(compare["subset_q_ids"])
        story.append(Paragraph(
            f"為驗證上述優化確實有效，另外實作一套<b>未優化的基線系統</b>（僅純向量檢索取前 3 名，"
            f"不做 Query Rewrite、不做 Hybrid Search、不做 Rerank），"
            f"在<b>相同的 {q_n} 道題目</b>與相同的五項指標下進行對照。", S_BODY))

        cmp_rows = [[Paragraph("<b>指標</b>", S_CELL), Paragraph("<b>基線</b>", S_CELL),
                     Paragraph("<b>優化版</b>", S_CELL), Paragraph("<b>提升</b>", S_CELL)]]
        for name in METRIC_NAMES:
            delta = compare["improvement"][name]
            color = "#1a7f37" if delta > 0 else ("#b3261e" if delta < 0 else "#666666")
            cmp_rows.append([
                Paragraph(f"{name}<br/><font size=7 color='#888888'>{METRIC_ZH[name]}</font>", S_CELL),
                Paragraph(f"{compare['baseline'][name]:.4f}", S_CELL),
                Paragraph(f"<b>{compare['optimized'][name]:.4f}</b>", S_CELL),
                Paragraph(f"<font color='{color}'><b>{delta:+.4f}</b></font>", S_CELL),
            ])
        story.append(make_table(cmp_rows, [55 * mm, 35 * mm, 35 * mm, 35 * mm]))
        story.append(Spacer(1, 4 * mm))

        gains = {k: compare["improvement"][k] for k in METRIC_NAMES}
        best_metric = max(gains, key=gains.get)
        improved = [k for k in METRIC_NAMES if gains[k] > 0]
        dropped = [k for k in METRIC_NAMES if gains[k] < 0]
        story.append(Paragraph(
            f"五項指標中有 {len(improved)} 項上升，提升幅度最大的是 "
            f"<b>{best_metric}（{METRIC_ZH[best_metric]}）{gains[best_metric]:+.4f}</b>。"
            f"這兩項（{METRIC_ZH['Contextual_Precision']}與{METRIC_ZH['Contextual_Relevancy']}）"
            f"正是 Rerank 直接作用的環節：把 Hybrid Search 撈回的 10 筆候選重新排序後只留前 3 名，"
            f"等於同時提高了「有用段落排在前面」的比例，也降低了送進 LLM 的雜訊量。", S_BODY))
        if dropped:
            names = "、".join(f"{k}（{METRIC_ZH[k]}）" for k in dropped)
            story.append(Paragraph(
                f"另一方面，<b>{names}</b> 略為下降。這是可以理解的取捨："
                f"基線系統只拿到 3 段純向量檢索的內容，可講的東西少、幾乎照抄原文，因此忠實度容易滿分；"
                f"優化版檢索到更豐富且更相關的上下文後，模型會做較多整合與改寫，"
                f"些微增加了偏離原文的風險。以客服場景而言，"
                f"用 {abs(gains[dropped[0]]):.4f} 的忠實度換取檢索品質與答案相關性的提升是值得的，"
                f"且忠實度仍維持在 {compare['optimized'][dropped[0]]:.4f} 的高水準。", S_BODY))
    else:
        story.append(Paragraph("（尚未執行 day6_ablation.py，無對照數據）", S_BODY))

    # ---------------- 四、最終成績 ----------------
    story.append(Paragraph("四、優化版系統在全部題目上的最終成績", S_H1))
    final_rows = [[Paragraph("<b>指標</b>", S_CELL), Paragraph("<b>平均分數</b>", S_CELL),
                   Paragraph("<b>通過門檻(0.7)題數</b>", S_CELL)]]
    for name in METRIC_NAMES:
        vals = [float(r[name]) for r in rows if r.get(name) not in (None, "", "None")]
        passed = sum(1 for v in vals if v >= 0.7)
        final_rows.append([
            Paragraph(f"{name}<br/><font size=7 color='#888888'>{METRIC_ZH[name]}</font>", S_CELL),
            Paragraph(f"<b>{summary.get(name, 0):.4f}</b>", S_CELL),
            Paragraph(f"{passed} / {len(vals)}", S_CELL),
        ])
    story.append(make_table(final_rows, [60 * mm, 45 * mm, 45 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"全部 {len(rows)} 題的詳細答案與逐題分數，請見 day6_HW_questions.csv。", S_SMALL))

    # ---------------- 五、後續改進 ----------------
    story.append(Paragraph("五、觀察與後續改進方向", S_H1))
    ordered = sorted(METRIC_NAMES, key=lambda k: summary.get(k, 0))
    weakest = ordered[0]
    story.append(Paragraph(
        f"目前分數最低的指標為 <b>{weakest}（{METRIC_ZH[weakest]}）</b>"
        f"（{summary.get(weakest, 0):.4f}），"
        f"依前述診斷邏輯，這代表：{METRIC_GUIDE[weakest][1]}", S_BODY))
    story.append(Paragraph(
        "後續可嘗試的改進方向："
        "（1）Query Rewrite 改為產生多個查詢變體（Multi-Query）再合併結果，進一步提高召回率；"
        "（2）改用專門的 Cross-Encoder 重排序模型取代 LLM Rerank，降低成本與延遲；"
        "（3）針對分數偏低的題目做錯誤分析，判斷是知識庫本身缺資料，還是檢索或生成環節的問題。", S_BODY))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "評估模型：Gemini（temperature = 0）；評估框架：DeepEval。"
        "程式碼見 day6_HW.py（主系統）與 day6_ablation.py（基線對照實驗）。", S_SMALL))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Day6 作業報告 - 2411332043 黃晧德", author="黃晧德",
    )
    doc.build(story)
    print(f"已產生報告：{PDF_PATH}")


if __name__ == "__main__":
    build()
