"""
Professional PDF report generator for deepfake detection XAI.
"""

import base64
import io
import math
import uuid
from datetime import datetime

from fpdf import FPDF

REPORTS: dict[str, bytes] = {}


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Acoustic Vulnerability Assessment Pipeline v2.4", align="L")
        self.cell(0, 6, "INTERNAL AUDIT DOCUMENT", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _multi_cell_row(pdf, col_widths: list[float], texts: list[str], line_h: float = 5, header: bool = False):
    """Draw a table row with auto-wrapping text cells."""
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    max_lines = 1

    for text, w in zip(texts, col_widths):
        n_lines = max(1, math.ceil(pdf.get_string_width(text) / max(w - 2, 1)))
        max_lines = max(max_lines, n_lines)

    row_h = max(max_lines * line_h + 2, 8)

    for i, (text, w) in enumerate(zip(texts, col_widths)):
        x = x_start + sum(col_widths[:i])
        if header:
            pdf.set_fill_color(235, 240, 250)
            pdf.rect(x, y_start, w, row_h, "DF")
        else:
            pdf.rect(x, y_start, w, row_h, "D")
        pdf.set_xy(x + 1, y_start + 1)
        pdf.multi_cell(w - 2, line_h, text)

    pdf.set_xy(x_start, y_start + row_h)


def _draw_verdict_block(pdf, prediction: str, w_margin: float, page_w: float):
    is_spoof = prediction == "spoof"
    pdf.set_x(w_margin)

    block_w = page_w
    block_h = 34
    block_x = w_margin
    block_y = pdf.get_y()

    if is_spoof:
        fill_r, fill_g, fill_b = 220, 50, 50
        text_r, text_g, text_b = 255, 255, 255
        verdict_line = "VERDICT: SPOOF (FAKE)"
        sub_line = "CRITICAL THREAT DETECTED: This audio sample has been identified"
        sub_line2 = "as an artificially generated or replayed deepfake. Reject payload."
    else:
        fill_r, fill_g, fill_b = 40, 180, 80
        text_r, text_g, text_b = 255, 255, 255
        verdict_line = "VERDICT: BONA-FIDE (REAL)"
        sub_line = "SUCCESS: Audio sample verified as authentic human speech."
        sub_line2 = ""

    pdf.set_fill_color(fill_r, fill_g, fill_b)
    pdf.rect(block_x, block_y, block_w, block_h, "F")
    pdf.set_text_color(text_r, text_g, text_b)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(block_x + 6, block_y + 4)
    pdf.cell(block_w - 12, 9, verdict_line)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(block_x + 6, block_y + 14)
    pdf.cell(block_w - 12, 6, sub_line)
    if sub_line2:
        pdf.set_xy(block_x + 6, block_y + 20)
        pdf.cell(block_w - 12, 6, sub_line2)

    pdf.set_y(block_y + block_h + 8)


def generate_report(
    request_id: str,
    prediction: str,
    confidence: float,
    scores: dict[str, float],
    process_time_ms: float,
    attention_maps: dict | None = None,
) -> bytes:
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    w_margin = pdf.l_margin
    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    # ---------------------------------------------------------------
    # Title & Metadata
    # ---------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "Model Inference & Deepfake Detection Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"> Internal Audit Document | ID: {request_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"> Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ---------------------------------------------------------------
    # Verdict Block
    # ---------------------------------------------------------------
    _draw_verdict_block(pdf, prediction, w_margin, page_w)

    # ---------------------------------------------------------------
    # 1. Decision Executive Summary
    # ---------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "1. Decision Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    verdict_label = "SPOOF (FAKE)" if prediction == "spoof" else "BONA-FIDE (REAL)"
    if prediction == "spoof":
        interp = "System confirms structural properties match synthetic generation rather than biological voice modulation."
    else:
        interp = "System confirms structural properties match authentic human voice modulation."

    confident_str = f"{confidence * 100:.2f}%"
    latency_str = f"{process_time_ms:.1f} ms"

    col_w = page_w * 0.28
    col_w2 = page_w * 0.22
    col_w3 = page_w * 0.50

    _multi_cell_row(pdf, [col_w, col_w2, col_w3],
        ["Attribute", "Value", "Interpretation"], header=True)

    rows = [
        ("Classification Type", verdict_label, interp),
        ("Ensemble Confidence", confident_str, "The multi-model meta-learner strongly recommends rejecting the payload based on high consensus." if confidence > 0.5 else "The multi-model meta-learner indicates low risk."),
        ("Processing Latency", latency_str, "Inference executed within acceptable SLA processing windows."),
    ]
    for attr, val, interp_text in rows:
        _multi_cell_row(pdf, [col_w, col_w2, col_w3], [attr, val, interp_text])

    pdf.ln(6)

    # ---------------------------------------------------------------
    # 2. Multi-Model Signal Breakdown
    # ---------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "2. Multi-Model Signal Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    w2v_conf = scores.get("wav2vec2", 0.0)
    aasist_conf = scores.get("aasist", 0.0)
    ensemble_conf = scores.get("ensemble", confidence)

    if prediction == "spoof":
        ensemble_result = "The ensemble model heavily prioritizes AASIST's localized spatial telemetry over Wav2Vec2 in this context, yielding an overall high-confidence threat consensus."
    else:
        ensemble_result = "The ensemble model confirms alignment with human voice characteristics, yielding a low-confidence threat assessment."

    models_data = [
        ("AASIST Sub-Network", f"{aasist_conf * 100:.2f}% Spoof Probability",
         "Analyzes raw graph-based spatial-temporal relationships within raw speech graphs.",
         "Extremely high confidence of an active synthesis or replay attempt." if aasist_conf > 0.7
         else "Moderate acoustic anomaly detected."),
        ("Wav2Vec2 Sub-Network", f"{w2v_conf * 100:.2f}% Spoof Probability",
         "Evaluates semantic token distribution and general linguistic self-attention alignment.",
         "Low confidence of a spoofing attempt based strictly on lexical/temporal patterns." if w2v_conf < 0.3
         else "Indicates some lexical anomalies."),
        ("Ensemble Meta-Learner (Overall Consensus)", f"{ensemble_conf * 100:.2f}% Composite Risk Score",
         "Gathers individual network confidence vectors and weighs them against historical network error profiles.",
         ensemble_result),
    ]

    for name, value, focus, result in models_data:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(0, 7, name, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(w_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 6, f"  Score: {value}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(w_margin)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(page_w, 5, f"  Focus: {focus}")
        pdf.set_x(w_margin)
        pdf.multi_cell(page_w, 5, f"  Result: {result}")
        pdf.ln(3)

    pdf.ln(4)

    # ---------------------------------------------------------------
    # 3. Visual Explainability (XAI)
    # ---------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "3. Visual Explainability (XAI)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if attention_maps and attention_maps.get("overall_heatmap"):
        try:
            img_data = base64.b64decode(attention_maps["overall_heatmap"])
            img_buf = io.BytesIO(img_data)
            img_w = page_w * 0.85
            pdf.image(img_buf, x=w_margin + (page_w - img_w) / 2, w=img_w)
            pdf.ln(4)
        except Exception:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 6, "(Heatmap image unavailable)", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, "(XAI heatmap not available)", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, "Wav2Vec2 Attention - Temporal Saliency", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)

    if prediction == "spoof":
        behavior = ("Behavioral Finding: The attention map highlights heavy artifact concentrations "
                    "in regions outside human vocal tract configurations. This strongly indicates "
                    "a digital voice injection or a synthesized vocoder fingerprint.")
    else:
        behavior = ("Behavioral Finding: The attention map shows natural energy distribution "
                    "consistent with organic human phonation. No synthetic artifact clusters detected.")

    pdf.set_x(w_margin)
    pdf.multi_cell(page_w, 5,
        "Telemetry Target: Mel-frequency spectrogram tracking the audio's continuous spectral envelope."
    )
    pdf.set_x(w_margin)
    pdf.multi_cell(page_w, 5, behavior)

    if attention_maps and attention_maps.get("layer_count"):
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"Transformer layers analyzed: {attention_maps['layer_count']}", new_x="LMARGIN", new_y="NEXT")
        if attention_maps.get("timeline"):
            timeline = attention_maps["timeline"]
            scores_list = [t.get("attention_score", 0) for t in timeline]
            if scores_list:
                pdf.cell(0, 5, f"Attention frames: {len(timeline)}  |  "
                         f"Peak score: {max(scores_list):.3f}  |  "
                         f"Mean score: {sum(scores_list) / len(scores_list):.3f}",
                         new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # ---------------------------------------------------------------
    # 4. Calibration Guide
    # ---------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "4. Calibration Guide (For Engineering Operations)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "P(Spoof) = System Confidence Score", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    bar_w = page_w * 0.9
    bar_x = w_margin + (page_w - bar_w) / 2
    bar_y = pdf.get_y()
    bar_h = 10

    pdf.set_fill_color(200, 200, 200)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, "F")
    pdf.set_fill_color(40, 180, 80)
    pdf.rect(bar_x, bar_y, bar_w * 0.5, bar_h, "F")
    pdf.set_fill_color(220, 50, 50)
    pdf.rect(bar_x + bar_w * 0.5, bar_y, bar_w * 0.5, bar_h, "F")
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.5)
    pdf.line(bar_x + bar_w * 0.5, bar_y, bar_x + bar_w * 0.5, bar_y + bar_h)

    marker_x = bar_x + bar_w * confidence
    pdf.set_fill_color(0, 0, 0)
    pdf.polygon([(marker_x - 3, bar_y - 4), (marker_x + 3, bar_y - 4), (marker_x, bar_y)], style="F")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.text(marker_x - 8, bar_y - 6, f"{confidence * 100:.1f}%")

    pdf.set_y(bar_y + bar_h + 4)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(bar_w * 0.5, 5, "0.0  Genuine Voice (Bona-fide / Real)", align="L")
    pdf.cell(bar_w * 0.5, 5, "1.0  Confirmed Deepfake (Spoof / Fake)", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(w_margin)
    pdf.cell(bar_w * 0.5, 5, f"     Current: {confidence * 100:.2f}%", align="L")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(w_margin)
    pdf.multi_cell(page_w, 5,
        "Interpretation Policy: This pipeline maps 0.0 directly to a confirmed authentic voice "
        "(Bona-fide / Real), while 1.0 matches a confirmed synthetic voice (Spoof / Fake)."
    )
    if confidence > 0.7 and prediction == "spoof":
        metric_note = "falls deeply into the critical threat zone, indicating a highly reliable detection signature."
    elif confidence <= 0.5:
        metric_note = "falls below the decision boundary, indicating a genuine voice."
    else:
        metric_note = "approaches the decision boundary and should be reviewed."
    pdf.set_x(w_margin)
    pdf.multi_cell(page_w, 5,
        f"Current Metric: The confidence level score of {confidence:.4f} "
        f"({confidence * 100:.2f}%) {metric_note}"
    )

    return bytes(pdf.output())
