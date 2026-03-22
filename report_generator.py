"""
report_generator.py — PDF report per patient using fpdf2
"""
import os
import io
import re
from datetime import datetime
from fpdf import FPDF, XPos, YPos
import plotly.io as pio
from PIL import Image

from config import REPORTS_DIR


# ══════════════════════════════════════════════════════════════
# Unicode sanitizer — strip/replace chars Helvetica can't render
# ══════════════════════════════════════════════════════════════

def _safe(text: str) -> str:
    """Replace common Unicode symbols with ASCII equivalents."""
    if not text:
        return ""
    replacements = {
        "\u2014": "-",   # em dash  —
        "\u2013": "-",   # en dash  –
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u00b0": " deg",# degree sign
        "\u00b1": "+/-", # plus-minus
        "\u00d7": "x",   # multiplication sign
        "\u00e9": "e",   # é
        "\u00e8": "e",   # è
        "\u00ea": "e",   # ê
        "\u00e0": "a",   # à
        "\u00e2": "a",   # â
        "\u00fc": "u",   # ü
        "\u00f6": "o",   # ö
        "\u00e4": "a",   # ä
        "\u2026": "...", # ellipsis
        "\u2264": "<=",  # ≤
        "\u2265": ">=",  # ≥
        "\u00a0": " ",   # non-breaking space
        "\u00ab": "<<",  # «
        "\u00bb": ">>",  # »
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Strip any remaining non-latin1 characters
    text = text.encode("latin-1", errors="ignore").decode("latin-1")
    return text


class MammoReport(FPDF):

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
        self.set_font("Helvetica", size=11)

    # ── Header ────────────────────────────────────────────────

    def header(self):
        self.set_fill_color(20, 20, 20)
        self.rect(0, 0, 210, 22, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(229, 9, 20)
        self.set_xy(10, 6)
        self.cell(0, 10, "MammoCAD - AI Mammogram Analysis Report", align="L")
        self.set_font("Helvetica", size=8)
        self.set_text_color(180, 180, 180)
        self.set_xy(-70, 8)
        self.cell(60, 8, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
                  align="R")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0, 10,
            "CONFIDENTIAL - For clinical use only. "
            "AI predictions must be reviewed by a qualified radiologist.",
            align="C"
        )

    # ── Section helpers ────────────────────────────────────────

    def section_title(self, text: str):
        self.set_fill_color(40, 40, 40)
        self.set_text_color(229, 9, 20)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, f"  {_safe(text)}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def info_row(self, label: str, value: str, color_value: bool = False):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(180, 180, 180)
        self.cell(55, 7, _safe(label) + ":")
        self.set_font("Helvetica", size=10)
        val = _safe(str(value))
        if color_value and val.lower() == "malignant":
            self.set_text_color(229, 9, 20)
        elif color_value and val.lower() == "benign":
            self.set_text_color(70, 211, 105)
        else:
            self.set_text_color(255, 255, 255)
        self.cell(0, 7, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def add_plotly_image(self, fig, width: int = 180, height: int = 90):
        """Render plotly fig to PNG bytes and embed."""
        try:
            img_bytes = pio.to_image(fig, format="png",
                                     width=width * 4, height=height * 4, scale=1)
            buf = io.BytesIO(img_bytes)
            x = (210 - width) / 2
            self.image(buf, x=x, w=width, h=height)
            self.ln(3)
        except Exception as e:
            self.set_text_color(120, 120, 120)
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 7, f"[Chart could not be rendered: {_safe(str(e))}]",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def safe_multi_cell(self, w, h, text: str):
        """multi_cell with Unicode sanitization."""
        self.multi_cell(w, h, _safe(text))

    def divider(self):
        self.set_draw_color(60, 60, 60)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def generate_report(patient: dict, analysis: dict, figs: dict = None) -> str:
    pdf = MammoReport()
    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 0, 210, 297, "F")

    # ── Patient information ───────────────────────────────────
    pdf.section_title("Patient Information")
    pdf.info_row("Patient ID",   patient.get("patient_id", "N/A"))
    pdf.info_row("Name",         patient.get("full_name",  "N/A"))
    pdf.info_row("Age",          str(patient.get("age",    "N/A")))
    pdf.info_row("Contact",      patient.get("contact",    "N/A"))
    history = patient.get("history", "")
    if history:
        pdf.info_row("Clinical History", str(history)[:120])
    pdf.ln(4)
    pdf.divider()

    # ── Analysis results ──────────────────────────────────────
    pdf.section_title("AI Analysis Results")
    pred = analysis.get("prediction", "N/A")
    pdf.info_row("Prediction",           pred, color_value=True)
    pdf.info_row("Benign Probability",
                 f"{analysis.get('benign_prob', 0) * 100:.1f}%")
    pdf.info_row("Malignant Probability",
                 f"{analysis.get('malignant_prob', 0) * 100:.1f}%")
    pdf.info_row("BI-RADS Category",     analysis.get("birads_category", "N/A"))
    # birads_desc often contains em-dashes — _safe() handles this
    pdf.info_row("BI-RADS Assessment",   analysis.get("birads_desc",     "N/A"))
    pdf.info_row("Analysed By",          analysis.get("analysed_by",     "N/A"))
    pdf.info_row("Analysis Date",        analysis.get("analysed_at",     "N/A"))
    notes = analysis.get("notes", "")
    if notes:
        pdf.info_row("Notes", str(notes)[:200])
    pdf.ln(4)
    pdf.divider()

    # ── Charts ────────────────────────────────────────────────
    if figs:
        pdf.section_title("Visualisations")
        for name, fig in figs.items():
            pdf.set_text_color(180, 180, 180)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 6, _safe(name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.add_plotly_image(fig, width=180, height=85)
        pdf.divider()

    # ── Feature table ─────────────────────────────────────────
    features = analysis.get("features", {})
    if features:
        pdf.section_title("Extracted Features (Normalized 0-1)")
        mean_keys = sorted([k for k in features if k.endswith("_mean")])
        pdf.set_font("Helvetica", size=9)
        col_w   = [70, 35, 70, 35]
        headers = ["Feature", "Value", "Feature", "Value"]
        pdf.set_fill_color(40, 40, 40)
        pdf.set_text_color(180, 180, 180)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, border=1, fill=True)
        pdf.ln()
        pairs = [(mean_keys[i], mean_keys[i + 1])
                 for i in range(0, len(mean_keys) - 1, 2)]
        if len(mean_keys) % 2 == 1:
            pairs.append((mean_keys[-1], None))
        for k1, k2 in pairs:
            pdf.set_text_color(220, 220, 220)
            label1 = k1.replace("_mean", "").replace("_", " ").title()
            pdf.cell(col_w[0], 6, label1, border=1)
            pdf.cell(col_w[1], 6, f"{features[k1]:.4f}", border=1)
            if k2:
                label2 = k2.replace("_mean", "").replace("_", " ").title()
                pdf.cell(col_w[2], 6, label2, border=1)
                pdf.cell(col_w[3], 6, f"{features[k2]:.4f}", border=1)
            else:
                pdf.cell(col_w[2] + col_w[3], 6, "", border=1)
            pdf.ln()
        pdf.ln(4)
        pdf.divider()

    # ── Disclaimer ────────────────────────────────────────────
    pdf.section_title("Important Notice")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(160, 160, 160)
    disclaimer = (
        "This report is generated by an AI-assisted Computer-Aided Diagnosis (CAD) system "
        "and is intended solely as a decision-support tool. It does not replace clinical "
        "judgment, radiological interpretation, or histopathological confirmation. "
        "All findings must be reviewed and validated by a qualified medical professional. "
        "Do not make clinical decisions based solely on this report."
    )
    pdf.safe_multi_cell(0, 5, disclaimer)

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = (
        f"report_{patient.get('patient_id', 'unknown')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    path = os.path.join(REPORTS_DIR, filename)
    pdf.output(path)
    return path