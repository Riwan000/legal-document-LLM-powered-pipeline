"""
Report builders extracted from frontend/app.py (st-free, pure data->output).

Previously nested closures inside the contract-review page function; lifted to
module level so they can be unit-tested and to reduce the page function's
cognitive complexity (SonarQube S3776). Behavior is unchanged — these are a
verbatim move. Imported by app.py as a sibling module (``import report_builders``)
which resolves because Streamlit runs frontend/app.py with its own dir on sys.path.
"""
import requests  # used by _translate_texts (module-level, as in the original)

# Mandatory disclaimer for workflow outputs (single source of truth; imported by app.py)
WORKFLOW_DISCLAIMER = "This system does not provide legal advice. All outputs are for review purposes only."


# ── Helper: Format short review summary for Agent 1 chat bubble ──────────
def _format_review_summary(resp: dict) -> str:
    lines = []
    ct = resp.get("contract_type") or "contract"
    jur = resp.get("jurisdiction") or "—"
    lines.append(f"**Contract Review Complete** — Type: `{ct}` | Jurisdiction: `{jur}`\n")

    exec_items = resp.get("executive_summary", []) or []
    risks_exec = [i for i in exec_items if isinstance(i, dict) and i.get("category") == "risk"][:5]
    findings = [i for i in exec_items if isinstance(i, dict) and i.get("category") == "finding"][:3]

    if risks_exec:
        lines.append("**Key Risks:**")
        for item in risks_exec:
            sev = item.get("severity", "")
            txt = item.get("text", "")
            lines.append(f"- **{sev}**: {txt}" if sev else f"- {txt}")

    not_detected = resp.get("not_detected_clauses", []) or []
    if not_detected:
        lines.append(f"\n**Clauses Not Detected:** {', '.join(not_detected[:5])}" +
                     (" …and more" if len(not_detected) > 5 else ""))

    if findings:
        lines.append("\n**Findings:**")
        for item in findings:
            lines.append(f"- {item.get('text', '')}")

    lines.append(f"\n*{resp.get('disclaimer', WORKFLOW_DISCLAIMER)}*")
    return "\n".join(lines)

# ── Helper: Generate PDF report with reportlab ────────────────────────────
def _generate_pdf_buffer(resp: dict):
    """Build PDF in-memory; return BytesIO. Raises on failure."""
    from io import BytesIO
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError(f"reportlab not installed: {exc}") from exc

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["Normal"]
    story = []

    # ── 8a: Reusable cell paragraph styles ────────────────────────────────
    cell_style = ParagraphStyle(
        "cell", parent=normal,
        fontSize=7, leading=9,
        wordWrap="LTR",
        spaceAfter=0, spaceBefore=0,
    )
    header_cell_style = ParagraphStyle(
        "header_cell", parent=cell_style,
        textColor=colors.whitesmoke, fontName="Helvetica-Bold",
    )
    toc_style = ParagraphStyle(
        "toc", parent=normal, fontSize=9, leftIndent=12, spaceAfter=2,
    )

    # Title page
    story.append(Paragraph("Contract Review Report", h1))
    story.append(Spacer(1, 0.4*cm))
    ct = resp.get("contract_type") or "—"
    jur = resp.get("jurisdiction") or "—"
    doc_id = resp.get("document_id") or "—"
    risk_label = resp.get("risk_label", "low_risk")
    risk_score = resp.get("risk_score", 0)
    story.append(Paragraph(f"Contract type: {ct} | Jurisdiction: {jur}", normal))
    story.append(Paragraph(f"Document: {doc_id}", normal))
    story.append(Paragraph(f"Overall Risk: {risk_label.replace('_', ' ').title()} (score: {risk_score})", normal))
    story.append(Spacer(1, 0.4*cm))

    # ── 8h: Table of contents ─────────────────────────────────────────────
    statutory_notes = resp.get("statutory_notes") or {}
    exec_items = resp.get("executive_summary", []) or []
    risks = resp.get("risks", []) or []
    toc_items = ["Executive Summary", "Clause Coverage", "Risk Analysis", "Evidence Excerpts", "Clauses Not Detected"]
    if statutory_notes:
        toc_items.append("Statutory References")
    for toc_name in toc_items:
        story.append(Paragraph(f"• {toc_name}", toc_style))
    story.append(Spacer(1, 0.6*cm))

    # Executive Summary
    if exec_items:
        story.append(Paragraph("Executive Summary", h2))
        by_cat: dict = {}
        for item in exec_items:
            if isinstance(item, dict):
                by_cat.setdefault(item.get("category") or "other", []).append(item)
        for section_title, cat_key in [("Risks", "risk"), ("Findings", "finding"), ("Confirmations", "confirmation")]:
            group = by_cat.get(cat_key, [])
            if group:
                story.append(Paragraph(f"<b>{section_title}</b>", normal))
                for item in group:
                    sev = item.get("severity", "")
                    txt = item.get("text", "")
                    bullet = f"• [{sev}] {txt}" if sev else f"• {txt}"
                    story.append(Paragraph(bullet, normal))
        story.append(Spacer(1, 0.4*cm))

    # ── 8f: Clause Coverage Table ─────────────────────────────────────────
    if exec_items:
        story.append(Paragraph("Clause Coverage", h2))
        coverage_data = [[
            Paragraph("Clause", header_cell_style),
            Paragraph("Status", header_cell_style),
            Paragraph("Severity", header_cell_style),
        ]]
        for item in exec_items:
            if not isinstance(item, dict):
                continue
            cat = item.get("category", "")
            icon = "Confirmed" if cat == "confirmation" else ("Finding" if cat == "finding" else "Risk")
            sev_text = (item.get("severity") or "").capitalize() or "—"
            coverage_data.append([
                Paragraph(item.get("text", ""), cell_style),
                Paragraph(icon, cell_style),
                Paragraph(sev_text, cell_style),
            ])
        cov_tbl = Table(coverage_data, repeatRows=1, colWidths=[9.5*cm, 4.5*cm, 3.0*cm])
        cov_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(cov_tbl)
        story.append(Spacer(1, 0.4*cm))

    # ── 8b/8c/8d: Risk Analysis Table with Paragraph cells + row colors ──
    if risks:
        story.append(Paragraph("Risk Analysis", h2))
        # ── 8b: fixed 17cm total ──────────────────────────────────────────
        COL_WIDTHS = [1.5*cm, 5.5*cm, 3.0*cm, 2.5*cm, 1.0*cm, 3.5*cm]
        table_data = [[
            Paragraph("Severity", header_cell_style),
            Paragraph("Description", header_cell_style),
            Paragraph("Status", header_cell_style),
            Paragraph("Clauses", header_cell_style),
            Paragraph("Pages", header_cell_style),
            Paragraph("Recommendation", header_cell_style),
        ]]
        # ── 8c: wrap all cells in Paragraph — no truncation ───────────────
        for r in risks:
            if not isinstance(r, dict):
                continue
            display_names = r.get("display_names", []) or r.get("clause_ids", []) or []
            pages = r.get("page_numbers", []) or []
            table_data.append([
                Paragraph(r.get("severity", "").capitalize(), cell_style),
                Paragraph(r.get("description", ""), cell_style),
                Paragraph(r.get("status", "").replace("_", " ").title(), cell_style),
                Paragraph(", ".join(display_names[:3]), cell_style),
                Paragraph(", ".join(str(p) for p in pages[:5]), cell_style),
                Paragraph(r.get("recommendation") or "", cell_style),
            ])
        tbl = Table(table_data, repeatRows=1, colWidths=COL_WIDTHS)
        # ── 8d: color-code rows by severity ──────────────────────────────
        row_colors = []
        for i, r in enumerate(risks, start=1):
            if not isinstance(r, dict):
                continue
            bg = (
                colors.HexColor("#FFD5D5") if r.get("severity") == "high" else
                colors.HexColor("#FFF3CD") if r.get("severity") == "medium" else
                colors.HexColor("#D4EDDA")
            )
            row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            *row_colors,
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

        # ── 8e: Evidence Excerpts with split header/body styles ────────────
        has_snippets = any(r.get("verbatim_evidence") for r in risks if isinstance(r, dict))
        if has_snippets:
            story.append(Paragraph("Evidence Excerpts", h2))
            ev_header_style = ParagraphStyle(
                "ev_h", parent=normal,
                fontSize=7, fontName="Helvetica-Bold", leftIndent=12, spaceAfter=1,
            )
            ev_body_style = ParagraphStyle(
                "ev_b", parent=normal,
                fontSize=7, leftIndent=12, leading=9,
                textColor=colors.HexColor("#1A3C6B"), spaceAfter=4,
            )
            for r in risks:
                if not isinstance(r, dict):
                    continue
                snippets = r.get("verbatim_evidence", []) or []
                if not snippets:
                    continue
                story.append(Paragraph(f"<b>{r.get('description', '')}</b>", normal))
                if r.get("severity_reason"):
                    story.append(Paragraph(f"<i>{r['severity_reason']}</i>", normal))
                for snippet in snippets:
                    s_name = snippet.get("display_name") or snippet.get("clause_id", "")
                    s_page = snippet.get("page_number", "")
                    s_text = snippet.get("text", "")
                    story.append(Paragraph(f"{s_name} — Page {s_page}", ev_header_style))
                    story.append(Paragraph(f'"{s_text}"', ev_body_style))
                story.append(Spacer(1, 0.2*cm))

    # Not Detected
    not_detected = resp.get("not_detected_clauses", []) or []
    if not_detected:
        story.append(Paragraph("Clauses Not Detected", h2))
        for name in not_detected:
            story.append(Paragraph(f"• {name}", normal))
        story.append(Spacer(1, 0.3*cm))

    # ── 8g: Statutory References ──────────────────────────────────────────
    if statutory_notes:
        story.append(Paragraph("Statutory References", h2))
        story.append(Paragraph(
            "<i>[Reference only — not legal advice. Verify with qualified counsel.]</i>", normal
        ))
        stat_data = [[
            Paragraph("Clause", header_cell_style),
            Paragraph("Reference", header_cell_style),
        ]]
        for clause_display, note in statutory_notes.items():
            if not isinstance(note, dict):
                continue
            art = note.get("article", "")
            text_val = note.get("text", "")
            src = note.get("source", "")
            stat_data.append([
                Paragraph(str(clause_display), cell_style),
                Paragraph(f"{art}: {text_val} — {src}", cell_style),
            ])
        stat_tbl = Table(stat_data, repeatRows=1, colWidths=[4.5*cm, 12.5*cm])
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 0.3*cm))

    # Disclaimer
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph(resp.get("disclaimer", WORKFLOW_DISCLAIMER), ParagraphStyle("small", parent=normal, fontSize=8)))

    doc.build(story)
    buf.seek(0)
    return buf

# ── Arabic PDF helpers ─────────────────────────────────────────────────────
def _ar(text: str) -> str:
    """Reshape + bidi-correct Arabic text for ReportLab rendering."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except ImportError:
        return str(text)

def _translate_texts(texts: list, api_base: str) -> list:
    """Call backend /api/translate; returns translations in same order."""
    import json as _json
    try:
        r = requests.post(
            f"{api_base}/api/translate",
            data={"texts": _json.dumps(texts), "target_lang": "ar", "source_lang": "en"},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("translations", texts)
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning(f"Arabic translation failed, falling back to source text: {_e}")
        return texts  # fallback: untranslated

_AR = {
    "title": "تقرير مراجعة العقد",
    "contract_type": "نوع العقد",
    "jurisdiction": "الولاية القضائية",
    "document": "المستند",
    "overall_risk": "المخاطر الإجمالية",
    "score": "النتيجة",
    "exec_summary": "الملخص التنفيذي",
    "clause_coverage": "تغطية البنود",
    "risk_analysis": "تحليل المخاطر",
    "evidence_excerpts": "مقتطفات الأدلة",
    "not_detected": "البنود غير المكتشفة",
    "statutory_refs": "المراجع القانونية",
    "risks_cat": "المخاطر",
    "findings_cat": "النتائج",
    "confirmations_cat": "التأكيدات",
    "confirmed": "مؤكد",
    "finding": "نتيجة",
    "risk_status": "خطر",
    "clause": "البند",
    "status": "الحالة",
    "severity": "الخطورة",
    "description": "الوصف",
    "clauses": "البنود",
    "pages": "الصفحات",
    "recommendation": "التوصية",
    "reference": "المرجع",
    "high_risk": "مخاطرة عالية",
    "medium_risk": "مخاطرة متوسطة",
    "low_risk": "مخاطرة منخفضة",
    "high": "مرتفع",
    "medium": "متوسط",
    "low": "منخفض",
    "ref_disclaimer": "[للمرجعية فقط — لا تمثل استشارة قانونية. يُرجى التحقق مع مستشار قانوني مؤهل.]",
    "disclaimer": "لا يقدم هذا النظام استشارات قانونية. جميع المخرجات لأغراض المراجعة فقط.",
    "page_label": "صفحة",
    "toc_items": ["الملخص التنفيذي", "تغطية البنود", "تحليل المخاطر", "مقتطفات الأدلة", "البنود غير المكتشفة", "المراجع القانونية"],
}

def _generate_arabic_pdf_buffer(resp: dict, api_base: str):
    """Build Arabic PDF in-memory; return BytesIO. Raises on failure."""
    import os
    from io import BytesIO
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_RIGHT
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as exc:
        raise RuntimeError(f"reportlab not installed: {exc}") from exc

    # Register Arabic-capable font
    font_path = r"C:\Windows\Fonts\arial.ttf"
    if os.path.exists(font_path):
        try:
            if "Arabic" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("Arabic", font_path))
            AR_FONT = "Arabic"
        except Exception:
            AR_FONT = "Helvetica"
    else:
        AR_FONT = "Helvetica"

    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    ar_h1 = ParagraphStyle(
        "ar_h1", parent=normal,
        fontName=AR_FONT, fontSize=16, leading=20,
        wordWrap="RTL", alignment=TA_RIGHT, spaceAfter=8,
    )
    ar_h2 = ParagraphStyle(
        "ar_h2", parent=normal,
        fontName=AR_FONT, fontSize=12, leading=15,
        wordWrap="RTL", alignment=TA_RIGHT, spaceAfter=4,
    )
    ar_normal = ParagraphStyle(
        "ar_normal", parent=normal,
        fontName=AR_FONT, fontSize=9, leading=12,
        wordWrap="RTL", alignment=TA_RIGHT,
    )
    ar_cell = ParagraphStyle(
        "ar_cell", parent=normal,
        fontName=AR_FONT, fontSize=7, leading=9,
        wordWrap="RTL", alignment=TA_RIGHT,
        spaceAfter=0, spaceBefore=0,
    )
    ar_header_cell = ParagraphStyle(
        "ar_header_cell", parent=ar_cell,
        textColor=colors.whitesmoke, fontName=AR_FONT,
    )
    ar_toc_style = ParagraphStyle(
        "ar_toc", parent=ar_normal, fontSize=9, spaceAfter=2,
    )
    ar_small = ParagraphStyle(
        "ar_small", parent=ar_normal, fontSize=8,
    )

    # ── Collect all dynamic strings to translate ──────────────────────────
    exec_items = resp.get("executive_summary", []) or []
    risks = resp.get("risks", []) or []
    statutory_notes = resp.get("statutory_notes") or {}
    not_detected = resp.get("not_detected_clauses", []) or []

    dyn_strings = []
    # 1: contract_type, jurisdiction
    dyn_strings.append(resp.get("contract_type") or "")
    dyn_strings.append(resp.get("jurisdiction") or "")
    # 2: executive_summary texts
    for item in exec_items:
        if isinstance(item, dict):
            dyn_strings.append(item.get("text", ""))
    # 3: risk descriptions, recommendations, severity_reason, display_names, snippets
    for r in risks:
        if not isinstance(r, dict):
            continue
        dyn_strings.append(r.get("description", ""))
        dyn_strings.append(r.get("recommendation") or "")
        dyn_strings.append(r.get("severity_reason") or "")
        for dn in (r.get("display_names") or []):
            dyn_strings.append(str(dn))
        for snippet in (r.get("verbatim_evidence") or []):
            dyn_strings.append(snippet.get("text", ""))
            dyn_strings.append(snippet.get("display_name") or snippet.get("clause_id", ""))
    # 4: not_detected_clauses
    for name in not_detected:
        dyn_strings.append(str(name))
    # 5: statutory_notes keys and note fields
    for clause_display, note in statutory_notes.items():
        dyn_strings.append(str(clause_display))
        if isinstance(note, dict):
            dyn_strings.append(note.get("text", ""))
            dyn_strings.append(note.get("article", ""))
            dyn_strings.append(note.get("source", ""))
    # 6: disclaimer
    dyn_strings.append(resp.get("disclaimer", WORKFLOW_DISCLAIMER) or "")

    # Translate all at once
    translated = _translate_texts(dyn_strings, api_base)

    # ── Reconstruct translated values ─────────────────────────────────────
    idx = 0
    def _t():
        nonlocal idx
        val = translated[idx] if idx < len(translated) else ""
        idx += 1
        return val or ""

    t_contract_type = _t()
    t_jurisdiction = _t()

    t_exec_items = []
    for item in exec_items:
        if isinstance(item, dict):
            t_exec_items.append({**item, "text": _t()})

    t_risks = []
    for r in risks:
        if not isinstance(r, dict):
            continue
        t_desc = _t()
        t_rec = _t()
        t_sev_reason = _t()
        t_display_names = [_t() for _ in (r.get("display_names") or [])]
        t_snippets = []
        for snippet in (r.get("verbatim_evidence") or []):
            t_text = _t()
            t_dname = _t()
            t_snippets.append({**snippet, "text": t_text,
                                "display_name": t_dname or snippet.get("clause_id", "")})
        t_risks.append({**r, "description": t_desc, "recommendation": t_rec,
                        "severity_reason": t_sev_reason,
                        "display_names": t_display_names,
                        "verbatim_evidence": t_snippets})

    t_not_detected = [_t() for _ in not_detected]

    t_statutory = {}
    for clause_display, note in statutory_notes.items():
        t_clause = _t()
        if isinstance(note, dict):
            t_text_val = _t()
            t_art = _t()
            t_src = _t()
            t_statutory[t_clause] = {**note, "text": t_text_val, "article": t_art, "source": t_src}
        else:
            t_statutory[t_clause] = note

    t_disclaimer = _t()

    # ── Build PDF ─────────────────────────────────────────────────────────
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []

    # Title page
    story.append(Paragraph(_ar(_AR["title"]), ar_h1))
    story.append(Spacer(1, 0.4*cm))
    risk_label = resp.get("risk_label", "low_risk")
    risk_score = resp.get("risk_score", 0)
    doc_id = resp.get("document_id") or "—"
    risk_label_ar = _AR.get(risk_label, risk_label.replace("_", " "))
    story.append(Paragraph(
        _ar(f"{_AR['contract_type']}: {t_contract_type or '—'} | {_AR['jurisdiction']}: {t_jurisdiction or '—'}"),
        ar_normal,
    ))
    story.append(Paragraph(_ar(f"{_AR['document']}: {doc_id}"), ar_normal))
    story.append(Paragraph(
        _ar(f"{_AR['overall_risk']}: {risk_label_ar} ({_AR['score']}: {risk_score})"),
        ar_normal,
    ))
    story.append(Spacer(1, 0.4*cm))

    # Table of contents
    toc_labels = list(_AR["toc_items"])
    if not t_statutory:
        toc_labels = toc_labels[:5]
    for toc_name in toc_labels:
        story.append(Paragraph(_ar(f"• {toc_name}"), ar_toc_style))
    story.append(Spacer(1, 0.6*cm))

    # Executive Summary
    if t_exec_items:
        story.append(Paragraph(_ar(_AR["exec_summary"]), ar_h2))
        by_cat: dict = {}
        for item in t_exec_items:
            if isinstance(item, dict):
                by_cat.setdefault(item.get("category") or "other", []).append(item)
        for section_title_key, cat_key in [
            ("risks_cat", "risk"), ("findings_cat", "finding"), ("confirmations_cat", "confirmation")
        ]:
            group = by_cat.get(cat_key, [])
            if group:
                story.append(Paragraph(f"<b>{_ar(_AR[section_title_key])}</b>", ar_normal))
                for item in group:
                    sev = item.get("severity", "")
                    txt = item.get("text", "")
                    bullet = f"• [{sev}] {txt}" if sev else f"• {txt}"
                    story.append(Paragraph(_ar(bullet), ar_normal))
        story.append(Spacer(1, 0.4*cm))

    # Clause Coverage Table
    if t_exec_items:
        story.append(Paragraph(_ar(_AR["clause_coverage"]), ar_h2))
        coverage_data = [[
            Paragraph(_ar(_AR["clause"]), ar_header_cell),
            Paragraph(_ar(_AR["status"]), ar_header_cell),
            Paragraph(_ar(_AR["severity"]), ar_header_cell),
        ]]
        for item in t_exec_items:
            if not isinstance(item, dict):
                continue
            cat = item.get("category", "")
            icon = _AR["confirmed"] if cat == "confirmation" else (_AR["finding"] if cat == "finding" else _AR["risk_status"])
            sev_raw = (item.get("severity") or "").lower()
            sev_text = _AR.get(sev_raw, sev_raw.capitalize()) or "—"
            coverage_data.append([
                Paragraph(_ar(item.get("text", "")), ar_cell),
                Paragraph(_ar(icon), ar_cell),
                Paragraph(_ar(sev_text), ar_cell),
            ])
        cov_tbl = Table(coverage_data, repeatRows=1, colWidths=[9.5*cm, 4.5*cm, 3.0*cm])
        cov_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(cov_tbl)
        story.append(Spacer(1, 0.4*cm))

    # Risk Analysis Table
    if t_risks:
        story.append(Paragraph(_ar(_AR["risk_analysis"]), ar_h2))
        COL_WIDTHS = [1.5*cm, 5.5*cm, 3.0*cm, 2.5*cm, 1.0*cm, 3.5*cm]
        table_data = [[
            Paragraph(_ar(_AR["severity"]), ar_header_cell),
            Paragraph(_ar(_AR["description"]), ar_header_cell),
            Paragraph(_ar(_AR["status"]), ar_header_cell),
            Paragraph(_ar(_AR["clauses"]), ar_header_cell),
            Paragraph(_ar(_AR["pages"]), ar_header_cell),
            Paragraph(_ar(_AR["recommendation"]), ar_header_cell),
        ]]
        _status_ar = {
            "risk": _AR["risk_status"],
            "finding": _AR["finding"],
            "confirmation": _AR["confirmed"],
        }
        for r in t_risks:
            if not isinstance(r, dict):
                continue
            display_names = r.get("display_names") or []
            pages = r.get("page_numbers", []) or []
            sev = r.get("severity", "")
            sev_ar = _AR.get(sev, sev).capitalize()
            table_data.append([
                Paragraph(_ar(sev_ar), ar_cell),
                Paragraph(_ar(r.get("description", "")), ar_cell),
                Paragraph(_ar(_status_ar.get(r.get("status", "").lower(), r.get("status", ""))), ar_cell),
                Paragraph(_ar(", ".join(str(d) for d in display_names[:3])), ar_cell),
                Paragraph(_ar(", ".join(str(p) for p in pages[:5])), ar_cell),
                Paragraph(_ar(r.get("recommendation") or ""), ar_cell),
            ])
        tbl = Table(table_data, repeatRows=1, colWidths=COL_WIDTHS)
        row_colors = []
        for i, r in enumerate(t_risks, start=1):
            if not isinstance(r, dict):
                continue
            bg = (
                colors.HexColor("#FFD5D5") if r.get("severity") == "high" else
                colors.HexColor("#FFF3CD") if r.get("severity") == "medium" else
                colors.HexColor("#D4EDDA")
            )
            row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("FONTNAME", (0, 0), (-1, 0), AR_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            *row_colors,
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

        # Evidence Excerpts
        has_snippets = any(r.get("verbatim_evidence") for r in t_risks if isinstance(r, dict))
        if has_snippets:
            story.append(Paragraph(_ar(_AR["evidence_excerpts"]), ar_h2))
            ar_ev_header = ParagraphStyle(
                "ar_ev_h", parent=ar_normal,
                fontSize=7, spaceAfter=1,
            )
            ar_ev_body = ParagraphStyle(
                "ar_ev_b", parent=ar_normal,
                fontSize=7, leading=9,
                textColor=colors.HexColor("#1A3C6B"), spaceAfter=4,
            )
            for r in t_risks:
                if not isinstance(r, dict):
                    continue
                snippets = r.get("verbatim_evidence", []) or []
                if not snippets:
                    continue
                story.append(Paragraph(f"<b>{_ar(r.get('description', ''))}</b>", ar_normal))
                if r.get("severity_reason"):
                    story.append(Paragraph(f"<i>{_ar(r['severity_reason'])}</i>", ar_normal))
                for snippet in snippets:
                    s_name = snippet.get("display_name") or snippet.get("clause_id", "")
                    s_page = snippet.get("page_number", "")
                    s_text = snippet.get("text", "")
                    story.append(Paragraph(
                        _ar(f"{s_name} — {_AR['page_label']} {s_page}"), ar_ev_header
                    ))
                    story.append(Paragraph(_ar(f'"{s_text}"'), ar_ev_body))
                story.append(Spacer(1, 0.2*cm))

    # Clauses Not Detected
    if t_not_detected:
        story.append(Paragraph(_ar(_AR["not_detected"]), ar_h2))
        for name in t_not_detected:
            story.append(Paragraph(_ar(f"• {name}"), ar_normal))
        story.append(Spacer(1, 0.3*cm))

    # Statutory References
    if t_statutory:
        story.append(Paragraph(_ar(_AR["statutory_refs"]), ar_h2))
        story.append(Paragraph(_ar(_AR["ref_disclaimer"]), ar_normal))
        stat_data = [[
            Paragraph(_ar(_AR["clause"]), ar_header_cell),
            Paragraph(_ar(_AR["reference"]), ar_header_cell),
        ]]
        for clause_display, note in t_statutory.items():
            if not isinstance(note, dict):
                continue
            art = note.get("article", "")
            text_val = note.get("text", "")
            src = note.get("source", "")
            stat_data.append([
                Paragraph(_ar(str(clause_display)), ar_cell),
                Paragraph(_ar(f"{art}: {text_val} — {src}"), ar_cell),
            ])
        stat_tbl = Table(stat_data, repeatRows=1, colWidths=[4.5*cm, 12.5*cm])
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 0.3*cm))

    # Disclaimer
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph(_ar(t_disclaimer or _AR["disclaimer"]), ar_small))

    doc.build(story)
    buf.seek(0)
    return buf
