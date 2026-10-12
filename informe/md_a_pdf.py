"""Convierte informe/informe.md en informe/informe.pdf con ReportLab.

Soporta el subconjunto de Markdown usado en el informe: titulos, parrafos,
negrita/cursiva/codigo inline, listas numeradas y con guion, tablas con pipes,
bloques de codigo, imagenes y separadores.
"""
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                ListFlowable, ListItem, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(BASE, "informe.md")
PDF = os.path.join(BASE, "informe.pdf")

AZUL = colors.HexColor("#1F3864")
AZUL_CLARO = colors.HexColor("#2E5C9A")
GRIS = colors.HexColor("#444444")
GRIS_SUAVE = colors.HexColor("#EEF2F7")

ss = getSampleStyleSheet()
E = {
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15.5, leading=19, textColor=AZUL,
                         spaceBefore=13, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=12, leading=15, textColor=AZUL_CLARO,
                         spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                         fontSize=10.3, leading=13, textColor=GRIS,
                         spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", parent=ss["BodyText"], fontName="Helvetica",
                        fontSize=9.2, leading=12.8, alignment=TA_JUSTIFY,
                        spaceAfter=6, textColor=colors.HexColor("#1A1A1A")),
    "li": ParagraphStyle("li", parent=ss["BodyText"], fontName="Helvetica",
                         fontSize=9.2, leading=12.4, alignment=TA_JUSTIFY,
                         spaceAfter=2.5),
    "code": ParagraphStyle("code", parent=ss["Code"], fontName="Courier",
                           fontSize=7.8, leading=10, textColor=colors.HexColor("#102A43"),
                           backColor=GRIS_SUAVE, borderPadding=6,
                           spaceBefore=5, spaceAfter=9),
    "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontName="Helvetica-Oblique",
                          fontSize=8.2, leading=11, alignment=TA_CENTER,
                          textColor=GRIS, spaceBefore=3, spaceAfter=11),
    "th": ParagraphStyle("th", parent=ss["BodyText"], fontName="Helvetica-Bold",
                         fontSize=8.2, leading=10.5, textColor=colors.white),
    "td": ParagraphStyle("td", parent=ss["BodyText"], fontName="Helvetica",
                         fontSize=8.2, leading=10.5),
    "portada_t": ParagraphStyle("pt", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=25, leading=30, textColor=AZUL,
                                alignment=TA_CENTER, spaceAfter=10),
    "portada_s": ParagraphStyle("ps", parent=ss["Title"], fontName="Helvetica",
                                fontSize=12.5, leading=18, textColor=GRIS,
                                alignment=TA_CENTER, spaceAfter=22),
    "portada_m": ParagraphStyle("pm", parent=ss["BodyText"], fontName="Helvetica",
                                fontSize=10.5, leading=17, alignment=TA_CENTER,
                                textColor=colors.HexColor("#1A1A1A")),
}


# Helvetica/Courier (WinAnsi) no tienen estos glifos: se reemplazan antes de dibujar.
SUSTITUCIONES = {
    "⁷": "7", "⁻": "-", "⁴": "4", "¹²": "12", "≈": "~", "→": "->", "←": "<-",
    "✓": "SI", "✗": "NO", "≤": "<=", "≥": ">=", "−": "-",
}


def inline(t):
    """Traduce marcas inline de Markdown a etiquetas de ReportLab."""
    for a, b in SUSTITUCIONES.items():
        t = t.replace(a, b)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"!\[.*?\]\(.*?\)", "", t)
    t = re.sub(r"\[(.+?)\]\((.+?)\)", r'<link href="\2" color="#2E5C9A">\1</link>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.6" color="#0B3D63">\1</font>', t)
    t = t.replace("✓", "SI").replace("✗", "NO")
    return t


def tabla(filas, ancho_total):
    """Arma una Table de ReportLab a partir de filas de Markdown ya separadas."""
    cab, cuerpo = filas[0], filas[1:]
    ncol = max(len(f) for f in filas)
    norm = [f + [""] * (ncol - len(f)) for f in [cab] + cuerpo]
    data = [[Paragraph(inline(c), E["th"]) for c in norm[0]]]
    data += [[Paragraph(inline(c), E["td"]) for c in f] for f in norm[1:]]
    cw = ancho_total / ncol
    t = Table(data, colWidths=[cw] * ncol, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C6D8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS_SUAVE]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def construir(lineas, ancho):
    fl = []
    i = 0
    buf_li, tipo_li = [], None

    def cerrar_lista():
        nonlocal buf_li, tipo_li
        if buf_li:
            fl.append(ListFlowable(
                [ListItem(Paragraph(x, E["li"]), leftIndent=12) for x in buf_li],
                bulletType="1" if tipo_li == "ol" else "bullet",
                bulletFontSize=8.5, leftIndent=16, spaceAfter=8))
            buf_li, tipo_li = [], None

    while i < len(lineas):
        ln = lineas[i].rstrip("\n")
        s = ln.strip()

        if s.startswith("```"):
            cerrar_lista()
            i += 1
            code = []
            while i < len(lineas) and not lineas[i].strip().startswith("```"):
                code.append(lineas[i].rstrip("\n"))
                i += 1
            i += 1
            txt = "\n".join(code)
            for a, b in SUSTITUCIONES.items():
                txt = txt.replace(a, b)
            fl.append(Preformatted(txt, E["code"]))
            continue

        m = re.match(r"^!\[(.*?)\]\((.+?)\)", s)
        if m:
            cerrar_lista()
            cap, rel = m.group(1), m.group(2)
            ruta = os.path.join(BASE, rel.replace("/", os.sep))
            if os.path.exists(ruta):
                from reportlab.lib.utils import ImageReader
                iw, ih = ImageReader(ruta).getSize()
                w = min(ancho, 14.2 * cm)
                h = w * ih / iw
                if h > 8.8 * cm:
                    h = 8.8 * cm
                    w = h * iw / ih
                grupo = [Image(ruta, width=w, height=h, hAlign="CENTER")]
                if cap:
                    grupo.append(Paragraph(f"Figura. {inline(cap)}", E["cap"]))
                fl.append(KeepTogether(grupo))
            i += 1
            continue

        if s.startswith("|") and s.endswith("|"):
            cerrar_lista()
            filas = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                raw = lineas[i].strip()
                if not re.match(r"^\|[\s:\-\|]+\|$", raw):
                    filas.append([c.strip() for c in raw.strip("|").split("|")])
                i += 1
            if filas:
                fl.append(tabla(filas, ancho))
                fl.append(Spacer(1, 9))
            continue

        if s in ("---", "***", "___"):
            cerrar_lista()
            fl.append(Spacer(1, 5))
            fl.append(Table([[""]], colWidths=[ancho], rowHeights=[0.7],
                            style=TableStyle([("BACKGROUND", (0, 0), (-1, -1),
                                               colors.HexColor("#C9D4E4"))])))
            fl.append(Spacer(1, 9))
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            cerrar_lista()
            n, txt = len(m.group(1)), m.group(2)
            fl.append(Paragraph(inline(txt), E["h1"] if n <= 1 else E["h2"] if n == 2 else E["h3"]))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            if tipo_li == "ul":
                cerrar_lista()
            tipo_li = "ol"
            buf_li.append(inline(m.group(2)))
            i += 1
            continue

        m = re.match(r"^[-*+]\s+(.*)$", s)
        if m:
            if tipo_li == "ol":
                cerrar_lista()
            tipo_li = "ul"
            buf_li.append(inline(m.group(1)))
            i += 1
            continue

        if s.startswith(">"):
            cerrar_lista()
            fl.append(Paragraph(inline(s.lstrip("> ").strip()),
                                ParagraphStyle("q", parent=E["p"], leftIndent=14,
                                               borderPadding=5, backColor=GRIS_SUAVE,
                                               fontName="Helvetica-Oblique")))
            i += 1
            continue

        if not s:
            cerrar_lista()
            i += 1
            continue

        cerrar_lista()
        parr = [s]
        i += 1
        while i < len(lineas):
            nx = lineas[i].strip()
            if (not nx or nx.startswith(("#", "|", "-", "*", ">", "```", "!["))
                    or re.match(r"^\d+\.\s", nx)):
                break
            parr.append(nx)
            i += 1
        fl.append(Paragraph(inline(" ".join(parr)), E["p"]))

    cerrar_lista()
    return fl


def main():
    with open(MD, encoding="utf-8") as fh:
        texto = fh.read()

    # separa la portada (hasta el primer ---) del cuerpo
    partes = texto.split("\n---\n", 1)
    cuerpo_md = partes[1] if len(partes) > 1 else texto

    doc = BaseDocTemplate(PDF, pagesize=A4,
                          leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                          topMargin=2.0 * cm, bottomMargin=1.9 * cm,
                          title="Master of the Harmonica - TP Final",
                          author="V. M. Danguise")
    ancho = doc.width

    def pie(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRIS)
        canvas.drawString(d.leftMargin, 1.15 * cm,
                          "Master of the Harmonica  ·  TP Final  ·  Redes Neuronales y Deep Learning  ·  UP")
        canvas.drawRightString(d.leftMargin + d.width, 1.15 * cm, f"{canvas.getPageNumber()}")
        canvas.setStrokeColor(colors.HexColor("#C9D4E4"))
        canvas.setLineWidth(0.4)
        canvas.line(d.leftMargin, 1.45 * cm, d.leftMargin + d.width, 1.45 * cm)
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="todo", frames=[frame], onPage=pie)])

    fl = [Spacer(1, 3.4 * cm),
          Paragraph("Master of the Harmonica", E["portada_t"]),
          Paragraph("Clasificación de tonalidad musical en 24 clases con CNN 1D sobre audio "
                    "crudo, y derivación determinística de armónica blues en segunda posición",
                    E["portada_s"]),
          Spacer(1, 0.9 * cm),
          Table([[""]], colWidths=[6 * cm], rowHeights=[1.1],
                hAlign="CENTER",
                style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), AZUL)])),
          Spacer(1, 1.0 * cm),
          Paragraph("<b>Trabajo Práctico Final</b>", E["portada_m"]),
          Paragraph("Redes Neuronales y Deep Learning", E["portada_m"]),
          Paragraph("Maestría en Inteligencia Artificial", E["portada_m"]),
          Paragraph("Universidad de Palermo", E["portada_m"]),
          Spacer(1, 1.0 * cm),
          Paragraph("<b>Autor:</b> V. M. Danguise", E["portada_m"]),
          Paragraph('<link href="https://github.com/vmdanguise/master-of-the-harmonica" '
                    'color="#2E5C9A">github.com/vmdanguise/master-of-the-harmonica</link>',
                    E["portada_m"]),
          Paragraph("Octubre de 2026", E["portada_m"]),
          PageBreak()]

    fl += construir(cuerpo_md.splitlines(), ancho)
    doc.build(fl)
    print("PDF ->", PDF)


if __name__ == "__main__":
    main()
