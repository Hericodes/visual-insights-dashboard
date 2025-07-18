# utils/pdf_generator.py
import sys
sys.path.append(r"C:\Users\HP\AppData\Roaming\Python\Python313\site-packages")

from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.pdfgen import canvas # type: ignore
from io import BytesIO

def create_pdf(summary_text: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Format: break lines
    lines = summary_text.split("\n")
    y = height - 50  # starting y-position

    for line in lines:
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(40, y, line.strip())
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer
