
import os
import io
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def test_vietnamese_pdf():
    buffer = io.BytesIO()
    font_name = "TimesNewRoman"
    font_path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "times.ttf")
    
    print(f"Checking for font at: {font_path}")
    if not os.path.exists(font_path):
        print("Font not found!")
        return
    
    pdfmetrics.registerFont(TTFont(font_name, font_path))
    
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = font_name
    
    elements = []
    text = "BÁO CÁO KẾT QUẢ KIỂM ĐỊNH CẤU HÌNH - Tiếng Việt có dấu"
    elements.append(Paragraph(text, styles["Normal"]))
    
    data = [
        ["HẠNG MỤC KIỂM TRA", "KẾT QUẢ"],
        ["Kiểm tra cài đặt bản vá", "ĐẠT"]
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), font_name),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))
    elements.append(table)
    
    doc.build(elements)
    print("PDF generated successfully with Vietnamese text.")

if __name__ == "__main__":
    test_vietnamese_pdf()
