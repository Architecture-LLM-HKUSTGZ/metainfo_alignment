import fitz
import os
import io
from PIL import Image
import pytesseract
import pymupdf4llm
import pathlib


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img)
    return text


def save_file(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w", encoding='utf-8') as file:
        file.write(content)


pdf_path = 'original_data/ms_pdf/1.13.41 (CS) Installation of Sheet Pile within 10m from nearest track.pdf'
documents = extract_text_from_pdf(pdf_path)
md_output = 'results/test/test_alignment/output_test.md'

save_file(documents, md_output)
# md_text = pymupdf4llm.to_markdown(pdf_path)
# pathlib.Path("results/test/test_alignment/output_test.md").write_bytes(md_text.encode())
