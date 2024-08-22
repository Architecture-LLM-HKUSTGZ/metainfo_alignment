import fitz
import os
import io
from PIL import Image
import pytesseract
import pymupdf4llm
import pathlib


pdf_path = 'original_data/ms_pdf/1.13.2 Method Statement for General Site Survey Works.pdf'
md_output = 'results/test/test_alignment/1.13.2 Method Statement for General Site Survey Works.md'

md_text = pymupdf4llm.to_markdown(pdf_path)

pathlib.Path(md_output).write_bytes(md_text.encode('utf-8'))
