# extract_career_alignment_numbers.py
# Purpose: Find "Nature of Position" text in a PDF

from pathlib import Path
import pdfplumber

# path to your folder with PDFs
PDF_DIR = Path(r"C:\Users\sousman\Python\UMD_GradReports")

# choose one PDF to test
PDF_FILE = PDF_DIR / "2016 Graduation Survey Report Final Web Version.pdf"

with pdfplumber.open(PDF_FILE) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if "NATURE OF POSITION" in text.upper():
            print(f"\n--- Page {i} ---")
            print(text)
            print("="*80)
