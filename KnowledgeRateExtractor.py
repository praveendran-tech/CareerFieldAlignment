#!/usr/bin/env python3
"""
Extract University-wide data-collection / knowledge-rate stats from UMD Graduation Survey PDFs.

Looks for text like:
  "data from 5,242 of 6,688 ... resulting in a knowledge rate of 78%"

OUTPUT (wide):
  Unit, Year, Total Graduates, Data_Collected, Data_Collected_Rate

Behavior:
- Scans all PDFs in the script's directory by default (non-recursive).
- You can override with --pdf_dir and --recursive.
- Writes CSV next to the script by default.
- Picks the best match per PDF (prefers pages mentioning "University-wide").
- If the % isn't explicitly present, it computes it from counts.
- Infers Year using the graduating window on the page (e.g., "August 2014 ... May 2015").
  Fallback: first 4-digit year in the filename.

USAGE
-----
  pip install pdfplumber pandas
  python KnowledgeRate_UniversityWide_Extractor.py
  python KnowledgeRate_UniversityWide_Extractor.py --pdf_dir "." --out knowledge_rate_university_wide.csv --recursive

Author: ChatGPT (GPT-5 Thinking)
"""

import os
import re
import argparse
from typing import List, Dict, Tuple, Optional

import pdfplumber
import pandas as pd

# ------------------------ Defaults ------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# ------------------------ Regex Patterns ------------------------
# Core pattern: "data from X of Y ... knowledge rate (of|was) Z%"
RX_DATA_OF = re.compile(
    r"data\s+from\s+([\d,]+)\s+of\s+([\d,]+).*?knowledge\s*rate\s*(?:of|was)?\s*([\d\.]+)\s*(?:%|percent)?",
    re.IGNORECASE | re.DOTALL,
)
# Fallback when the % isn't present in the sentence
RX_DATA_OF_MIN = re.compile(
    r"data\s+from\s+([\d,]+)\s+of\s+([\d,]+)", re.IGNORECASE | re.DOTALL
)

# Year windows like "between August 2014 and May 2015" or "August 2014 to May 2015"
RX_AUG_MAY = re.compile(
    r"(?:between|from)?\s*August\s+(\d{4})\s*(?:to|and|\-|–|—)\s*May\s+(\d{4})",
    re.IGNORECASE,
)
# Class of NNNN
RX_CLASS_OF = re.compile(r"class\s+of\s+(\d{4})", re.IGNORECASE)
# Academic year like 2015-2016 or 2015–2016
RX_AY = re.compile(r"(20\d{2})\s*[\-–—]\s*(20\d{2})")

# Preference signal for choosing the best page candidate
RX_UNI_WIDE = re.compile(r"university[- ]wide", re.IGNORECASE)


# ------------------------ Helpers ------------------------

def to_int(x: str) -> Optional[int]:
    try:
        return int(x.replace(",", ""))
    except Exception:
        return None


def guess_year_from_text(text: str) -> Optional[int]:
    text = text or ""
    m = RX_AUG_MAY.search(text)
    if m:
        try:
            return int(m.group(2))  # May YYYY -> graduating year
        except Exception:
            pass
    m = RX_CLASS_OF.search(text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m = RX_AY.search(text)
    if m:
        try:
            y1, y2 = int(m.group(1)), int(m.group(2))
            return max(y1, y2)
        except Exception:
            pass
    return None


def guess_year_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", fname)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def best_candidate_from_text(text: str) -> Optional[Tuple[int, int, Optional[float]]]:
    """Return (collected, total, rate_pct) from a page of text, if found."""
    text = text or ""
    m = RX_DATA_OF.search(text)
    if m:
        collected, total, rate = to_int(m.group(1)), to_int(m.group(2)), m.group(3)
        rate_pct = float(rate) if rate else None
        if collected and total and collected <= total:
            return collected, total, rate_pct
    # fallback without rate
    m = RX_DATA_OF_MIN.search(text)
    if m:
        collected, total = to_int(m.group(1)), to_int(m.group(2))
        if collected and total and collected <= total:
            return collected, total, None
    return None


def parse_pdf(pdf_path: str, debug: bool=False) -> Optional[Dict]:
    """Return a row dict: {Unit, Year, Total Graduates, Data_Collected, Data_Collected_Rate}"""
    best: Optional[Tuple[int, int, Optional[float], int, Optional[int]]] = None
    # tuple plus page score & year: (collected, total, rate_pct, score, year_from_text)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            cand = best_candidate_from_text(text)
            if not cand:
                continue
            collected, total, rate_pct = cand
            year_text = guess_year_from_text(text)
            score = 0
            if RX_UNI_WIDE.search(text):
                score += 5
            # prefer pages that also mention "knowledge rate" explicitly
            if re.search(r"knowledge\s*rate", text, re.IGNORECASE):
                score += 3
            # prefer larger totals as a weak heuristic if multiple candidates
            score += min(total // 1000, 10)

            if debug:
                print(f"  page {page.page_number}: collected={collected}, total={total}, rate={rate_pct}, score={score}, year_hint={year_text}")

            if best is None or score > best[3]:
                best = (collected, total, rate_pct, score, year_text)

    if not best:
        return None

    collected, total, rate_pct, _, year_text = best
    if rate_pct is None:
        rate_pct = round(100.0 * collected / total, 2)

    year = year_text
    if year is None:
        year = guess_year_from_filename(os.path.basename(pdf_path))

    if year is None:
        # Last ditch: use file modified year (not great)
        try:
            year = int(pd.Timestamp(os.path.getmtime(pdf_path), unit="s").year)
        except Exception:
            year = None

    row = {
        "Unit": "University-wide",
        "Year": int(year) if year is not None else None,
        "Total Graduates": int(total),
        "Data_Collected": int(collected),
        "Data_Collected_Rate": f"{rate_pct:.2f}%",
    }
    return row


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract University-wide knowledge rate counts from UMD Graduation Survey PDFs")
    ap.add_argument("--pdf_dir", default=SCRIPT_DIR, help="Directory containing PDFs (non-recursive)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "knowledge_rate_university_wide.csv"), help="Output CSV path")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    pdf_paths: List[str] = []
    for root, dirs, files in os.walk(args.pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, fn))
        if not args.recursive:
            break

    if not pdf_paths:
        print(f"No PDFs found in {args.pdf_dir}")
        return

    rows: List[Dict] = []
    for pdf_path in sorted(pdf_paths):
        if args.debug:
            print(f"Processing: {os.path.basename(pdf_path)}")
        row = parse_pdf(pdf_path, debug=args.debug)
        if row:
            rows.append(row)
        elif args.debug:
            print("  -> No matching knowledge-rate sentence found")

    if not rows:
        print("No data extracted.")
        return

    # Deduplicate by Year, keeping the first occurrence (in case multiple PDFs per year)
    df = pd.DataFrame(rows, columns=["Unit", "Year", "Total Graduates", "Data_Collected", "Data_Collected_Rate"]).drop_duplicates(subset=["Year"]).sort_values("Year")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
