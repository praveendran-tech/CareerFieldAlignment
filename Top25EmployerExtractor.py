#!/usr/bin/env python3
"""
Extract the University-wide "Top 25 Employers" (or similar) from UMD Graduation Survey PDFs.

- Scans all PDFs in a given folder (non-recursive by default).
- Looks for section headings like "Top 25 Employers", "Top Employers",
  or similar phrases in the UNIVERSITY-WIDE section and extracts up to 25 employer names.
- Writes a CSV with columns: Year, Employer_Rank, Employer.

USAGE
-----
  pip install pdfplumber pandas
  python Extract_Top25_Employers_UMD.py --pdf_dir "/path/to/pdfs" --out top_employers.csv

Tested with UMD Graduation Survey reports (2015–2024). The parser uses multiple
heuristics to handle heading/text variations and two-column layouts. If a year has
fewer than 25 items listed, it will capture whatever is available.

Notes
-----
- The script first tries line-based parsing. If that yields < 10 employers, it
  falls back to word-level clustering by y-position and column splitting.
- You can use --debug to print what the parser is seeing and why it keeps/filters lines.
- If multiple candidate lists are found in a PDF, the longest one is kept.

Author: ChatGPT (GPT-5 Thinking)
"""

import os
import re
import argparse
from typing import List, Tuple, Dict, Optional

import pdfplumber
import pandas as pd

# Resolve the folder where THIS script lives (fallback to CWD when __file__ is unavailable)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# ------------------------ Config ------------------------
HEADING_PATTERNS = [
    r"\btop\s*25\s*employers\b",
    r"\btop\s*employers\b",
    r"\btop\s*hiring\s*employers\b",
    r"\bemployers?\s+of\s+bachelor",
    r"\bmajor\s*employers\b",
]

# Text that likely indicates we moved out of the list
STOP_PATTERNS = [
    r"\buniversity[- ]wide\b",
    r"\bcontinuing education\b",
    r"\baverage\b",
    r"\bnature of position\b",
    r"\bplacement rate\b",
    r"\bcareer outcomes?\b",
    r"\bcollege of\b",
    r"\bschool of\b",
    r"\bappendix\b",
]

# Lines that are clearly not employer names
NOISE_PATTERNS = [
    r"^top\b.*employers?\b",                 # the heading itself
    r"^university[- ]wide$",
    r"^page\s*\d+",
    r"^figure\s*\d+",
    r"^table\s*\d+",
    r"^201[0-9]|^202[0-9]",                    # isolated year on a line
    r"^\s*$",
]

BULLET_TOKENS = ["•", "·", "-", "–", "—", "•\t"]

EMPLOYER_MIN_CHARS = 2  # after cleaning tokens

# ------------------------ Helpers ------------------------

def guess_year_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", fname)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def normalize(s: str) -> str:
    s = s.replace("\u00A0", " ")  # non-breaking space
    s = s.strip()
    # remove trailing page headers like "...University-wide" sometimes glued
    return re.sub(r"\s+", " ", s)


def is_heading(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in HEADING_PATTERNS)


def looks_like_stop(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in STOP_PATTERNS)


def looks_like_noise(line: str) -> bool:
    t = line.strip().lower()
    if any(re.search(p, t) for p in NOISE_PATTERNS):
        return True
    # discard lines that are just numbers or ranks
    if re.fullmatch(r"\d+\.?", t):
        return True
    # discard single-character bullets
    if t in {"-", "–", "—", "•", "·"}:
        return True
    return False


def clean_employer_token(line: str) -> Optional[str]:
    # remove bullets and leading ranks like "1.", "12)", "15 -"
    original = line
    line = line.strip()
    for b in BULLET_TOKENS:
        line = line.replace(b, " ")
    line = re.sub(r"^\s*(?:\d+\s*[\).\-–—:]\s*)", "", line)  # leading rank
    line = re.sub(r"\s{2,}", " ", line)
    line = line.strip("-–—:·•. ")

    # drop trailing rank fragments (rare)
    line = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", line)

    # obvious non-employer lines
    if not line or looks_like_noise(line):
        return None

    # over-aggressive all-caps single word that's not a company? allow anyway
    if len(line) < EMPLOYER_MIN_CHARS:
        return None

    return line


def extract_lines_after_heading(text: str, debug: bool=False) -> List[str]:
    lines = [normalize(x) for x in (text or "").splitlines()]

    # find first heading line index
    start_idx = None
    for i, ln in enumerate(lines):
        if is_heading(ln):
            start_idx = i
            break
    if start_idx is None:
        return []

    collected: List[str] = []
    for ln in lines[start_idx+1:]:
        if looks_like_stop(ln):
            break
        token = clean_employer_token(ln)
        if token:
            collected.append(token)
    if debug:
        print(f"[line-parse] collected {len(collected)} tokens after heading")
    return collected


def cluster_words_by_line(words: List[Dict], y_tol: float=3.0) -> List[List[Dict]]:
    """Cluster word dicts into visual lines by their 'top' position."""
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[List[Dict]] = []
    for w in words_sorted:
        if not lines:
            lines.append([w])
            continue
        last_line = lines[-1]
        if abs(w["top"] - last_line[0]["top"]) <= y_tol:
            last_line.append(w)
        else:
            lines.append([w])
    # sort within each line by x0
    for line in lines:
        line.sort(key=lambda w: w["x0"])  # left-to-right
    return lines


def split_lines_into_columns(lines: List[List[Dict]]) -> Tuple[List[str], List[str]]:
    """Split into two columns by median x center heuristic."""
    xs = [ (w["x0"] + w["x1"]) / 2 for line in lines for w in line ]
    if not xs:
        return [], []
    mid = sorted(xs)[len(xs)//2]
    left_lines: List[str] = []
    right_lines: List[str] = []
    for line in lines:
        # dominant side for this visual line by avg center
        centers = [ (w["x0"] + w["x1"]) / 2 for w in line ]
        avgc = sum(centers) / len(centers)
        text = " ".join(w["text"] for w in line)
        if avgc <= mid:
            left_lines.append(text)
        else:
            right_lines.append(text)
    return left_lines, right_lines


def extract_words_fallback(page: pdfplumber.page.Page, debug: bool=False) -> List[str]:
    text = page.extract_text() or ""
    if not any(re.search(p, text.lower()) for p in HEADING_PATTERNS):
        return []

    words = page.extract_words(use_text_flow=True, extra_attrs=["x0", "x1", "top", "bottom"]) or []
    if not words:
        return []

    # find approximate heading Y using the first matching word occurrence
    heading_y = None
    joined = " ".join(w["text"] for w in words).lower()
    if any(re.search(p, joined) for p in HEADING_PATTERNS):
        # find first word 'Top' near 'Employer'
        for i, w in enumerate(words):
            if re.search(r"^top$", w["text"].lower()):
                heading_y = w["bottom"]
                break
    if heading_y is None:
        heading_y = min(w["bottom"] for w in words) + 30  # arbitrary below title region

    # take words below heading
    payload = [w for w in words if w["top"] > heading_y]
    line_words = cluster_words_by_line(payload)
    left, right = split_lines_into_columns(line_words)
    candidates = left + right

    cleaned: List[str] = []
    for ln in candidates:
        tok = clean_employer_token(ln)
        if tok:
            cleaned.append(tok)
    if debug:
        print(f"[words-fallback] page {page.page_number}: got {len(cleaned)} tokens")
    return cleaned


def unique_ordered(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def parse_pdf_for_top_employers(pdf_path: str, debug: bool=False) -> List[str]:
    best_list: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 1) quick line-based parse
            text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5)
            if not text:
                continue
            lines_list = extract_lines_after_heading(text, debug=debug)
            # Stop early if the next heading or blank ends; try to cap to reasonable size
            if len(lines_list) >= 10:
                # sometimes includes narrative lines; trim to likely employer-like lines
                trimmed = [x for x in lines_list if re.search(r"[A-Za-z]", x)]
                trimmed = unique_ordered(trimmed)
                if len(trimmed) > len(best_list):
                    best_list = trimmed

            # 2) fallback using word-level clustering if we don't have enough
            if len(best_list) < 10:
                wlist = extract_words_fallback(page, debug=debug)
                if len(wlist) > len(best_list):
                    best_list = wlist

    # Final cleanup: stop when we hit a likely section spillover by keywords
    cleaned_final: List[str] = []
    for item in best_list:
        if looks_like_stop(item):
            break
        cleaned_final.append(item)

    # Only keep up to 25
    return unique_ordered(cleaned_final)[:25]


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract University-wide Top 25 Employers from UMD Graduation Survey PDFs")
    ap.add_argument("--pdf_dir", default=SCRIPT_DIR, help="Directory containing PDFs (non-recursive)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "top_employers_university_wide.csv"), help="Output CSV path")
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
        year = guess_year_from_filename(os.path.basename(pdf_path))
        if args.debug:
            print(f"Processing: {os.path.basename(pdf_path)} (year={year})")
        employers = parse_pdf_for_top_employers(pdf_path, debug=args.debug)
        if not employers and args.debug:
            print("  -> No employer list found")
        for rank, emp in enumerate(employers, start=1):
            rows.append({
                "Year": year,
                "Employer_Rank": rank,
                "Employer": emp,
                })

    if not rows:
        print("No employer data extracted.")
        return

    df_long = pd.DataFrame(rows, columns=["Year", "Employer_Rank", "Employer"])
    df_long = df_long.sort_values(["Year", "Employer_Rank"])

    # Pivot to one row per Year with Employer_1..Employer_25 columns
    out_rows = []
    MAX_COLS = 25
    for year, grp in df_long.groupby("Year", dropna=False):
        emps = grp.sort_values("Employer_Rank")["Employer"].tolist()
        emps = (emps + [""] * MAX_COLS)[:MAX_COLS]
        row = {"Year": int(year) if pd.notnull(year) else None}
        for i in range(MAX_COLS):
            row[f"Employer_{i+1}"] = emps[i]
        out_rows.append(row)

    wide_df = pd.DataFrame(out_rows).sort_values("Year")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wide_df.to_csv(args.out, index=False)
    print(f"Wrote {len(wide_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
