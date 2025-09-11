#!/usr/bin/env python3
"""
Extract the UNIVERSITY-WIDE "Top Schools" for Continuing Education from UMD Graduation Survey PDFs.

- Scans all PDFs in a given folder (non-recursive by default).
- Finds a section around Continuing Education listing top schools / graduate schools / institutions (University-wide)
  and extracts up to 25 names.
- Outputs ONE ROW PER YEAR with columns: Year, School_1..School_25

USAGE
-----
  pip install pdfplumber pandas
  python TopSchools_ContinuingEd_Extractor.py                # scans the script's folder by default
  python TopSchools_ContinuingEd_Extractor.py --pdf_dir .    # choose a folder
  python TopSchools_ContinuingEd_Extractor.py --recursive    # recurse into subfolders
  python TopSchools_ContinuingEd_Extractor.py --out schools.csv

Notes
-----
- The script uses multiple heuristics to handle wording changes:
  e.g., "Top Schools", "Top Graduate Schools", "Continuing Education: Top Schools",
        "Graduate Schools Attended", "Top Institutions", etc.
- It prefers pages that include both "University-wide" and "Continuing Education" text.
- If a year lists fewer than 25 schools, the remaining columns are left blank.

Author: ChatGPT (GPT-5 Thinking)
"""

import os
import re
import argparse
from typing import List, Tuple, Dict, Optional

import pdfplumber
import pandas as pd

# ------------------------ Defaults ------------------------
# Resolve the folder where THIS script lives (fallback to CWD when __file__ is unavailable)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# ------------------------ Config ------------------------
# Headings that likely begin the Top Schools list
SCHOOLS_HEADING_PATTERNS = [
    r"\btop\s*25\s*(schools|graduate\s*schools|institutions)\b",
    r"\btop\s*(schools|graduate\s*schools|institutions)\b",
    r"\bcontinuing\s*education\b.*(schools|institutions)",
    r"\bgraduate\s*(schools|programs)\s*(attended|enrolled|matriculated)\b",
    r"\btop\s*universities\b",
    r"\binstitutions\s*(attending|enrolled)\b",
]

# Text hint phrases to PREFER lists on the page
PREFERENCE_HINTS = [
    r"\buniversity[- ]wide\b",
    r"\bcontinuing\s*education\b",
    r"\bgraduate\s*(school|program|study|studies)\b",
]

# Lines that likely indicate the list ended or moved to another section
STOP_PATTERNS = [
    r"\buniversity[- ]wide\b",  # a new section header repeating can signal end
    r"\bemployment\b",
    r"\bemployers?\b",
    r"\bsalary\b",
    r"\bwage\b",
    r"\bnature of position\b",
    r"\bplacement rate\b",
    r"\bcareer outcomes?\b",
    r"\bcollege of\b",
    r"\bschool of\b",
    r"\bappendix\b",
]

# Lines that are noise, not school names
NOISE_PATTERNS = [
    r"^top\b.*(schools|institutions|universities)\b",  # the heading itself
    r"^university[- ]wide$",
    r"^page\s*\d+",
    r"^figure\s*\d+",
    r"^table\s*\d+",
    r"^201[0-9]|^202[0-9]",  # isolated year
    r"^\s*$",
    r"^employers?$",         # avoid mixing employer pages
]

BULLET_TOKENS = ["•", "·", "-", "–", "—", "•\t"]

SCHOOL_MIN_CHARS = 2
MAX_COLS = 25

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
    return re.sub(r"\s+", " ", s)


def is_schools_heading(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in SCHOOLS_HEADING_PATTERNS)


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
    if t in {"-", "–", "—", "•", "·"}:
        return True
    return False


def clean_school_token(line: str) -> Optional[str]:
    line = line.strip()
    for b in BULLET_TOKENS:
        line = line.replace(b, " ")
    line = re.sub(r"^\s*(?:\d+\s*[\).\-–—:]\s*)", "", line)  # leading rank markers
    line = re.sub(r"\s{2,}", " ", line).strip("-–—:·•. ")
    line = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", line)  # trailing rank in parens

    if not line or looks_like_noise(line):
        return None
    if len(line) < SCHOOL_MIN_CHARS:
        return None
    return line


def extract_lines_after_heading(text: str, debug: bool=False) -> List[str]:
    lines = [normalize(x) for x in (text or "").splitlines()]

    start_idx = None
    for i, ln in enumerate(lines):
        if is_schools_heading(ln):
            start_idx = i
            break
    if start_idx is None:
        return []

    collected: List[str] = []
    for ln in lines[start_idx+1:]:
        if looks_like_stop(ln):
            break
        token = clean_school_token(ln)
        if token:
            collected.append(token)
    if debug:
        print(f"[line-parse] collected {len(collected)} tokens after schools heading")
    return collected


def cluster_words_by_line(words: List[Dict], y_tol: float=3.0) -> List[List[Dict]]:
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
    for line in lines:
        line.sort(key=lambda w: w["x0"])  # left-to-right
    return lines


def split_lines_into_columns(lines: List[List[Dict]]) -> Tuple[List[str], List[str]]:
    xs = [ (w["x0"] + w["x1"]) / 2 for line in lines for w in line ]
    if not xs:
        return [], []
    mid = sorted(xs)[len(xs)//2]
    left_lines: List[str] = []
    right_lines: List[str] = []
    for line in lines:
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
    if not any(re.search(p, text.lower()) for p in SCHOOLS_HEADING_PATTERNS):
        return []

    words = page.extract_words(use_text_flow=True, extra_attrs=["x0", "x1", "top", "bottom"]) or []
    if not words:
        return []

    # try to find approximate heading Y using the first matching 'Top' near school terms
    heading_y = None
    joined = " ".join(w["text"] for w in words).lower()
    if any(re.search(p, joined) for p in SCHOOLS_HEADING_PATTERNS):
        for i, w in enumerate(words):
            if re.search(r"^top$", w["text"].lower()):
                heading_y = w["bottom"]
                break
    if heading_y is None:
        heading_y = min(w["bottom"] for w in words) + 30

    payload = [w for w in words if w["top"] > heading_y]
    line_words = cluster_words_by_line(payload)
    left, right = split_lines_into_columns(line_words)
    candidates = left + right

    cleaned: List[str] = []
    for ln in candidates:
        tok = clean_school_token(ln)
        if tok:
            cleaned.append(tok)
    if debug:
        print(f"[words-fallback] page {page.page_number}: got {len(cleaned)} tokens (schools)")
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


def score_page_for_schools(page_text: str, list_len: int) -> int:
    score = list_len
    t = (page_text or "").lower()
    for p in PREFERENCE_HINTS:
        if re.search(p, t):
            score += 5
    return score


def parse_pdf_for_top_schools(pdf_path: str, debug: bool=False) -> List[str]:
    best_list: List[str] = []
    best_score: int = -1

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            if not text:
                continue

            # 1) line-based parsing
            lines_list = extract_lines_after_heading(text, debug=debug)
            if lines_list:
                trimmed = [x for x in lines_list if re.search(r"[A-Za-z]", x)]
                trimmed = unique_ordered(trimmed)
                score = score_page_for_schools(text, len(trimmed))
                if debug:
                    print(f"[line-parse] page {page.page_number}: {len(trimmed)} schools, score {score}")
                if score > best_score:
                    best_list = trimmed
                    best_score = score

            # 2) fallback word-level parse
            if best_score < 15:  # still weak, try fallback
                wlist = extract_words_fallback(page, debug=debug)
                if wlist:
                    wlist = unique_ordered([x for x in wlist if re.search(r"[A-Za-z]", x)])
                    score = score_page_for_schools(text, len(wlist))
                    if debug:
                        print(f"[word-parse] page {page.page_number}: {len(wlist)} schools, score {score}")
                    if score > best_score:
                        best_list = wlist
                        best_score = score

    # Final cleanup and cap
    cleaned_final: List[str] = []
    for item in best_list:
        if looks_like_stop(item):
            break
        cleaned_final.append(item)

    return unique_ordered(cleaned_final)[:MAX_COLS]


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract University-wide Top Schools for Continuing Education from UMD Graduation Survey PDFs")
    ap.add_argument("--pdf_dir", default=SCRIPT_DIR, help="Directory containing PDFs (non-recursive)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "top_schools_continuing_ed_university_wide.csv"), help="Output CSV path")
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
        schools = parse_pdf_for_top_schools(pdf_path, debug=args.debug)
        if not schools and args.debug:
            print("  -> No school list found")
        for rank, school in enumerate(schools, start=1):
            rows.append({
                "Year": year,
                "School_Rank": rank,
                "School": school,
            })

    if not rows:
        print("No schools data extracted.")
        return

    # Long -> Wide
    df_long = pd.DataFrame(rows, columns=["Year", "School_Rank", "School"]).sort_values(["Year", "School_Rank"]) 

    out_rows = []
    for year, grp in df_long.groupby("Year", dropna=False):
        names = grp.sort_values("School_Rank")["School"].tolist()
        names = (names + [""] * MAX_COLS)[:MAX_COLS]
        row = {"Year": int(year) if pd.notnull(year) else None}
        for i in range(MAX_COLS):
            row[f"School_{i+1}"] = names[i]
        out_rows.append(row)

    wide_df = pd.DataFrame(out_rows).sort_values("Year")

    # Ensure output folder exists and write CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wide_df.to_csv(args.out, index=False)
    print(f"Wrote {len(wide_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
