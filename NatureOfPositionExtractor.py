#!/usr/bin/env python3
"""
Extract Career Field Alignment / Nature of Position stats:
  - Directly Aligned, Stepping Stone, Pays the Bills, N

Default:
  - Reads PDFs in the SAME FOLDER as this script (non-recursive).
  - Writes nature_of_position.csv in the SAME FOLDER.

Usage (optional):
  python NatureOfPositionExtractor.py
  python NatureOfPositionExtractor.py --pdf_dir "C:/path/to/pdfs"
  python NatureOfPositionExtractor.py --out "C:/path/to/out.csv"
  python NatureOfPositionExtractor.py --debug_sample 2

Dependencies:
  pip install pdfplumber pandas
"""

import os
import re
import argparse
from typing import Optional, List, Dict, Tuple
import pdfplumber
import pandas as pd

# ---------- Config: unit names & normalization ----------
KNOWN_UNITS = [
    "University-wide", "University-Wide",
    "College of Agriculture and Natural Resources",
    "College of Arts and Humanities",
    "College of Behavioral and Social Sciences",
    "College of Computer, Mathematical, and Natural Sciences",
    "College of Education",
    "College of Information Studies", "College of Information",
    "The A. James Clark School of Engineering", "A. James Clark School of Engineering",
    "Philip Merrill College of Journalism", "Phillip Merrill College of Journalism",
    "School of Architecture, Planning, and Preservation",
    "School of Public Health",
    "School of Public Policy",
    "The Robert H. Smith School of Business", "Robert H. Smith School of Business",
    "College Park Scholars",
    "Honors College",
    "Letters and Sciences", "Letters & Sciences",
    "Undergraduate Studies", "Office of Undergraduate Studies", "Undergraduate Studies (UGST)",
]
UNIT_NORMALIZATION = {
    "University-Wide": "University-wide",
    "Phillip Merrill College of Journalism": "Philip Merrill College of Journalism",
    "A. James Clark School of Engineering": "The A. James Clark School of Engineering",
    "Robert H. Smith School of Business": "The Robert H. Smith School of Business",
    "Office of Undergraduate Studies": "Undergraduate Studies",
    "Undergraduate Studies (UGST)": "Undergraduate Studies",
    "College of Information Studies": "College of Information",
}

# Section heading variants seen across years (mainly for N extraction)
HEADING_VARIANTS = [
    "NATURE OF POSITION",
    "CAREER FIELD ALIGNMENT",
    "ALIGNMENT OF POSITION",
    "ALIGNMENT WITH CAREER GOALS",
    "NATURE OF EMPLOYMENT",
    "CAREER GOAL ALIGNMENT",
]

def normalize_unit(u: str) -> str:
    u = u.strip()
    return UNIT_NORMALIZATION.get(u, u)

# ---------- Left-bottom label patterns (robust to hyphenation/quotes) ----------
LBL_DIRECT_LEFT = re.compile(
    r"\bEmployment\s+is\s+direct[\W_]*ly\s+aligned\s+with\s+(?:their\s+)?career\s+goals\b", re.I
)
LBL_STEP_LEFT = re.compile(
    r"\bEmployment\s+is\s+a\s+stepp[\W_]*ing\s+stone\s+toward\s+(?:their\s+)?ultimate\s+career\s+goals\b", re.I
)
LBL_PAYS_LEFT = re.compile(
    r"\bPosition\s+simply\s+['\"“”]?pays\s+the\s+bills['\"“”]?\b", re.I
)

# ---------- Narrative / generic regex patterns (for ≤2019 fallbacks) ----------
PAT_DIRECT_INLINE = re.compile(r"\bdirect[\W_]*ly\s+aligned\b[^()%]*\((\d{1,3})%\)", re.I)
PAT_STEP_INLINE   = re.compile(r"\bstepp[\W_]*ing\s+stone\b[^()%]*\((\d{1,3})%\)", re.I)
PAT_DIRECT_BULLET = re.compile(r"\bdirect[\W_]*ly\s*aligned\b[^0-9%]{0,60}(\d{1,3})\s*%", re.I)
PAT_STEP_BULLET   = re.compile(r"\bstepp[\W_]*ing\s*stone\b[^0-9%]{0,60}(\d{1,3})\s*%", re.I)
LBL_PAYS_GENERIC  = re.compile(r"\bpays\s+the\s+bills\b", re.I)
PCT               = re.compile(r"(\d{1,3})\s*%")

# ---- N patterns (various phrasings) ----
PAT_NS = [
    re.compile(r"Based on (?:the\s+)?([0-9,]+)\s+(?:graduates?|respondents?|students?)\s+who\s+(?:completed|answered)\s+(?:the\s+)?(?:entire\s+)?(?:employment|career)\s+outcomes?\s+(?:section|questions?)\s*(?:of\s+the\s+survey)?", re.I),
    re.compile(r"Based on (?:the\s+)?([0-9,]+)\s+(?:graduates?|respondents?|students?)\b", re.I),
    re.compile(r"Among (?:the\s+)?([0-9,]+)\s+(?:graduates?|respondents?|students?)\b", re.I),
    re.compile(r"(?:These|The) (?:findings|results|data)\s+are\s+based\s+on\s+([0-9,]+)\s+(?:graduates?|respondents?|students?)\b", re.I),
    re.compile(r"[\(\[\{]?\s*[Nn]\s*=\s*([0-9,]+)\s*[\)\]\}]?", re.I),
    re.compile(r"Of\s+the\s+([0-9,]+)\s+graduates?\s+who\s+(?:completed|answered)\b", re.I),
]

def normalize_text(s: str) -> str:
    return (s or "").replace("–", "-").replace("—", "-").replace("", "•") \
                    .replace("“", '"').replace("”", '"').replace("’", "'")

def infer_year_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None

def find_units_on_page(text: str) -> List[str]:
    found = set()
    tlower = (text or "").lower()
    for u in KNOWN_UNITS:
        if u.lower() in tlower:
            found.add(u)
    return sorted(found, key=len, reverse=True)

def extract_n_from_text(chunk: str) -> Optional[int]:
    t = normalize_text(chunk)
    for pat in PAT_NS:
        m = pat.search(t)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except Exception:
                pass
    # light fallback window
    head_pos = 0
    for h in HEADING_VARIANTS:
        p = t.upper().find(h)
        if p != -1:
            head_pos = p
            break
    window = t[head_pos: head_pos + 1200]
    m2 = re.search(r"([0-9,]{1,6})\s+(?:graduates?|respondents?|students?)\b", window, re.I)
    if m2:
        try:
            return int(m2.group(1).replace(",", ""))
        except Exception:
            return None
    return None

# ---------- Layout helpers (2020+): read left-bottom graphic only ----------
def group_words_into_lines(words: List[dict], y_tol: float = 4.5) -> List[List[dict]]:
    """Group words by approximate y-center to form lines."""
    lines: List[List[dict]] = []
    for w in sorted(words, key=lambda x: (((x["top"]+x["bottom"])/2), x["x0"])):
        y = (w["top"] + w["bottom"]) / 2
        if not lines:
            lines.append([w]); continue
        y_last = (lines[-1][0]["top"] + lines[-1][0]["bottom"]) / 2
        if abs(y - y_last) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda x: x["x0"])
    return lines

def line_text(ln: List[dict]) -> str:
    return " ".join(w["text"] for w in ln)

def extract_left_block_percentages(page) -> Dict[str, Optional[int]]:
    """
    Extract values from LEFT-BOTTOM block using geometry:
      - consider only words in the left half of the page (x0 < 0.5 * width)
      - consider only words below 35% of the page height (below the heading zone)
      - for each left label, grab the % on the same line
    """
    width, height = page.width, page.height
    left_cut = 0.5 * width
    min_y = 0.35 * height  # anything below heading area

    words = [
        w for w in page.extract_words(x_tolerance=2, y_tolerance=2)
        if (w["x0"] < left_cut and ((w["top"] + w["bottom"]) / 2) > min_y)
    ]
    lines = group_words_into_lines(words)

    result = {"Directly Aligned": None, "Stepping Stone": None, "Pays the Bills": None}
    for ln in lines:
        tnorm = normalize_text(line_text(ln))
        # % on this line
        percents = [int(m.group(1)) for m in re.finditer(r"(\d{1,3})\s*%", tnorm)]
        if not percents:
            continue
        if LBL_DIRECT_LEFT.search(tnorm):
            result["Directly Aligned"] = percents[-1]
        elif LBL_STEP_LEFT.search(tnorm):
            result["Stepping Stone"] = percents[-1]
        elif LBL_PAYS_LEFT.search(tnorm):
            result["Pays the Bills"] = percents[-1]

    # derive pays if missing but D & S present
    d, s, p = result["Directly Aligned"], result["Stepping Stone"], result["Pays the Bills"]
    if p is None and (d is not None) and (s is not None):
        rem = 100 - d - s
        if 0 <= rem <= 100:
            result["Pays the Bills"] = rem

    return result

# ---------- ≤2019 extraction (text-based) ----------
def extract_pre2020_from_text(chunk: str) -> Dict[str, Optional[int]]:
    t = normalize_text(chunk)
    out = {"Directly Aligned": None, "Stepping Stone": None, "Pays the Bills": None}
    m = PAT_DIRECT_INLINE.search(t) or PAT_DIRECT_BULLET.search(t)
    if m: out["Directly Aligned"] = int(m.group(1))
    m = PAT_STEP_INLINE.search(t) or PAT_STEP_BULLET.search(t)
    if m: out["Stepping Stone"] = int(m.group(1))
    # anchor 'pays the bills' in same sentence/bullet
    sentences = re.split(r"(?<=[\.\?\!])\s+|\n+", t)
    for s in sentences:
        if LBL_PAYS_GENERIC.search(s):
            cands = [int(x) for x in re.findall(r"(\d{1,3})\s*%", s)]
            if cands:
                out["Pays the Bills"] = cands[-1]; break
    d, s, p = out["Directly Aligned"], out["Stepping Stone"], out["Pays the Bills"]
    if p is None and d is not None and s is not None:
        rem = 100 - d - s
        if 0 <= rem <= 100:
            out["Pays the Bills"] = rem
    return out

def list_pdfs_in_dir(dir_path: str) -> List[str]:
    pdfs: List[str] = []
    for f in os.listdir(dir_path):
        if f.lower().endswith(".pdf"):
            pdfs.append(os.path.join(dir_path, f))
    pdfs.sort()
    return pdfs

# ---------- Main ----------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default=script_dir,
                    help=f"Directory containing the PDF reports (default: script folder: {script_dir})")
    ap.add_argument("--out", default=os.path.join(script_dir, "nature_of_position.csv"),
                    help="Output CSV path (default: nature_of_position.csv in script folder)")
    ap.add_argument("--debug_sample", type=int, default=0,
                    help="Optional: only parse first N PDFs for debugging")
    args = ap.parse_args()

    pdf_paths = list_pdfs_in_dir(args.pdf_dir)
    if args.debug_sample > 0:
        pdf_paths = pdf_paths[: args.debug_sample]

    if not pdf_paths:
        print(f"[INFO] No PDFs found in: {args.pdf_dir}")
        return

    records = []

    for pdf_path in pdf_paths:
        pdf_name_only = os.path.basename(pdf_path)
        year = infer_year_from_filename(pdf_name_only)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_unit = None
                num_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    # always read the plain text (for unit + N and for ≤2019 fallbacks)
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""

                    # track the most recent unit text seen
                    units_here = find_units_on_page(page_text)
                    if units_here:
                        current_unit = UNIT_NORMALIZATION.get(units_here[0], units_here[0])

                    # -------- 2020+ : geometry-only scan on EVERY page ----------
                    if year and year >= 2020:
                        vals = extract_left_block_percentages(page)
                        got = sum(v is not None for v in vals.values())
                        if got >= 2:
                            # N: pull from this page + next page text (narrative sits right above the block)
                            chunk_for_n = page_text
                            if i + 1 < num_pages:
                                try:
                                    chunk_for_n += "\n" + (pdf.pages[i + 1].extract_text() or "")
                                except Exception:
                                    pass
                            n_val = extract_n_from_text(chunk_for_n)

                            records.append({
                                "Unit": current_unit or "Unknown",
                                "Year": year,
                                "Directly Aligned": vals["Directly Aligned"],
                                "Stepping Stone": vals["Stepping Stone"],
                                "Pays the Bills": vals["Pays the Bills"],
                                "N": n_val,
                            })
                    # -------- ≤2019 : use text detection, then fallbacks ----------
                    else:
                        # try to detect alignment-ish content on this page to limit parsing
                        if any(k in page_text.lower() for k in ("pays the bills", "directly aligned", "stepping stone")) \
                           or any(h in (page_text.upper()) for h in HEADING_VARIANTS):
                            # include 1–2 next pages to catch wrap
                            chunk = page_text
                            for j in (1, 2):
                                if i + j < num_pages:
                                    try:
                                        chunk += "\n" + (pdf.pages[i + j].extract_text() or "")
                                    except Exception:
                                        pass
                            vals = extract_pre2020_from_text(chunk)
                            got = sum(v is not None for v in vals.values())
                            if got >= 2:
                                n_val = extract_n_from_text(chunk)
                                records.append({
                                    "Unit": current_unit or "Unknown",
                                    "Year": year,
                                    "Directly Aligned": vals["Directly Aligned"],
                                    "Stepping Stone": vals["Stepping Stone"],
                                    "Pays the Bills": vals["Pays the Bills"],
                                    "N": n_val,
                                })
        except Exception as e:
            print(f"[WARN] Failed to parse {pdf_name_only}: {e}")

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["Unit"] = df["Unit"].apply(lambda x: UNIT_NORMALIZATION.get(x, x) if isinstance(x, str) else x)
        # keep the first hit per (Year, Unit)
        df = df.drop_duplicates(subset=["Year", "Unit"], keep="first").sort_values(["Year", "Unit"])
        for col in ["Directly Aligned", "Stepping Stone", "Pays the Bills", "N"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df = df.reset_index(drop=True)

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
