#!/usr/bin/env python3
"""
Extract hourly-wage stats from all PDFs in a folder (recursively) and output:

  Unit, Year, Internship_Wage_N, Internship_Wage_Median, Internship_Wage_Average

Looks for sentences like:
  "Of the 259 experiences that paid an hourly wage, the average reported income
   was $15.17 per hour and the median reported income was $13.50 per hour"

Usage:
  python extract_hourly_wage_stats.py --pdf_dir "./reports" --out "hourly_wage_stats.csv"

Deps:
  pip install pdfplumber PyPDF2 pandas
"""
import os
import re
import argparse
from typing import Dict, Tuple, Optional, List
import pandas as pd

# ----------------------- Known units (extend if needed) -----------------------
KNOWN_UNITS = [
    "University-wide", "University-Wide",
    "College of Agriculture and Natural Resources",
    "College of Arts and Humanities",
    "College of Behavioral and Social Sciences",
    "College of Computer, Mathematical, and Natural Sciences",
    "College of Education",
    "College of Information",
    "Honors College",
    "Letters and Sciences",
    "Philip Merrill College of Journalism",
    "School of Architecture, Planning, and Preservation",
    "School of Public Health",
    "School of Public Policy",
    "The A. James Clark School of Engineering",
    "The Robert H. Smith School of Business",
    "College Park Scholars",
    "Undergraduate Studies",
]
UNIT_PATTERNS = [(u, re.compile(re.escape(u), re.IGNORECASE)) for u in KNOWN_UNITS]

# ----------------------- Patterns to capture the stats ------------------------
NUM = r"([0-9][0-9,]*)"
MONEY = r"\$?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})|[0-9]+(?:\.[0-9]{1,2})?)"
WAGE_WORDS = r"(?:income|hourly\s*wage|wage|pay)"
ROLE_WORDS = r"(?:internship|internships|experience|experiences|position|positions|job|jobs|respondents)"

PATTERNS = [
    # Average then median
    re.compile(
        rf"Of\s+the\s+{NUM}\s+{ROLE_WORDS}\s+that\s+paid\s+an?\s+hourly\s+wage[, ]+"
        rf".*?average\s+(?:reported\s+)?{WAGE_WORDS}\s+was\s+{MONEY}\s+per\s+hour"
        rf".*?median\s+(?:reported\s+)?{WAGE_WORDS}\s+was\s+{MONEY}\s+per\s+hour",
        re.IGNORECASE | re.DOTALL,
    ),
    # Median then average
    re.compile(
        rf"Of\s+the\s+{NUM}\s+{ROLE_WORDS}\s+that\s+paid\s+an?\s+hourly\s+wage[, ]+"
        rf".*?median\s+(?:reported\s+)?{WAGE_WORDS}\s+was\s+{MONEY}\s+per\s+hour"
        rf".*?average\s+(?:reported\s+)?{WAGE_WORDS}\s+was\s+{MONEY}\s+per\s+hour",
        re.IGNORECASE | re.DOTALL,
    ),
    # Lenient fallback (either order on the same page)
    re.compile(
        rf"Of\s+the\s+{NUM}\s+{ROLE_WORDS}\s+that\s+paid\s+an?\s+hourly\s+wage.*?"
        rf"(?:average.*?{MONEY}.*?median.*?{MONEY}|median.*?{MONEY}.*?average.*?{MONEY})",
        re.IGNORECASE | re.DOTALL,
    ),
]

def try_import_pdf_libs():
    lib = None
    try:
        import pdfplumber  # noqa: F401
        lib = "pdfplumber"
    except Exception:
        pass
    if lib is None:
        try:
            import PyPDF2  # noqa: F401
            lib = "pypdf2"
        except Exception:
            pass
    if lib is None:
        raise RuntimeError("Neither pdfplumber nor PyPDF2 is installed. Please install one.")
    return lib

def read_pdf_pages(path: str, lib_name: str) -> List[str]:
    texts = []
    if lib_name == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
    else:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                texts.append(p.extract_text() or "")
    return texts

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def detect_unit(text: str, default_unit: str = "University-wide") -> str:
    for unit, rx in UNIT_PATTERNS:
        if rx.search(text):
            return unit
    return default_unit

def parse_year_from_filename(fname: str) -> Optional[int]:
    # Prefer a 4-digit year 2010–2099 from filename
    years = re.findall(r"(20[1-9][0-9])", fname)
    if years:
        try:
            return int(years[0])
        except ValueError:
            return None
    return None

def as_number(s: str) -> float:
    return float(s.replace(",", ""))

def extract_stats_from_text(text: str):
    """
    Returns (N, avg, median) if any pattern matches, else None.
    """
    for i, rx in enumerate(PATTERNS):
        m = rx.search(text)
        if not m:
            continue
        N = int(m.group(1).replace(",", ""))
        if i == 0:
            # avg then median
            avg = as_number(m.group(2)); med = as_number(m.group(3))
            return (N, avg, med)
        elif i == 1:
            # median then avg
            med = as_number(m.group(2)); avg = as_number(m.group(3))
            return (N, avg, med)
        # lenient case: infer order from keyword positions
        snippet = text[m.start():m.end()]
        money_vals = re.findall(MONEY, snippet, flags=re.IGNORECASE)
        if len(money_vals) >= 2:
            a_idx = snippet.lower().find("average")
            m_idx = snippet.lower().find("median")
            first = as_number(money_vals[0])
            second = as_number(money_vals[1])
            if a_idx != -1 and m_idx != -1 and a_idx < m_idx:
                return (N, first, second)
            elif a_idx != -1 and m_idx != -1 and m_idx < a_idx:
                return (N, second, first)
            return (N, first, second)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", help="Folder to scan (recursively). Defaults to CWD.")
    ap.add_argument("--out", default="hourly_wage_stats.csv", help="Output CSV path")
    args = ap.parse_args()

    pdf_dir = args.pdf_dir or os.getcwd()
    lib = try_import_pdf_libs()

    rows = []
    for root, _, files in os.walk(pdf_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(root, fname)

            try:
                pages = read_pdf_pages(full_path, lib)
            except Exception as e:
                print(f"!! Failed to read {os.path.relpath(full_path, pdf_dir)}: {e}")
                continue

            year = parse_year_from_filename(fname)
            current_unit = "University-wide"

            for raw in pages:
                if not raw:
                    continue
                text = normalize_spaces(raw)

                # Update unit context if a known unit appears on the page
                current_unit = detect_unit(text, default_unit=current_unit)

                hit = extract_stats_from_text(text)
                if hit:
                    N, avg, med = hit
                    rows.append({
                        "Unit": current_unit,
                        "Year": year,
                        "Internship_Wage_N": N,
                        "Internship_Wage_Median": round(med, 2),
                        "Internship_Wage_Average": round(avg, 2),
                    })

    # Deduplicate: keep the first hit seen for a (Unit, Year)
    dedup: Dict[Tuple[str, Optional[int]], Dict] = {}
    for r in rows:
        key = (r["Unit"], r["Year"])
        if key not in dedup:
            dedup[key] = r

    df = pd.DataFrame(dedup.values()).sort_values(
        ["Unit", "Year"], ascending=[True, True], na_position="last"
    )

    # Normalize unit casing to canonical
    canon = {u.lower(): u for u in KNOWN_UNITS}
    df["Unit"] = df["Unit"].apply(lambda x: canon.get(str(x).lower(), x))

    # Ensure column order and write
    df = df[["Unit", "Year", "Internship_Wage_N", "Internship_Wage_Median", "Internship_Wage_Average"]]
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
