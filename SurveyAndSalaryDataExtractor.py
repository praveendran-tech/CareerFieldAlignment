# graduation_survey_extractor_auto_v6_3_fixed.py
# v6.3 (fixed placement + clean units):
# - Keep previous placement fixes (Career Outcomes Rate, donut legend, windows, outcomes math).
# - NEW: Strip trailing numbers from unit lines (e.g., "College Park Scholars 91" -> "College Park Scholars").
# - NEW: Normalize common partial headers to canonical names.
# - NEW: Enforce a strict whitelist of 17 units (filters out junk like "College And Department Surveys").
# - Everything else unchanged.

import os
import re
import logging
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# -------------------- Normalization helpers --------------------

THOUSANDS_BLOCK = re.compile(r'(?<!\d)(\d{1,3})(?:[ ,](\d{3}))(?:[ ,](\d{3}))*\b')

def _fix_split_thousands(text: str) -> str:
    def repl(m: re.Match):
        digits = re.sub(r'[ ,]', '', m.group(0))
        try:
            return f"{int(digits):,}"
        except Exception:
            return m.group(0)
    return THOUSANDS_BLOCK.sub(repl, text)

DIGIT_PUNCT_SPACING = re.compile(r'(?<=\d)\s*([,.])\s*(?=\d)')
def _fix_digit_punct_spacing(text: str) -> str:
    return DIGIT_PUNCT_SPACING.sub(r'\1', text)

def _normalize_text(text: str) -> str:
    t = (text or "").replace("\u2013","-").replace("\u2014","-").replace("\xa0"," ")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = _fix_split_thousands(t)
    t = _fix_digit_punct_spacing(t)
    return t

# -------------------- PDF reading --------------------

def read_pdf_pages(pdf_path: str):
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                pages.append(_normalize_text(pg.extract_text() or ""))
    except Exception:
        try:
            from PyPDF2 import PdfReader
            r = PdfReader(pdf_path)
            for p in r.pages:
                pages.append(_normalize_text(p.extract_text() or ""))  # type: ignore
        except Exception:
            pass
    return pages

# -------------------- Units (canonicalization + whitelist) --------------------

REQUIRED_UNITS = [
    "University-wide",
    "College of Agriculture and Natural Resources",
    "College of Arts and Humanities",
    "College of Behavioral and Social Sciences",
    "College of Computer, Mathematical, and Natural Sciences",
    "College of Education",
    "College of Information",
    "The A. James Clark School of Engineering",
    "Philip Merrill College of Journalism",
    "School of Architecture, Planning, and Preservation",
    "School of Public Health",
    "School of Public Policy",
    "The Robert H. Smith School of Business",
    "College Park Scholars",
    "Honors College",
    "Letters and Sciences",
    "Undergraduate Studies",
]

# Acceptable synonyms/partials -> canonical
UNIT_SYNONYMS = {
    "University Of Maryland - Overall": "University-wide",
    "University Of Maryland - Overall ": "University-wide",
    "University-Wide": "University-wide",
    "University Of Maryland": "University-wide",  # if wide on page
    "Letters And Sciences": "Letters and Sciences",
    "Letters & Sciences": "Letters and Sciences",
    "Letters And Science": "Letters and Sciences",
    "Letters And  Sciences": "Letters and Sciences",
    "Philip Merrill College Of Journalism": "Philip Merrill College of Journalism",
    "Phillip Merrill College Of Journalism": "Philip Merrill College of Journalism",
    "The A. James Clark School Of Engineering": "The A. James Clark School of Engineering",
    "The Robert H. Smith School Of Business": "The Robert H. Smith School of Business",
    "College Of Information Studies": "College of Information",
    "College Of Information": "College of Information",
    "College of Information Studies": "College of Information",
    # partial / comma-tail variants we saw in your sample:
    "College Of Agriculture And": "College of Agriculture and Natural Resources",
    "College Of Agriculture And Natural Resources": "College of Agriculture and Natural Resources",
    "College of Agriculture And": "College of Agriculture and Natural Resources",
    "College of Agriculture And Natural Resources": "College of Agriculture and Natural Resources",
    "College Of Arts And Humanities": "College of Arts and Humanities",
    "College of Arts And Humanities": "College of Arts and Humanities",
    "College Of Behavioral And Social Sciences": "College of Behavioral and Social Sciences",
    "College of Behavioral And Social Sciences": "College of Behavioral and Social Sciences",
    "College Of Computer, Mathematical And Natural Sciences": "College of Computer, Mathematical, and Natural Sciences",
    "College of Computer, Mathematical And Natural Sciences": "College of Computer, Mathematical, and Natural Sciences",
    "College Of Computer, Mathematical,": "College of Computer, Mathematical, and Natural Sciences",
    "College of Computer, Mathematical,": "College of Computer, Mathematical, and Natural Sciences",
    "College Of Computer, Mathematical": "College of Computer, Mathematical, and Natural Sciences",
    "College of Computer, Mathematical": "College of Computer, Mathematical, and Natural Sciences",
    "School Of Architecture, Planning, And Preservation": "School of Architecture, Planning, and Preservation",
    "School of Architecture, Planning, And Preservation": "School of Architecture, Planning, and Preservation",
    "School Of Architecture, Planning,": "School of Architecture, Planning, and Preservation",
    "School of Architecture, Planning,": "School of Architecture, Planning, and Preservation",
    "School Of Public Health": "School of Public Health",
    "School Of Public Policy": "School of Public Policy",
}

def normalize_unit(u: str | None, page_text: str | None = None) -> str | None:
    if not u:
        return u
    # 1) Drop trailing numbers like "... 91", "... 17"
    u = re.sub(r"\s+\d{1,4}\b$", "", u.strip())
    # 2) Title-case & standardize "of"
    title = u.title().replace("College Of ", "College of ").replace("School Of ", "School of ")
    # 3) Map synonyms/partials
    mapped = UNIT_SYNONYMS.get(title, title)
    # 4) Special-case: "University of Maryland" with UNIVERSITY-WIDE on page
    if mapped == "University Of Maryland" and page_text and ("UNIVERSITY-WIDE" in page_text.upper()):
        mapped = "University-wide"
    # 5) Final canonical acceptance (whitelist gate)
    if mapped in REQUIRED_UNITS:
        return mapped
    # If mapped equals exact canonical casing (already), keep it; else, drop
    return mapped if mapped in REQUIRED_UNITS else None

# -------------------- Utilities --------------------

def clean_number(val):
    if val is None and val != 0:
        return None
    s = re.sub(r"[,\s$%]", "", str(val))
    try:
        return int(float(s))
    except Exception:
        return None

def year_from_name(path: str, default=2015) -> int:
    m = re.search(r"(20\d{2})", os.path.basename(path))
    return int(m.group(1)) if m else default

HEADER_LINE_RE = re.compile("|".join([
    r"^UNIVERSITY OF MARYLAND - OVERALL$",
    r"^UNIVERSITY OF MARYLAND$",
    r"^UNIVERSITY[-\s]WIDE.*REPORT$",
    r"^COLLEGE OF [A-Z0-9 ,&\.-]+$",
    r"^SCHOOL OF [A-Z0-9 ,&\.-]+$",
    r"^THE A\.? JAMES CLARK SCHOOL OF ENGINEERING$",
    r"^THE ROBERT H\.? SMITH SCHOOL OF BUSINESS$",
    r"^PHILIP MERRILL COLLEGE OF JOURNALISM$",
    r"^HONORS COLLEGE$",
    r"^COLLEGE PARK SCHOLARS$",
    r"^LETTERS & SCIENCES$",
    r"^LETTERS AND SCIENCES$",
]), re.IGNORECASE)

def detect_unit_on_page(page_text: str) -> str | None:
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    # explicit near top
    for ln in lines[:20]:
        if HEADER_LINE_RE.search(ln):
            title = ln.strip()
            nu = normalize_unit(title, page_text)
            if nu:
                return nu
    # heuristic fallback
    for ln in lines[:25]:
        alpha = [ch for ch in ln if ch.isalpha()]
        if not alpha:
            continue
        up_ratio = sum(ch.isupper() for ch in alpha)/len(alpha)
        if up_ratio > 0.6 and any(k in ln.upper() for k in ["COLLEGE","SCHOOL","UNIVERSITY","HONORS","SCHOLARS","LETTERS"]):
            nu = normalize_unit(ln, page_text)
            if nu:
                return nu
    return None

# -------------------- Label-adjacent % fallback --------------------

def _windowed_percent_near(label_pat: str, text: str, window_back: int = 260, window_fwd: int = 420):
    pct_pat = re.compile(r"(\d[\d\s]{0,3})\s*%")
    for m in re.finditer(label_pat, text, re.IGNORECASE):
        lo = max(0, m.start() - window_back)
        hi = min(len(text), m.start() + window_fwd)
        window = text[lo:hi]
        m2 = pct_pat.search(window)
        if m2:
            digits = m2.group(1).replace(" ", "")
            try:
                val = int(digits)
                if 0 <= val <= 100:
                    return val
            except Exception:
                pass
    return None

# -------------------- Money/number groups --------------------
N_GROUP           = r'(\d{1,5}(?:[ ,]\d{3}){0,3})'
MONEY_MAIN        = r'(\d{1,3}(?:[ ,]\d{3})+|\d{5,6})'
MONEY_GROUP_WC    = rf'\$?\s*{MONEY_MAIN}(?:\.(\d{{2}}))?'

def _money_to_int(money_main: str, cents: str | None) -> int | None:
    return clean_number(money_main)

# -------------------- Placement helpers --------------------

CAREER_OUTCOMES_LABEL = r"Career\s+Outcomes?\s+Rate"

PLACED_LABELS = [
    r"Employed\s*FT",
    r"Employed\s*PT",
    r"Continuing\s+Education",
    r"Volunteering\s+or\s+in\s+service\s+program",
    r"Serving\s+in\s+the\s+Military",
    r"Starting\s+a\s+business",
]
UNPLACED_LABEL   = r"\bUnplaced\b"
UNRESOLVED_LABEL = r"\bUnresolved\b"

def _sum_first_integer_after(label_pat: str, text: str) -> int:
    m = re.search(label_pat + r".{0,120}?(\d{1,6}(?:[ ,]\d{3})*)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{1,6}(?:[ ,]\d{3})*)\s+" + label_pat, text, re.IGNORECASE)
    return clean_number(m.group(1)) or 0 if m else 0

def _placement_from_outcomes_counts(text: str) -> int | None:
    placed = sum(_sum_first_integer_after(p, text) for p in PLACED_LABELS)
    unplaced = _sum_first_integer_after(UNPLACED_LABEL, text)
    unresolved = _sum_first_integer_after(UNRESOLVED_LABEL, text)
    denom = placed + unplaced + unresolved
    if denom > 0:
        return round(100 * placed / denom)
    return None

# -------------------- Metric extraction --------------------

def extract_metrics_from_text(text: str, unit: str, year: int) -> dict | None:
    s = text
    out = {"Unit": unit, "Year": year,
           "Survey Response Rate": None, "Knowledge Rate": None, "Placement Rate": None,
           "Salary N": None, "Salary 25th": None, "Salary Median": None, "Salary 75th": None}

    pct = r"(\d[\d\s]{0,3})\s%"

    # SRR + KR (primary + fallback windows)
    m = re.search(rf"SURVEY\s+RESPONSE\s+RATE\s*:?\s*{pct}", s, re.IGNORECASE)
    if m: out["Survey Response Rate"] = clean_number(m.group(1).replace(" ", ""))
    if out["Survey Response Rate"] is None:
        out["Survey Response Rate"] = _windowed_percent_near(r"SURVEY\s+RESPONSE\s+RATE", s)

    m = re.search(rf"KNOWLEDGE\s+RATE\s*:?\s*{pct}", s, re.IGNORECASE)
    if m: out["Knowledge Rate"] = clean_number(m.group(1).replace(" ", ""))
    if out["Knowledge Rate"] is None:
        out["Knowledge Rate"] = _windowed_percent_near(r"KNOWLEDGE\s+RATE", s)

    # Placement rate (robust)
    pr = None
    if year >= 2023:
        m = re.search(rf"{CAREER_OUTCOMES_LABEL}\s*[-–—:]?\s*{pct}", s, re.IGNORECASE)
        if m:
            v = clean_number(m.group(1).replace(" ", ""))
            if v is not None and 0 <= v <= 100:
                pr = v
        if pr is None:
            pr = _windowed_percent_near(CAREER_OUTCOMES_LABEL, s, 300, 520)

    if pr is None:
        for pat in [
            rf"TOTAL\s+PLACEMENT\s*[-–—:]?\s*{pct}",
            rf"Total\s+Placement\s*[-–—:]?\s*{pct}",
            rf"Placement\s+Rate\s*[-–—:]?\s*{pct}",
            rf"\bPlacement\s*[-–—:]?\s*{pct}",
            rf"\bPlacements?\s*[-–—:]?\s*{pct}",
            rf"Employed\s+or\s+Continuing\s+Education\s*[-–—:]?\s*{pct}",
            rf"(Employment|Outcome|Outcomes)\s*(Rate|Total)?\s*[-–—:]?\s*{pct}",
        ]:
            m = re.search(pat, s, re.IGNORECASE)
            if m:
                v = clean_number(m.group(1).replace(" ", ""))
                if v is not None and 0 <= v <= 100:
                    pr = v; break

    if pr is None:
        m = re.search(rf"\bPlaced\b[^\n%]{{0,140}}{pct}", s, re.IGNORECASE)
        if m:
            v = clean_number(m.group(1).replace(" ", ""))
            if v is not None and 0 <= v <= 100:
                pr = v
    if pr is None:
        m = re.search(rf"\bUnplaced\b[^\n%]{{0,140}}{pct}", s, re.IGNORECASE)
        if m:
            v = clean_number(m.group(1).replace(" ", ""))
            if v is not None and 0 <= v <= 100:
                pr = max(0, min(100, 100 - v))

    if pr is None:
        for lbl in [
            r"TOTAL\s+PLACEMENT", r"Total\s+Placement",
            r"Placement\s+Rate", r"\bPlacement\b", r"\bPlacements?\b",
            r"Employed\s+or\s+Continuing\s+Education",
            r"Employment\s+Outcomes?", r"Outcomes?\s+Rate",
            CAREER_OUTCOMES_LABEL,
        ]:
            val = _windowed_percent_near(lbl, s, 280, 520)
            if val is not None:
                pr = val; break

    if pr is None:
        pr = _placement_from_outcomes_counts(s)

    out["Placement Rate"] = pr

    # Salary block
    sal_block = s
    anchor = None
    for pat in [r"REPORTED\s+SALAR(Y|IES)", r"\bSALARY\b", r"\bSALARIES\b", r"Reported\s+Salar", r"Reported\s+Salary"]:
        anchor = re.search(pat, s, re.IGNORECASE)
        if anchor: break
    if anchor:
        start = max(0, anchor.start() - 600)
        end   = min(len(s), anchor.start() + 4200)
        sal_block = s[start:end]

    got_salary = False
    for pat in [
        rf"Reported\s+Salar(?:y|ies)[^\n]*?\b{N_GROUP}\b[^\n$]*?{MONEY_GROUP_WC}\s+{MONEY_GROUP_WC}\s+{MONEY_GROUP_WC}",
        rf"\b{N_GROUP}\b[^\n$]*?{MONEY_GROUP_WC}\s+{MONEY_GROUP_WC}\s+{MONEY_GROUP_WC}",
    ]:
        m = re.search(pat, sal_block, re.IGNORECASE | re.DOTALL)
        if m:
            N   = clean_number(m.group(1))
            p25 = _money_to_int(m.group(2), m.group(3))
            p50 = _money_to_int(m.group(4), m.group(5))
            p75 = _money_to_int(m.group(6), m.group(7))
            if N and p25 and p50 and p75 and p25 < p50 < p75:
                out.update({"Salary N": N, "Salary 25th": p25, "Salary Median": p50, "Salary 75th": p75})
                got_salary = True
                break

    if not got_salary:
        n = None
        for p in [
            rf"Reported\s+Salar(?:y|ies)\s*\(?\s*{N_GROUP}\s*\)?",
            rf"\bN\s*=\s*{N_GROUP}",
            rf"\bNumber\s+of\s+Salar(?:y|ies)\s*:?\s*{N_GROUP}",
            rf"\bReported\s+By\s+(\d{{1,5}})\b"
        ]:
            m = re.search(p, sal_block, re.IGNORECASE)
            if m:
                n = clean_number(m.group(1)); break

        def grab(label_regexes):
            for lab in label_regexes:
                m = re.search(lab + r"\s*[:\-]?\s*(?:\$?\s*)?" + MONEY_MAIN + r"(?:\.(\d{2}))?", sal_block, re.IGNORECASE)
                if m:
                    return _money_to_int(m.group(1), m.group(2))
            return None

        p25 = grab([r"25(?:th)?\s+Percentile", r"\b25(?:th)?\b"])
        p50 = grab([r"50(?:th)?\s+Percentile", r"\bMedian\b", r"\b50(?:th)?\b"])
        p75 = grab([r"75(?:th)?\s+Percentile", r"\b75(?:th)?\b"])
        if p25 and p50 and p75 and p25 < p50 < p75:
            out.update({"Salary N": n, "Salary 25th": p25, "Salary Median": p50, "Salary 75th": p75})

    return out if any(out[k] is not None for k in ("Survey Response Rate","Knowledge Rate","Placement Rate","Salary N")) else None

# -------------------- Per-PDF (page-wise) --------------------

ANCHOR_ANY = re.compile(
    r"(SURVEY\s+RESPONSE\s+RATE|TOTAL\s+PLACEMENT|PLACEMENT\s+RATE|REPORTED\s+SALAR|SALARY|SALARIES|"
    r"CAREER\s+OUTCOMES?\s+RATE|PLACED|UNPLACED|EMPLOYED\s+OR\s+CONTINUING\s+EDUCATION|OUTCOMES?\s+RATE)",
    re.IGNORECASE
)

def extract_survey_data_from_pdf(pdf_path: str) -> pd.DataFrame:
    pages = read_pdf_pages(pdf_path)
    year = year_from_name(pdf_path, 2015)
    if not pages: return pd.DataFrame()

    rows, seen = [], set()
    for i, ptxt in enumerate(pages):
        unit = detect_unit_on_page(ptxt)

        if not unit and not ANCHOR_ANY.search(ptxt):
            continue

        if not unit:
            if i < 5 and "UNIVERSITY-WIDE" in ptxt.upper():
                unit = "University-wide"
            else:
                candidates = re.findall(r"(College|School) of [A-Za-z ,&\-]+", ptxt)
                if candidates:
                    unit = candidates[0].title()
        unit = normalize_unit(unit, ptxt)
        if not unit or unit in seen:
            continue

        block = ptxt
        for k in (1, 2):
            if i+k < len(pages):
                block += "\n" + pages[i+k]

        data = extract_metrics_from_text(block, unit, year)
        if data:
            rows.append(data); seen.add(unit)

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["Unit","Year"])
            .sort_values(["Year","Unit"])
            .reset_index(drop=True))

# -------------------- Folder runner --------------------

def process_folder(folder_path: str, output_file: str="graduation_survey_output.xlsx"):
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf"))
    frames = []
    for pdf in pdfs:
        print(f"Processing: {pdf.name}")
        df = extract_survey_data_from_pdf(str(pdf))
        if not df.empty:
            # filter to whitelist, again (defensive)
            df = df[df["Unit"].isin(REQUIRED_UNITS)]
            frames.append(df)
    if not frames:
        print("No data extracted from any files")
        return None
    combined = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset=["Unit","Year"])
                  .sort_values(["Year","Unit"]))
    # Save Excel with auto column widths
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(output_file, engine="openpyxl") as w:
        combined.to_excel(w, index=False, sheet_name="Survey Data")
        ws = w.sheets["Survey Data"]
        for i, col in enumerate(combined.columns, start=1):
            width = min(max(len(str(col)), combined[col].astype(str).map(len).max())+2, 50)
            ws.column_dimensions[get_column_letter(i)].width = width
    print(f"✓ Saved Excel: {output_file} ({len(combined)} rows)")
    return combined

# -------------------- Auto-run --------------------

if __name__ == "__main__":
    cwd = os.getcwd()
    print(f"Processing PDFs in: {cwd}")
    print("="*60)
    df = process_folder(cwd, "graduation_survey_output.xlsx")
    if df is not None:
        print("\nDone.")
