#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

# ---------------- Fixed grid (17 units × 2015–2024 = 170 rows) ----------------
UNITS_FIXED = [
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
YEARS_FIXED = list(range(2015, 2025))
EXPECTED_ROWS = len(UNITS_FIXED) * len(YEARS_FIXED)

OUT_COLS = [
    "Unit","Year",
    "Masters N","Masters %","PhD/Doctoral N","PhD/Doctoral %","Law N","Law %",
    "Health Prof N","Health Prof %","Certificate N","Certificate %",
    "Second Bachelor's N","Second Bachelor's %","Associate's N","Associate's %",
    "Non-degree N","Non-degree %","Unspecified N","Unspecified %","Total N",
]
DEFAULT_ROW = {
    "Masters N": 0, "Masters %": None,
    "PhD/Doctoral N": 0, "PhD/Doctoral %": None,
    "Law N": 0, "Law %": None,
    "Health Prof N": 0, "Health Prof %": None,
    "Certificate N": 0, "Certificate %": None,
    "Second Bachelor's N": 0, "Second Bachelor's %": None,
    "Associate's N": 0, "Associate's %": None,
    "Non-degree N": 0, "Non-degree %": None,
    "Unspecified N": 0, "Unspecified %": None,
    "Total N": 0,
}

# ---------------- Cleaning ----------------
def clean_int(tok: Optional[str]) -> Optional[int]:
    if tok is None: return None
    s = str(tok).replace(",", "").strip()
    return int(s) if re.fullmatch(r"\d+", s) else None

def clean_pct(tok: Optional[str]) -> Optional[float]:
    if tok is None: return None
    s = str(tok).strip().replace("%", "")
    if s.startswith("<"):          # treat "<1%" as 1.0
        return 1.0
    m = re.fullmatch(r"\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None

# ---------------- Label normalization ----------------
LABEL_PATTERNS = {
    r"^masters(?:\s*/\s*mba)?\b|mba\b": ("Masters N","Masters %"),
    r"^ph\.?\s*d\.?.*|^phd\b|^doctoral\b|ph\.?d\.?\s*or\s*doctoral": ("PhD/Doctoral N","PhD/Doctoral %"),
    r"^law\b|law\s*\(j\.?d\.?\)": ("Law N","Law %"),
    r"^health professional\b|^graduate/first professional\b": ("Health Prof N","Health Prof %"),
    r"^certificate\b|certificate/?certification": ("Certificate N","Certificate %"),
    r"^second bachelor": ("Second Bachelor's N","Second Bachelor's %"),
    r"^associate": ("Associate's N","Associate's %"),
    r"^non-?degree": ("Non-degree N","Non-degree %"),
    r"^unspecified$": ("Unspecified N","Unspecified %"),
    r"^other$": ("Unspecified N","Unspecified %"),  # tan-table "Other" -> Unspecified
}
LABEL_REGEXES = [(re.compile(p, re.I), cols) for p, cols in LABEL_PATTERNS.items()]

def normalize_label_text(label: str) -> str:
    lab = re.sub(r"^[^\wA-Za-z]+", "", (label or "")).strip()
    return re.sub(r"\s+", " ", lab)

def map_label(label: str):
    lab = normalize_label_text(label)
    for rx, cols in LABEL_REGEXES:
        if rx.search(lab):
            return cols
    return None

# ---------------- Unit + Year detection ----------------
UNIT_HINTS = {
    "University-wide": [r"university[-\s]wide", r"\boverall\b"],
    "College of Agriculture and Natural Resources": [r"college of agriculture.*natural resources", r"\bagnr\b"],
    "College of Arts and Humanities": [r"college of arts.*humanities", r"\barhu\b"],
    "College of Behavioral and Social Sciences": [r"college of behavioral.*social sciences", r"\bbsos\b"],
    "College of Computer, Mathematical, and Natural Sciences": [r"college of computer.*mathematical.*natural sciences", r"\bcmns\b"],
    "College of Education": [r"college of education", r"\beduc\b"],
    "College of Information": [r"college of information studies|college of information\b", r"\binfo\b"],
    "The A. James Clark School of Engineering": [r"a\.?\s*james clark school of engineering", r"\bengr\b"],
    "Philip Merrill College of Journalism": [r"philip+p? merrill college of journalism", r"\bjour\b"],
    "School of Architecture, Planning, and Preservation": [r"school of architecture.*planning.*preservation", r"\barch\b"],
    "School of Public Health": [r"school of public health", r"\bsphl\b"],
    "School of Public Policy": [r"school of public policy", r"\bplcy\b"],
    "The Robert H. Smith School of Business": [r"robert h\.?\s*smith school of business", r"\bbmgt\b|\bbusiness school\b"],
    "College Park Scholars": [r"college park scholars"],
    "Honors College": [r"honors college"],
    "Letters and Sciences": [r"letters (?:&|and)\s*sciences", r"\bltsc\b"],
    "Undergraduate Studies": [r"undergraduate studies", r"\bugst\b"],
}
def detect_unit_in_text(text: str) -> Optional[str]:
    for unit, pats in UNIT_HINTS.items():
        if any(re.search(p, text, re.I) for p in pats):
            return unit
    return None

def detect_year_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"(20\d{2})", path.stem)
    return int(m.group(1)) if m else None

# ---------------- Table pass (extract_tables + scoring) ----------------
def score_table(table) -> Tuple[int, List[Tuple[str,int,float]]]:
    rows = [[(c or "").strip() for c in r] for r in (table or [])]
    rows = [r for r in rows if any(r)]
    if len(rows) < 5:
        return -1, []

    # find a header row with "#" and "%"
    header_idx = None
    for i, r in enumerate(rows[:5]):
        j = " ".join(r).lower()
        if "#" in j and "%" in j:
            header_idx = i; break
    data_rows = rows[(header_idx+1 if header_idx is not None else 0):]

    recognized, parsed = 0, []
    for r in data_rows:
        label = r[0] if r else ""
        if re.fullmatch(r"TOTAL", label.strip(), re.I):
            # usually last meaningful number in row is N
            nums = [x for x in r if re.fullmatch(r"\d{1,3}(?:,\d{3})*", x)]
            parsed.append(("TOTAL", clean_int(nums[-1]) if nums else None, None))
            continue

        cols = map_label(label)
        if not cols:
            continue

        n = clean_int(r[-2]) if len(r) >= 2 else None
        pct = clean_pct(r[-1]) if len(r) >= 3 else None
        if n is None or pct is None:
            j = " ".join(r)
            m = re.search(r"(-?[\d,]+)\s+((?:<\s*1)|\d+(?:\.\d+)?)\s*%?$", j)
            if m:
                n = clean_int(m.group(1)); pct = clean_pct(m.group(2))

        if n is not None and pct is not None:
            parsed.append((label, n, pct)); recognized += 1

    pct_sum = sum(v for _,_,v in parsed if v is not None)
    score = recognized + (2 if 90 <= pct_sum <= 110 else 0) + (1 if any(l=="TOTAL" for l,_,_ in parsed) else 0)
    return score, parsed

def parse_best_table(page) -> List[Tuple[str,int,float]]:
    best_rows, best_score = [], -1
    try:
        tables = page.extract_tables()
    except Exception:
        tables = []
    for tbl in tables or []:
        score, rows = score_table(tbl)
        if score > best_score:
            best_score, best_rows = score, rows
    return best_rows

# ---------------- Word pass (handles 1/2/3-line rows) ----------------
def parse_by_words(page) -> List[Tuple[str,int,float]]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    upper = " ".join(w["text"].upper() for w in words)
    if "CONTINUING" not in upper or "EDUCATION" not in upper:
        return []

    # band: from header to TOTAL
    header_top = None
    for w in words:
        if re.search(r"type of degree|type.*program", w["text"], re.I):
            header_top = w["top"] - 6; break
    if header_top is None:
        for w in words:
            if re.search(r"(Masters|Ph\.?D|Law|Health Professional|Second Bachelor|Certificate|Unspecified|Non-?Degree|Other)", w["text"], re.I):
                header_top = w["top"] - 8; break
    if header_top is None: return []

    total_bottom = None
    for w in words:
        if re.fullmatch(r"TOTAL", w["text"].strip(), re.I):
            total_bottom = w["bottom"] + 6; break
    if total_bottom is None:
        pcts = [w for w in words if re.search(r"%$", w["text"].strip())]
        if not pcts: return []
        total_bottom = max(w["bottom"] for w in pcts) + 8

    band = [w for w in words if header_top <= (w["top"]+w["bottom"])/2 <= total_bottom]

    # group by y and sort by x
    lines: Dict[float, List[dict]] = {}
    for w in band:
        cy = round(0.5*(w["top"]+w["bottom"]), 2)
        lines.setdefault(cy, []).append(w)

    seq = []
    for cy, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        ws = sorted(ws, key=lambda z: z["x0"])
        toks = [t["text"].strip() for t in ws if t["text"].strip()]
        if toks: seq.append({"cy": cy, "toks": toks, "text": " ".join(toks)})

    def split_line(toks):
        """Return (label, n, pct) if present on one line; else None."""
        pct_i = next((k for k in range(len(toks)-1, -1, -1)
                      if re.fullmatch(r"(?:<\s*1|\d+(?:\.\d+)?)%?", toks[k])), None)
        if pct_i is None: return None
        n_i = None
        for j in range(pct_i-1, -1, -1):
            if re.fullmatch(r"\d{1,3}(?:,\d{3})*", toks[j]):
                n_i = j; break
        if n_i is None: return None
        label = " ".join(toks[:n_i]).strip(" :")
        return label, clean_int(toks[n_i]), clean_pct(toks[pct_i])

    parsed: List[Tuple[str,int,float]] = []
    i = 0
    while i < len(seq):
        toks = seq[i]["toks"]
        text = seq[i]["text"]
        # skip header row(s)
        if re.search(r"type of degree|type.*program", text, re.I):
            i += 1; continue

        # A) try one-line
        one = split_line(toks)
        if one:
            label, n, pct = one
            if re.fullmatch(r"TOTAL", label, re.I):
                parsed.append(("TOTAL", n, None))
            else:
                cols = map_label(label)
                if cols and n is not None and pct is not None:
                    parsed.append((label, n, pct))
            i += 1; continue

        # B) try two-line (current label + next numeric line)
        if i + 1 < len(seq):
            two = split_line(seq[i+1]["toks"])
            if two:
                label = text.strip(" :")
                n, pct = two[1], two[2]
                if re.fullmatch(r"TOTAL", label, re.I):
                    parsed.append(("TOTAL", n, None))
                else:
                    cols = map_label(label)
                    if cols and n is not None and pct is not None:
                        parsed.append((label, n, pct))
                i += 2; continue

        # C) try three-line (wrapped label across 2 lines + numeric line)
        if i + 2 < len(seq):
            three = split_line(seq[i+2]["toks"])
            if three:
                label = (text + " " + seq[i+1]["text"]).strip(" :")
                n, pct = three[1], three[2]
                cols = map_label(label)
                if cols and n is not None and pct is not None:
                    parsed.append((label, n, pct))
                i += 3; continue

        i += 1

    return parsed

# ---------------- Build record + richness ----------------
def build_record(unit: str, year: int, rows: List[Tuple[str,int,float]]) -> Dict:
    rec = {"Unit": unit, "Year": year, **DEFAULT_ROW}
    totalN = None
    for label, n, pct in rows:
        if re.fullmatch(r"TOTAL", label, re.I):
            totalN = n if n is not None else totalN
            continue
        cols = map_label(label)
        if not cols:
            continue
        n_col, p_col = cols
        rec[n_col] = n or 0
        rec[p_col] = pct
    rec["Total N"] = totalN or rec["Total N"]
    return rec

def richness(row: Dict) -> int:
    return sum(1 for k,v in row.items() if k not in ("Unit","Year") and v not in (0, None))

# ---------------- Page extractor (choose best of table vs words) ----------------
def extract_page_record(page, unit: str, year: int) -> Optional[Dict]:
    rows_tbl = parse_best_table(page)
    rows_wrd = parse_by_words(page)
    # choose the richer one
    def rcount(rows): return sum(1 for _,_,p in rows if p is not None)
    rows = rows_tbl if rcount(rows_tbl) >= rcount(rows_wrd) else rows_wrd
    if not rows: return None
    rec = build_record(unit, year, rows)
    return rec if richness(rec) > 0 else None

# ---------------- Document extractor (stateful unit memory) ----------------
def extract_document(pdf_path: Path) -> List[Dict]:
    year = detect_year_from_filename(pdf_path)
    results: List[Dict] = []
    recent_units = deque(maxlen=3)
    current_unit = "University-wide"

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # find any unit mentions on this page (even if not CE)
            text = page.extract_text() or ""
            unit_here = detect_unit_in_text(text)
            if unit_here:
                current_unit = unit_here
                recent_units.append(current_unit)

            # CE page?
            up = (text or "").upper()
            if "CONTINUING EDUCATION" not in up:
                continue

            # if CE page didn’t announce unit, use the most recent seen
            unit_for_page = unit_here or (recent_units[-1] if recent_units else current_unit)

            rec = extract_page_record(page, unit_for_page, year)
            if rec:
                results.append(rec)

    return results

# ---------------- Grid builder ----------------
def enforce_170_grid(df: pd.DataFrame) -> pd.DataFrame:
    # keep richest per (Unit, Year)
    df["_score"] = df.apply(lambda r: sum(1 for k in DEFAULT_ROW if r.get(k) not in (0, None)), axis=1)
    df = df.sort_values(["Unit","Year","_score"], ascending=[True,True,False])\
           .drop_duplicates(["Unit","Year"])\
           .drop(columns=["_score"], errors="ignore")

    # build strict grid
    grid = []
    for u in UNITS_FIXED:
        for y in YEARS_FIXED:
            hit = df[(df["Unit"] == u) & (df["Year"] == y)]
            grid.append(hit.iloc[0].to_dict() if not hit.empty else {"Unit": u, "Year": y, **DEFAULT_ROW})
    out = pd.DataFrame(grid)[OUT_COLS].sort_values(["Unit","Year"]).reset_index(drop=True)
    assert len(out) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS}, got {len(out)}"
    return out

# ---------------- Orchestrator ----------------
def process_all_pdfs(pdf_folder: str, out_csv: str = "continuing_education_data.csv") -> pd.DataFrame:
    folder = Path(pdf_folder)
    records: List[Dict] = []
    for pdf in sorted(folder.glob("*.pdf")):
        print(f"Processing: {pdf.name}")
        try:
            records.extend(extract_document(pdf))
        except Exception as e:
            print(f"  !! {pdf.name}: {e}")

    if not records:
        print("No data extracted.")
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.DataFrame(records)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = None if col.endswith("%") else 0
    df = df[OUT_COLS]
    df = enforce_170_grid(df)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows (expected {EXPECTED_ROWS}).")
    return df

if __name__ == "__main__":
    # Change to your PDF directory (e.g., ".")
    process_all_pdfs(".")
