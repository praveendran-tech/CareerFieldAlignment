#!/usr/bin/env python3
"""
UMD Graduation Survey (2015–2024) outcome extractor (robust + row stitching + strict whitelist).

Outputs:
  umd_outcomes_2015_2024.csv

Dependencies:
  pip install pdfplumber pandas
"""

import os, re, sys
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import pdfplumber
import pandas as pd

# ------------- IO -------------
PDF_DIR = os.path.abspath(os.getenv("PDF_DIR", "."))
OUT_CSV = os.path.abspath(os.getenv("OUT_CSV", "umd_outcomes_2015_2024.csv"))

PDF_FILES = [
    "2015 Graduation Survey Report.pdf",
    "2016 Graduation Survey Report Final Web Version.pdf",
    "2017 Graduation Survey Report Final Print Version.pdf",
    "2018 Graduation Survey Report Final Version.pdf",
    "2019 Graduation_Survey_2019_FINAL_Report_5-1-20.pdf",
    "2020 UCC Graduation Survey Report Final_For the WEB.pdf",
    "2021 UCC Graduation Survey Report 6.2.2022 Final Version2.pdf",
    "2022 UCC Graduation Survey Report_6.2.2023_Final_V1.pdf",
    "2023 UMD Graduation Survey - Final Web Version.pdf",
    "2024 UCC Graduation Survey Report Final.pdf",
]

# ------------- Strict unit whitelist (exact names you want) -------------
ALLOWED_UNITS = [
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

# ------------- Map common variants -> canonical allowed names -------------
UNIT_CANON_MAP = {
    # casing/spacing/hyphens
    "university-wide": "University-wide",
    "university wide": "University-wide",
    "university  wide": "University-wide",
    "college of information studies": "College of Information",
    "a. james clark school of engineering": "The A. James Clark School of Engineering",
    "the a. james clark school of engineering": "The A. James Clark School of Engineering",
    "phillip merrill college of journalism": "Philip Merrill College of Journalism",
    "robert h. smith school of business": "The Robert H. Smith School of Business",
    "the robert h. smith school of business": "The Robert H. Smith School of Business",
    "letters & sciences": "Letters and Sciences",
    "letters and sciences": "Letters and Sciences",
    # all-caps variants that sometimes appear in PDFs
    "UNIVERSITY-WIDE".lower(): "University-wide",
    "COLLEGE OF INFORMATION STUDIES".lower(): "College of Information",
    "A. JAMES CLARK SCHOOL OF ENGINEERING".lower(): "The A. James Clark School of Engineering",
    "PHILIP MERRILL COLLEGE OF JOURNALISM".lower(): "Philip Merrill College of Journalism",
    "ROBERT H. SMITH SCHOOL OF BUSINESS".lower(): "The Robert H. Smith School of Business",
    "LETTERS & SCIENCES".lower(): "Letters and Sciences",
    "LETTERS AND SCIENCES".lower(): "Letters and Sciences",
}

def canon_unit(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    key = re.sub(r"\s+", " ", u).strip()
    key_l = key.lower()
    # direct exact match first
    for allowed in ALLOWED_UNITS:
        if key == allowed:
            return allowed
    # map common variants
    mapped = UNIT_CANON_MAP.get(key_l)
    if mapped in ALLOWED_UNITS:
        return mapped
    # sometimes the page heading is "University of Maryland – <UNIT>"; strip the prefix
    if " – " in key or " - " in key:
        tail = re.split(r"\s[-–]\s", key, maxsplit=1)[-1].strip()
        return canon_unit(tail)
    return None  # anything else is rejected

# ------------- Labels -------------
LABEL_CANON = OrderedDict({
    r"^employed\s*ft$|^employed\s*full[- ]?time$": "Employed FT",
    r"^employed\s*pt$|^employed\s*part[- ]?time$": "Employed PT",
    r"^continuing\s*edu(cation)?$|^graduate\s*school$|^pursuing\s*(further\s*)?edu(cation)?$": "Continuing Edu",
    r"^volunteering(\s*or\s*in\s*service\s*program)?$": "Volunteering",
    r"^participating\s*in\s*a\s*volunteer\s*or\s*service\s*program$": "Volunteering",
    r"^service(\s*program)?$": "Volunteering",
    r"^serving\s*in\s*the\s*military$|^military$": "Military",
    r"^starting\s*(a\s*)?business$|^business$|^entrepreneur(ship)?$": "Business",
    r"^unplaced$|^still\s*seeking$|^seeking\s*(employment|grad\s*school)$": "Unplaced",
    r"^unresolved$": "Unresolved",
    r"^grand\s*total$|^total(\s*n)?$": "Total",
    r"^not\s*seeking$": "Not Seeking",
})

COLS = [
    "Unit","Year",
    "Employed FT N","Employed FT %",
    "Employed PT N","Employed PT %",
    "Continuing Edu N","Continuing Edu %",
    "Volunteering N","Volunteering %",
    "Military N","Military %",
    "Business N","Business %",
    "Unplaced N","Unplaced %",
    "Unresolved N","Unresolved %",
    "Total N","Not Seeking N","Placement Rate %",
]

PLACEMENT_PATTERNS = [
    re.compile(r"\bPlacement\s*Rate[:\s-]*\s*(\d{1,3}(?:\.\d+)?%)", re.I),
    re.compile(r"\bTotal\s*Placement\s*[-:]*\s*(\d{1,3}(?:\.\d+)?%)", re.I),
]

# ------------- Table extraction settings -------------
LATTICE = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 12,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}
STREAM = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "text_y_tolerance": 3,
    "text_x_tolerance": 2,
    "intersection_tolerance": 5,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

# ------------- Helpers -------------
def year_from_name(name: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None

def normalize_label(txt: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", (txt or "")).strip().strip(":")
    tl = t.lower()
    for patt, key in LABEL_CANON.items():
        if re.match(patt, tl):
            return key
    return None

def numify(x: str) -> Optional[str]:
    if x is None: return None
    s = str(x).replace(",", "").strip()
    m = re.search(r"\b(\d{1,6})\b", s)
    return m.group(1) if m else None

def pctify(x: str) -> Optional[str]:
    if x is None: return None
    s = str(x).replace(" ", "")
    m = re.search(r"(\d{1,3}(?:\.\d+)?%)", s)
    return m.group(1) if m else None

def placement_from_text(text: str) -> Optional[str]:
    for pat in PLACEMENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None

def find_outcome_bbox(page: pdfplumber.page.Page) -> Optional[Tuple[float,float,float,float]]:
    """Locate the REPORTED/GRADUATE OUTCOMES header and return a bbox under it (helps detection)."""
    try:
        words = page.extract_words()
    except Exception:
        return None
    joined = " ".join(w["text"] for w in words)
    if not re.search(r"(REPORTED\s+OUTCOMES|GRADUATE\s+OUTCOMES)", joined, re.I):
        return None
    for w in words:
        if re.search(r"(REPORTED|GRADUATE)", w["text"], re.I):
            top = max(0, w["top"] - 10)
            x0, y0, x1, y1 = page.bbox
            return (x0 + 20, top + 20, x1 - 20, y1 - 40)
    return None

def looks_like_outcome_table(table: List[List[str]]) -> bool:
    if not table or not table[0]: return False
    header = [ (c or "").strip().lower() for c in table[0] ]
    joined = " ".join(header)
    if ("outcome" in joined or "graduate" in joined) and ("#" in joined or "%" in joined):
        return True
    if any("%" in h or "#" in h for h in header) and len(table) >= 5:
        return True
    return False

def stitch_rows(table: List[List[str]]) -> List[List[str]]:
    """If a label is on one row and the numbers on the next, merge them."""
    def has_nums(row: List[str]) -> bool:
        s = " ".join((c or "") for c in row)
        return bool(re.search(r"\d{1,5}", s) or re.search(r"\d{1,3}\.\d+%|\d{1,3}%$", s))
    out = []
    skip_next = False
    for i in range(len(table)):
        if skip_next:
            skip_next = False
            continue
        row = [(c or "").strip() for c in table[i]]
        label = next((c for c in row if c), "")
        label_norm = normalize_label(label)
        if label_norm and not has_nums(row) and i+1 < len(table):
            nxt = [(c or "").strip() for c in table[i+1]]
            merged = [row[j] + (" " + nxt[j] if nxt[j] else "") if j < len(row) else (nxt[j] if j < len(nxt) else "") for j in range(max(len(row), len(nxt)))]
            out.append(merged)
            skip_next = True
        else:
            out.append(row)
    return out

def parse_outcome_table(table: List[List[str]]) -> Dict[str, Dict[str, Optional[str]]]:
    acc: Dict[str, Dict[str, Optional[str]]] = {}
    for r in table[1:]:
        cells = [(c if isinstance(c, str) else ("" if c is None else str(c))).strip() for c in r]
        if not any(cells):
            continue
        label_cell = next((c for c in cells if c), None)
        if not label_cell:
            continue
        label_norm = normalize_label(label_cell)
        if not label_norm:
            continue
        n_val = None
        p_val = None
        for c in cells:
            if n_val is None:
                n_val = numify(c)
            if p_val is None:
                p_val = pctify(c)
        if label_norm not in acc:
            acc[label_norm] = {"N": n_val, "P": p_val}
        else:
            if not acc[label_norm]["N"] and n_val: acc[label_norm]["N"] = n_val
            if not acc[label_norm]["P"] and p_val: acc[label_norm]["P"] = p_val
    return acc

def parse_outcome_from_text(text: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Text-mode fallback: scan lines for known labels then grab the nearest numbers/percents."""
    acc: Dict[str, Dict[str, Optional[str]]] = {}
    patterns = {
        "Employed FT": [r"Employed\s*FT", r"Employed\s*Full[- ]?Time"],
        "Employed PT": [r"Employed\s*PT", r"Employed\s*Part[- ]?Time"],
        "Continuing Edu": [r"Continuing\s*Edu(?:cation)?", r"Graduate\s*School", r"Pursuing\s*(?:Further\s*)?Edu(?:cation)?"],
        "Volunteering": [r"Participating\s*in\s*a\s*volunteer\s*or\s*service\s*program", r"Volunteering(?:\s*or\s*in\s*service\s*program)?", r"Service(?:\s*Program)?"],
        "Military": [r"Serving\s*in\s*the\s*Military", r"\bMilitary\b"],
        "Business": [r"Starting\s*(?:a\s*)?Business", r"\bBusiness\b", r"Entrepreneur(?:ship)?"],
        "Unplaced": [r"\bUnplaced\b", r"Still\s*Seeking", r"Seeking\s*(?:Employment|Grad\s*School)"],
        "Unresolved": [r"\bUnresolved\b"],
        "Total": [r"Grand\s*Total", r"^Total\b", r"Total\s*N"],
        "Not Seeking": [r"Not\s*Seeking"],
    }
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for canon, p_list in patterns.items():
        for pat in p_list:
            rx = re.compile(pat + r".{0,60}?(\d{1,6}).{0,30}?(\d{1,3}(?:\.\d+)?%)", re.I)
            for ln in lines:
                m = rx.search(ln)
                if m:
                    n, p = m.group(1), m.group(2)
                    acc[canon] = {"N": n, "P": p}
                    break
            if canon in acc:
                break
    return acc

def build_row(unit: str, year: int, acc: Dict[str, Dict[str, Optional[str]]], placement_hint: Optional[str]) -> Dict[str, Optional[str]]:
    def gp(key: str, k: str) -> Optional[str]:
        return acc.get(key, {}).get(k)
    row = {
        "Unit": unit, "Year": year,
        "Employed FT N": gp("Employed FT","N"), "Employed FT %": gp("Employed FT","P"),
        "Employed PT N": gp("Employed PT","N"), "Employed PT %": gp("Employed PT","P"),
        "Continuing Edu N": gp("Continuing Edu","N"), "Continuing Edu %": gp("Continuing Edu","P"),
        "Volunteering N": gp("Volunteering","N"), "Volunteering %": gp("Volunteering","P"),
        "Military N": gp("Military","N"), "Military %": gp("Military","P"),
        "Business N": gp("Business","N"), "Business %": gp("Business","P"),
        "Unplaced N": gp("Unplaced","N"), "Unplaced %": gp("Unplaced","P"),
        "Unresolved N": gp("Unresolved","N"), "Unresolved %": gp("Unresolved","P"),
        "Total N": gp("Total","N"), "Not Seeking N": gp("Not Seeking","N"),
        "Placement Rate %": placement_hint,
    }
    # Derive percents when Total N exists
    try:
        total_n = float(row["Total N"]) if row["Total N"] else None
    except Exception:
        total_n = None
    for k in ["Employed FT","Employed PT","Continuing Edu","Volunteering","Military","Business","Unplaced","Unresolved"]:
        n_key = f"{k} N"; p_key = f"{k} %"
        if not row[p_key] and row[n_key] and total_n and total_n > 0:
            row[p_key] = f"{round((float(row[n_key]) / total_n) * 100, 1)}%"

    # Placement Rate: prefer text hint; else 100 - (Unplaced% + Unresolved%); else placed_N/total_N
    if not row["Placement Rate %"]:
        up, ur = row["Unplaced %"], row["Unresolved %"]
        try:
            if up and ur:
                pv = 100.0 - (float(up.rstrip('%')) + float(ur.rstrip('%')))
                row["Placement Rate %"] = f"{round(pv, 1)}%"
        except Exception:
            pass
    if not row["Placement Rate %"] and total_n:
        placed_n = 0.0
        for k in ["Employed FT","Employed PT","Continuing Edu","Volunteering","Military","Business"]:
            v = row[f"{k} N"]
            if v:
                try: placed_n += float(v)
                except: pass
        if placed_n and total_n:
            row["Placement Rate %"] = f"{round((placed_n/total_n)*100, 1)}%"

    return row

def page_unit_candidates(text: str) -> List[str]:
    """
    Resolve possible unit names from page text into ALLOWED_UNITS only.
    """
    hits = []
    # Pattern like "University of Maryland – <UNIT>"
    m = re.search(r"University of Maryland\s*[-–]\s*([^\n]+)", text, flags=re.I)
    if m:
        u = canon_unit(m.group(1))
        if u in ALLOWED_UNITS:
            hits.append(u)
    # Scan known names/aliases and canonize strictly
    for name in set(ALLOWED_UNITS + list(UNIT_CANON_MAP.keys())):
        for mm in re.finditer(re.escape(name), text, flags=re.I):
            u = canon_unit(mm.group(0))
            if u in ALLOWED_UNITS:
                hits.append(u)
    # unique preserve order
    res, seen = [], set()
    for h in hits:
        if h not in seen:
            seen.add(h); res.append(h)
    return res

def extract_from_pdf(pdf_path: str) -> List[Dict[str, Optional[str]]]:
    rows: List[Dict[str, Optional[str]]] = []
    year = year_from_name(os.path.basename(pdf_path)) or 0

    with pdfplumber.open(pdf_path) as pdf:
        page_text_cache = []
        for page in pdf.pages:
            try:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                txt = ""
            page_text_cache.append(txt)

        last_unit: Optional[str] = None
        for i, page in enumerate(pdf.pages):
            text = page_text_cache[i]

            # Strict unit detection: current page -> prev -> prev-2; only ALLOWED_UNITS survive
            units = page_unit_candidates(text)
            if not units and i > 0:
                units = page_unit_candidates(page_text_cache[i-1])
            if not units and i > 1:
                units = page_unit_candidates(page_text_cache[i-2])

            unit = units[0] if units else (last_unit if last_unit in ALLOWED_UNITS else "University-wide")
            if unit not in ALLOWED_UNITS:
                unit = "University-wide"
            last_unit = unit

            placement_hint = placement_from_text(text) or (placement_from_text(page_text_cache[i-1]) if i>0 else None)

            # Try to focus detection around the outcomes header
            bbox = find_outcome_bbox(page)

            def extract_tables_any():
                all_tbls = []
                # lattice
                try:
                    all_tbls.extend(page.extract_tables(LATTICE))
                except Exception:
                    pass
                if bbox:
                    try:
                        with page.crop(bbox) as cp:
                            all_tbls.extend(cp.extract_tables(LATTICE))
                    except Exception:
                        pass
                # stream
                try:
                    all_tbls.extend(page.extract_tables(STREAM))
                except Exception:
                    pass
                if bbox:
                    try:
                        with page.crop(bbox) as cp:
                            all_tbls.extend(cp.extract_tables(STREAM))
                    except Exception:
                        pass
                return all_tbls

            picked = False
            tables = extract_tables_any()

            for tbl in tables or []:
                table = [[(c if isinstance(c, str) else ("" if c is None else str(c))).strip() for c in row] for row in (tbl or [])]
                table = stitch_rows(table)
                if not looks_like_outcome_table(table):
                    continue
                acc = parse_outcome_table(table)
                if acc:
                    rows.append(build_row(unit, year, acc, placement_hint))
                    picked = True
                    break

            if not picked:
                # fallback regex
                acc = parse_outcome_from_text(text)
                if acc:
                    rows.append(build_row(unit, year, acc, placement_hint))

    return rows

def main():
    all_rows: List[Dict[str, Optional[str]]] = []
    missing: List[str] = []

    for name in PDF_FILES:
        p = os.path.join(PDF_DIR, name)
        if not os.path.exists(p):
            missing.append(name); continue
        try:
            all_rows.extend(extract_from_pdf(p))
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed {name}: {e}\n")

    if not all_rows:
        print("No rows extracted. Check PDF_DIR and filenames.")
        sys.exit(2)

    df = pd.DataFrame(all_rows)
    # Ensure all columns exist
    for c in COLS:
        if c not in df.columns: df[c] = None

    # Canonicalize Units strictly; drop anything not in whitelist
    df["Unit"] = df["Unit"].apply(canon_unit)
    df = df[df["Unit"].isin(ALLOWED_UNITS)].copy()

    # Deduplicate per (Unit, Year) by completeness score
    score_cols = [c for c in COLS if c not in ("Unit","Year")]
    df["_score"] = df[score_cols].notna().sum(axis=1)
    df = (df.sort_values(["Year","Unit","_score"], ascending=[True, True, False])
            .drop_duplicates(subset=["Unit","Year"], keep="first")
            .drop(columns=["_score"]))
    df = df[COLS].sort_values(["Year","Unit"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")
    print(f"Rows: {len(df)}")
    if missing:
        print("Missing files:", *missing, sep="\n- ")

if __name__ == "__main__":
    main()
