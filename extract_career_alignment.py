# extract_career_alignment_numbers.py
# Reads the 2016 PDF and pulls University‑Wide career-alignment numbers from the
# "Nature of Position" paragraph (directly aligned, stepping stone, pays the bills, N).

import re
import pdfplumber

# >>>>>>>>>>>>>>>>>>>> EDIT THIS PATH IF YOUR FILE NAME/PATH IS DIFFERENT <<<<<<<<<<<<<<<<<<
PDF_PATH = r"C:\Users\sousman\Python\UMD_GradReports\2016 Graduation Survey Report Final Web Version.pdf"

def read_all_text(pdf_path: str) -> str:
    """Concatenate text from all pages."""
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            # use layout=True to preserve reading order a bit better for bullet paragraphs
            out.append(p.extract_text(layout=True) or "")
    return "\n".join(out)

def clean_num(s: str) -> int:
    """Turn '2,058' -> 2058; '52' -> 52."""
    return int(re.sub(r"[^\d]", "", s))

def parse_from_nature_of_position(full_text: str):
    """
    Find the 'NATURE OF POSITION' block and pull:
      - N (number of students)
      - direct, stepping, pays (percentages)
    Returns dict or None.
    """
    # Grab the paragraph between "NATURE OF POSITION" and the next major section header.
    block_match = re.search(
        r"NATURE OF POSITION(.*?)(?:\n[A-Z][A-Z ]{3,}\n|SALARY|REPORTED SALARY|EMPLOYMENT SEARCH|CONTINUING EDUCATION|OUT OF CLASSROOM|$)",
        full_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not block_match:
        return None
    block = block_match.group(1)

    # N line: "Based on the 2,058 students who completed ..."
    n_match = re.search(r"Based on the\s+([\d,]+)\s+students", block, flags=re.IGNORECASE)
    N = clean_num(n_match.group(1)) if n_match else None

    # Percentages inside parentheses right after the phrases.
    # Example (2016): "(52%) ... (39%) ... (9%)"
    pct_match = re.search(
        r"aligned with (?:their )?career goals.*?\((\d+)%\).*?stepping stone.*?\((\d+)%\).*?pays the bills.*?\((\d+)%\)",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if pct_match:
        direct = int(pct_match.group(1))
        stepping = int(pct_match.group(2))
        pays = int(pct_match.group(3))
        return {"Directly aligned": direct, "Stepping stone": stepping, "Pays the bills": pays, "N": N}

    # -------- Fallback (if a report year lists explicit labels like "Directly aligned : 52") --------
    direct = re.search(r"Directly\s*aligned\s*:\s*(\d+)", block, flags=re.IGNORECASE)
    step = re.search(r"Stepping\s*stone\s*:\s*(\d+)", block, flags=re.IGNORECASE)
    bills = re.search(r"Pays\s*the\s*bills\s*:\s*(\d+)", block, flags=re.IGNORECASE)
    if direct and step and bills:
        return {
            "Directly aligned": int(direct.group(1)),
            "Stepping stone": int(step.group(1)),
            "Pays the bills": int(bills.group(1)),
            "N": N,
        }

    return None

def main():
    text = read_all_text(PDF_PATH)
    result = parse_from_nature_of_position(text)

    print("=" * 66)
    print("2016 University‑Wide Career Alignment")
    if not result:
        print("Could not find the 'Nature of Position' block. Double‑check the PDF path/name.")
        return

    direct = result["Directly aligned"]
    stepping = result["Stepping stone"]
    bills = result["Pays the bills"]
    N = result["N"]

    print(f"Directly aligned : {direct}")
    print(f"Stepping stone   : {stepping}")
    print(f"Pays the bills   : {bills}")
    print(f"N                : {N if N is not None else '—'}")
    print(f"Check total      : {direct + stepping + bills} (should be 100)")

if __name__ == "__main__":
    main()

