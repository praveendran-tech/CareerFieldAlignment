def extract_career_alignment_from_block(nop_text: str):
    """
    Extract Direct / Stepping / Pays / N strictly from the *career-alignment* bullet.
    We require the phrase 'career goals' and capture all three percents in one pass,
    so we don't accidentally pull the field-alignment numbers.
    """
    t = norm(nop_text)

    # 1) Try a single-bullet, in-order capture:
    # aligned ... (xx%) ... stepping ... (yy%) ... pays the bills ... (zz%)
    pat1 = re.compile(
        r"aligned[^()%]*\((\d{1,2})%\).*?"          # Directly aligned
        r"stepping(?:\s*stone)?[^()%]*\((\d{1,2})%\).*?"  # Stepping stone
        r"(?:pays the bills[^()%]{0,60}\((\d{1,2})%\)|\((\d{1,2})%\)[^.\n]{0,80}pays the bills)",
        flags=re.I | re.S,
    )
    m = pat1.search(t) if "career goals" in t.lower() else None

    direct = step = bills = None
    if m:
        direct = int(m.group(1))
        step   = int(m.group(2))
        # bills may be in group 3 OR 4 depending on phrasing order
        bills_group = m.group(3) or m.group(4)
        bills = int(bills_group) if bills_group else None

    # 2) If in-order failed (rare layout wrapping), try to find the specific bullet first,
    # then run smaller regexes inside just that bullet.
    if direct is None or step is None:
        # locate the bullet that mentions career goals / stepping stone / pays the bills
        parts = re.split(r"(?:\n|•)", t)
        career_bullet = None
        for p in parts:
            lp = p.lower()
            if "career goals" in lp and ("stepping" in lp or "stepping stone" in lp) and "pays the bills" in lp:
                career_bullet = p
                break

        if career_bullet:
            cb = career_bullet
            md = re.search(r"aligned[^()%]*\((\d{1,2})%\)", cb, flags=re.I)
            ms = re.search(r"stepping(?:\s*stone)?[^()%]*\((\d{1,2})%\)", cb, flags=re.I)
            mb = (re.search(r"pays the bills[^()%]{0,60}\((\d{1,2})%\)", cb, flags=re.I)
                  or re.search(r"\((\d{1,2})%\)\s*[^.\n]{0,80}pays the bills", cb, flags=re.I))
            direct = int(md.group(1)) if md else None
            step   = int(ms.group(1)) if ms else None
            bills  = int((mb.group(1) if mb else 0)) if mb else None

    # 3) N can be anywhere in this block
    mN = re.search(r"Based on the\s+([\d,]+)\s+students.*?employment outcome", t, flags=re.I)
    N  = int(mN.group(1).replace(",", "")) if mN else None

    # 4) Safe fallback for 'bills' if both direct & step present
    if bills is None and isinstance(direct, int) and isinstance(step, int):
        tmp = 100 - direct - step
        if 0 <= tmp <= 30:
            bills = tmp

    return direct, step, bills, N

