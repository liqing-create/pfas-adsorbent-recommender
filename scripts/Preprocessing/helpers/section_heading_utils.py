import re
import unicodedata
from typing import Iterable, Optional, Sequence, Tuple


ABSTRACT_LABEL = "abstract"
METHODS_LABEL = "materials_and_methods"
RESULTS_LABEL = "results_and_discussion"

SECTION_LABELS = {
    "abstract": ABSTRACT_LABEL,
    "materials_and_methods": METHODS_LABEL,
    "results_and_discussion": RESULTS_LABEL,
}

ROMAN = r"[ivxlcdm]+"

ENUM_PREFIX = re.compile(
    rf"""^\s*
         (?:\#+\s*)?
         (?:
           (?:\d+(?:\.\d+)*|{ROMAN})
           (?:\s*[\.\)\|\:\-\u2013\u2014]\s*|\s+)
           |
           [A-Z](?:\s*[\.\)\|\:\-\u2013\u2014]\s*|\s+)
         )?
         """,
    re.IGNORECASE | re.VERBOSE,
)

PAREN_ENUM_PREFIX = re.compile(
    r"""^\s*
        (?:\#+\s*)?
        \(\s*\d+(?:\.\d+)*\s*\)\s*
        """,
    re.IGNORECASE | re.VERBOSE,
)

ABSTRACT_PAT = re.compile(r"^(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)$", re.IGNORECASE)

METHODS_SECTIONS = [
    r"materials?\s*(?:&|and)\s*methods?\b.*$",
    r"materials?\s*(?:&|and)\s*(?:experimental\s+)?methods?$",
    r"materials?\s*(?:&|and)\s*methodolog(?:y|ies)$",
    r"methods?\s*(?:&|and)\s*materials?$",
    r"methods?$",
    r"methodolog(?:y|ies)$",
    r"methodolog(?:y|ies)\s*(?:&|and)\s*material\s+characterization$",
    r"experimental(?:\s+sections?)?$",
    r"experiment\s+sections?$",
    r"experimental\s+procedures?$",
    r"experimental\s*(?:and\s*)?methods?$",
    r"experimental\s+methods?\s*(?:&|and)\s*materials?$",
    r"experimental\s+materials?\s*(?:&|and)\s*methods?$",
    r"experimental\s*(?:and\s+\w+\s*)?methods?$",
    r"experimental\s+methodolog(?:y|ies)$",
    r"experimental\s+details?$",
    r"experimental\s+setup$",
    r"experimental\s+set\s+up$",
    r"experimental\s+apparatus$",
    r"measurement\s+apparatus$",
    r"analytical\s+methods?\s*(?:&|and)\s*calculations?$",
    r"chemicals?\s*(?:&|and)\s*(?:experimental\s+)?procedures?$",
    r"chemicals?\s*(?:&|and)\s*methods?$",
    r"experiments?$",
    r"approach$",
]

RESULTS_SECTIONS = [
    r"results?(?:\s+(?:and|&)\s+discussions?)?$",
    r"results?(?:\s+(?:and|&)\s+discussions?)?(?:\s+\d+(?:\.\d+)*\.?\s+results?(?:\s+(?:and|&)\s+discussions?)?)+$",
    r"results?\s+and\s+characterization$",
    r"experimental\s+results?(?:\s+(?:and|&)\s+discussions?)?$",
    r"discussion$",
]

CAPTION_CONTEXT_PREFIX_WORDS = ("Supplementary", "Supplemental", "Supporting", "Appendix")
CAPTION_CONTEXT_PREFIX_PAT = rf"(?:(?:{'|'.join(CAPTION_CONTEXT_PREFIX_WORDS)})\s+)?"
CAPTION_ROMAN_NUM_PAT = r"[IVXLCDM]+"
CAPTION_SUPP_NUM_PREFIX_PAT = r"(?:SM|SI|SF|ST|S|A)"
CAPTION_NUM_PAT = rf"{CAPTION_SUPP_NUM_PREFIX_PAT}\s*[\.\-]?\s*\d+|\d+\s*S|\d+|{CAPTION_ROMAN_NUM_PAT}"
TABLE_HEADING_PREFIXES = ("table ",) + tuple(
    f"{word.lower()} table " for word in CAPTION_CONTEXT_PREFIX_WORDS
)

END_SECTION_PAT = (
    r"^(?:\d+\s*[\.\)]\s*)?"
    r"(?:"
    r"conclusions?(?:\s*(?:and|&)\s*perspectives)?"
    r"|(?:environmental\s+)?implications?"
    r"|environmental\s+implications?"
    r")$"
)

EXCLUDED_SECTION_PATS = [
    r"Author\s*Information",
    r"Corresponding\s*Author",
    r"Notes?",
    r"Acknowledg(e)?ments?",
    r"References?",
    r"Funding",
    r"Data\s*Availability",
    r"Associated\s*Content",
    r"Declaration\s+of\s+Competing\s+Interests?",
    r"Competing\s+Interests?",
    r"Conflict\s+of\s+Interests?",
    r"Declaration\s+of\s+Interest",
    r"Credit\s+authorship\s+contribution\s+statement",
    r"CRediT\s+authorship\s+contribution\s+statement",
    r"Author(ship)?\s+contributions?",
    r"Contribution\s+statement",
    r"Ethics\s+statement",
    r"Consent\s+to\s+participate",
    r"Consent\s+for\s+publication",
    r"Availability\s+of\s+data\s+and\s+materials",
    r"Competing\s+financial\s+interests?",
]


def normalize_heading(line: str) -> str:
    txt = unicodedata.normalize("NFKC", line or "").strip()
    if txt.startswith("#"):
        txt = txt.lstrip("#").strip()

    txt = re.sub(r"\\([.\)\|\:\-\u2013\u2014])", r"\1", txt)

    # Docling/OCR can prepend equation-like numbering before real section labels,
    # e.g. "(3) 3. Results and discussion". Strip repeated numbering until stable.
    for _ in range(4):
        new_txt = PAREN_ENUM_PREFIX.sub("", txt).strip()
        new_txt = ENUM_PREFIX.sub("", new_txt).strip()
        if new_txt == txt:
            break
        txt = new_txt

    txt = re.sub(r"\s*\|\s*", " ", txt)
    txt = re.sub(
        r"^[^\x00-\x7F]+\s+(?=(abstract|materials?|methods?|results?|discussion)\b)",
        "",
        txt,
        flags=re.IGNORECASE,
    )
    txt = re.sub(r"^[^\w]+", "", txt, flags=re.UNICODE)
    txt = re.sub(r"\s+", " ", txt.lower()).strip()
    txt = re.sub(r"[^\w\s]+$", "", txt, flags=re.UNICODE).strip()
    if txt.replace(" ", "") == "abstract":
        txt = "abstract"
    return txt


def hit_any_section(patterns: Iterable[str], norm_hdr: str) -> bool:
    return any(re.fullmatch(p, norm_hdr or "", re.IGNORECASE) for p in patterns)


def is_abstract_heading(norm_hdr: str) -> bool:
    return bool(ABSTRACT_PAT.fullmatch(norm_hdr or ""))


def is_methods_heading(norm_hdr: str) -> bool:
    return hit_any_section(METHODS_SECTIONS, norm_hdr)


def is_results_heading(norm_hdr: str) -> bool:
    return hit_any_section(RESULTS_SECTIONS, norm_hdr)


def classify_heading(raw_heading: str) -> Optional[str]:
    norm = normalize_heading(raw_heading)
    if is_abstract_heading(norm):
        return ABSTRACT_LABEL
    if is_methods_heading(norm):
        return METHODS_LABEL
    if is_results_heading(norm):
        return RESULTS_LABEL
    return None


def leading_major_section_number(raw_heading: str) -> Optional[int]:
    m_num = re.match(r"^\s*(?:\(\s*)?(\d+)(?:\.\d+)*\.?\s+", raw_heading or "")
    if not m_num:
        return None
    try:
        return int(m_num.group(1))
    except ValueError:
        return None


def is_table_heading_text(norm_hdr: str) -> bool:
    norm_hdr = re.sub(
        rf"^\d{{1,6}}\s+(?={CAPTION_CONTEXT_PREFIX_PAT}table\b)",
        "",
        norm_hdr or "",
        flags=re.IGNORECASE,
    )
    return bool(
        re.match(
            rf"^{CAPTION_CONTEXT_PREFIX_PAT}table\s*(?:[-\u2013\u2014]\s*)?(?:{CAPTION_NUM_PAT})(?:\s*[_\.-]\s*\d+)?(?:\s+continued)?$",
            norm_hdr,
            re.IGNORECASE,
        )
    ) or norm_hdr.startswith(TABLE_HEADING_PREFIXES)


def is_pre_methods_results_candidate(norm_hdr: str) -> bool:
    if not norm_hdr:
        return False
    if is_methods_heading(norm_hdr) or is_results_heading(norm_hdr) or is_abstract_heading(norm_hdr):
        return False
    if re.search(END_SECTION_PAT, norm_hdr, re.IGNORECASE):
        return False
    if any(re.search(p, norm_hdr, re.IGNORECASE) for p in EXCLUDED_SECTION_PATS):
        return False
    if is_table_heading_text(norm_hdr):
        return False
    if re.fullmatch(
        r"(?:abstract|keywords?|article|a\s*r\s*t\s*i\s*c\s*l\s*e\s*i\s*n\s*f\s*o|introduction)",
        norm_hdr,
        flags=re.IGNORECASE,
    ):
        return False
    if norm_hdr.startswith("journal of ") or norm_hdr in {
        "nature water",
        "contents lists available at sciencedirect",
        "journal homepage",
    }:
        return False
    return True


def find_pre_methods_results_window(lines: Sequence[str]) -> Tuple[Optional[int], Optional[int]]:
    method_idx = None
    explicit_results = False
    candidate_idxs = []
    substantive_lines = 0

    for scan_idx, raw_line in enumerate(lines):
        stripped = (raw_line or "").strip()
        h = re.match(r"^(#+)\s+(.*)", stripped)
        if h:
            norm = normalize_heading(h.group(2).strip())
            if is_results_heading(norm):
                explicit_results = True
            if is_methods_heading(norm):
                method_idx = scan_idx
                break
            if substantive_lines >= 4 and is_pre_methods_results_candidate(norm):
                candidate_idxs.append(scan_idx)
            continue

        if not stripped:
            continue
        if len(re.sub(r"\W+", "", stripped, flags=re.UNICODE)) >= 80:
            substantive_lines += 1

    if method_idx is None or explicit_results or len(candidate_idxs) < 2:
        return None, None
    return candidate_idxs[0], method_idx
