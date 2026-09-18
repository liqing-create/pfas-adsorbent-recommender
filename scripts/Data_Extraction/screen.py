# scripts/screen.py
import os
import re
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Set

SCRIPT_DIR = Path(__file__).resolve().parent
AD_ROOT = SCRIPT_DIR.parent.parent
if str(AD_ROOT) not in sys.path:
    sys.path.insert(0, str(AD_ROOT))

from chains.llm_config import get_llm
from chains.llm_usage import predict_with_usage
from chains.screen_chain import create_screen_chain

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for lib in ("httpx", "openai", "httpcore", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ── Config (edit here) ─────────────────────────────────────────────
WOS_BASE_DIR = AD_ROOT.parent / "web of science search"
INPUT_XLSX = str(WOS_BASE_DIR / "wos_04282026.xlsx")
OUTPUT_CSV = str(WOS_BASE_DIR / "wos_04282026_screened.csv")
PROMPTS_DIR = str(AD_ROOT / "prompts")

# Model selection knobs (same style as classify.py)
LLM_PROVIDER   = "together"  # "openai" or "together"
OPENAI_MODEL   = "gpt-4.1-2025-04-14"
TOGETHER_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

LLM_MODELS = {
    "openai": OPENAI_MODEL,
    "together": TOGETHER_MODEL,
}

# Columns to keep from source
KEEP_ORIGINAL_FIELDS = [
    "Authors", "Article Title", "Source Title", "Author Keywords", "Keywords Plus",
    "Abstract", "Publication Year", "DOI Link"
]

# ── Helpers ────────────────────────────────────────────────────────
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).encode("utf-8", "ignore").decode("utf-8", "ignore")

def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}

    # Make DOI Link if missing
    if "doi link" not in df.columns:
        doi_col = lower.get("doi")
        if doi_col:
            df["DOI Link"] = df[doi_col].map(lambda d: f"https://doi.org/{str(d).strip()}" if d and str(d).strip() else "")
        else:
            df["DOI Link"] = ""

    # Ensure standard casing for our kept columns
    canonical = {
        "authors": "Authors",
        "article title": "Article Title",
        "source title": "Source Title",
        "author keywords": "Author Keywords",
        "keywords plus": "Keywords Plus",
        "abstract": "Abstract",
        "publication year": "Publication Year",
        "doi link": "DOI Link",
    }
    for low, proper in canonical.items():
        if proper not in df.columns and low in lower:
            df[proper] = df[lower[low]]

    # Normalize to strings (preserve original casing for output)
    for c in KEEP_ORIGINAL_FIELDS:
        if c in df.columns:
            df[c] = df[c].map(_safe_str)

    return df

def _normalize_yn(s: str) -> str:
    """
    Coerce model output to strict 'Y' or 'N'.
    """
    if not s:
        return "N"
    s = s.strip().lower()
    if s.startswith("y") or s == "yes":
        return "Y"
    if s.startswith("n") or s == "no":
        return "N"
    m = re.search(r"\b(y|n)\b", s)
    return "Y" if (m and m.group(1) == "y") else "N"

def _block(title: str, content: str, preview_chars: int, no_truncate: bool):
    print(f"\n--- {title} ---")
    if content is None:
        print("(None)")
        return
    if no_truncate or len(content) <= preview_chars:
        print(content)
    else:
        head = content[:preview_chars]
        tail = content[-min(preview_chars//2, 400):] if len(content) > 2*preview_chars else ""
        print(head + ("\n...\n" + tail if tail else ""))

def _output_columns() -> List[str]:
    return KEEP_ORIGINAL_FIELDS + [
        "_source_row", "_text_all", "_screen_YN",
        "tokens_prompt", "tokens_completion", "tokens_total",
        "_rt_seconds",
    ]

def _append_record_csv(record: Dict[str, Any], output_path: str, columns: List[str]) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    pd.DataFrame([record]).reindex(columns=columns).to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )

# ── Main ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Abstract-level Y/N screening for PFAS adsorption papers.")
    ap.add_argument("-i", "--input",  default=INPUT_XLSX,  help="Path to Web of Science .xlsx")
    ap.add_argument("-o", "--output", default=OUTPUT_CSV, help="Path to output .csv")
    ap.add_argument("-p", "--prompts", default=PROMPTS_DIR, help="Directory containing screen.j2")
    ap.add_argument("--overwrite", action="store_true", help="Start fresh by deleting any existing output CSV")
    ap.add_argument("--provider", choices=sorted(LLM_MODELS), default=LLM_PROVIDER, help="LLM provider")
    ap.add_argument("--openai-model", default=OPENAI_MODEL, help="OpenAI model name")
    ap.add_argument("--together-model", default=TOGETHER_MODEL, help="Together AI model name")
    ap.add_argument("--verbose", action="store_true", help="Print prompt, abstract, raw response, etc. per row")
    ap.add_argument("--preview-chars", type=int, default=800, help="Chars to show for each printed block")
    ap.add_argument("--no-truncate", action="store_true", help="Print full prompt/abstract/response (no truncation)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.prompts):
        raise FileNotFoundError(f"Prompts directory not found: {args.prompts}")
    tpl_path = os.path.join(args.prompts, "screen.j2")
    if not os.path.exists(tpl_path):
        raise FileNotFoundError(f"Template not found at: {tpl_path}")
    if not args.output.lower().endswith(".csv"):
        raise ValueError(f"Output must be a .csv path for incremental writes: {args.output}")
    model_by_provider = {
        "openai": args.openai_model,
        "together": args.together_model,
    }
    model_name = model_by_provider[args.provider]
    llm = get_llm(provider=args.provider, model_name=model_name)

    # Build chain
    chain = create_screen_chain(args.prompts, llm)

    # Load and prep data
    df = pd.read_excel(args.input, sheet_name=0)
    df = _ensure_canonical_columns(df)
    output_cols = _output_columns()
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    processed_rows: Set[int] = set()
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        existing = pd.read_csv(args.output)
        if "_source_row" not in existing.columns:
            raise ValueError(
                f"Existing output CSV lacks _source_row, so it cannot be safely resumed: {args.output}"
            )
        processed_rows = set(
            pd.to_numeric(existing["_source_row"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        print(f"Resuming from existing CSV: {len(processed_rows)} rows already written")
    total = len(df)
    print(f"Screening {total} records with model='{model_name}' (provider={args.provider})")

    for row_num, (_, row) in enumerate(df.iterrows(), start=1):
        if row_num in processed_rows:
            continue

        title = row.get("Article Title", "")
        abstract = _safe_str(row.get("Abstract", ""))

        # Render the exact prompt that will be sent
        rendered_prompt = chain.prompt.format_prompt(abstract=abstract).to_string()

        if args.verbose:
            print(f"\n=== Row {row_num}/{total} ===")
            print(f"Title: {title}")
            _block("Rendered Prompt", rendered_prompt, args.preview_chars, args.no_truncate)
            _block("Fed Abstract", abstract, args.preview_chars, args.no_truncate)
        call = predict_with_usage(
            chain,
            provider=args.provider,
            model_name=model_name,
            abstract=abstract,
        )
        raw = call.text
        usage = call.usage
        rt = usage.elapsed_s

        label = _normalize_yn(raw)

        if args.verbose:
            _block("Raw LLM Response", raw, args.preview_chars, args.no_truncate)
            print(f"Normalized Label: {label}")
            print(f"Response Time: {rt:.3f}s")
            print(
                f"Tokens → prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, total={usage.total_tokens}"
            )
        rec = {
            "Authors":           row.get("Authors", ""),
            "Article Title":     title,
            "Source Title":      row.get("Source Title", ""),
            "Author Keywords":   row.get("Author Keywords", ""),
            "Keywords Plus":     row.get("Keywords Plus", ""),
            "Abstract":          abstract,
            "Publication Year":  row.get("Publication Year", ""),
            "DOI Link":          row.get("DOI Link", ""),
            "_source_row":       row_num,
            "_text_all":         " ".join([title, abstract, row.get("Author Keywords",""), row.get("Keywords Plus","")]).strip(),
            "_screen_YN":        label,
            "_rt_seconds":       round(rt, 3),
        }
        rec.update(usage.flat_token_dict())
        _append_record_csv(rec, args.output, output_cols)
        processed_rows.add(row_num)
        print(f"Saved row {row_num}/{total} ({label})")

    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        out = pd.read_csv(args.output)
        summary = out["_screen_YN"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
        print("\nSummary:")
        print(summary.to_string(index=False))
        print(f"\nSaved CSV incrementally: {args.output}")
    else:
        print("\nNo rows were written.")

if __name__ == "__main__":
    main()
