# scripts/classify.py

import os
import json
import logging
import pandas as pd
from typing import List, Dict

try:
    from .workflow_config import (
        ACTIVE_STUDIES,
        CHUNKED_DATA_DIR,
        OVERWRITE_EXISTING_OUTPUTS,
        PROMPTS_DIR,
        STUDY_RECORDS_XLSX,
        add_project_import_paths,
        check_stage_input,
        get_study_paths,
        iter_active_study_paths,
    )
except ImportError:  # Supports ``python classify.py`` from this folder.
    from workflow_config import (
        ACTIVE_STUDIES,
        CHUNKED_DATA_DIR,
        OVERWRITE_EXISTING_OUTPUTS,
        PROMPTS_DIR,
        STUDY_RECORDS_XLSX,
        add_project_import_paths,
        check_stage_input,
        get_study_paths,
        iter_active_study_paths,
    )

add_project_import_paths()

from chains.llm_config import get_llm
from chains.llm_usage import predict_with_usage
from chains.classify_chain import create_classification_chain

# ── Logging Configuration ───────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for lib in ("httpx", "openai", "httpcore", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ── Config ──────────────────────────────────────────────────
BASE_DIR = str(CHUNKED_DATA_DIR)
EXCEL_PATH = str(STUDY_RECORDS_XLSX)


LLM_PROVIDER   = "openai"  # "openai" or "together"

OPENAI_MODEL   = "gpt-4.1-2025-04-14"

TOGETHER_MODEL = os.getenv(
    "TOGETHER_MODEL",
    "google/gemma-4-31B-it",
)

def build_title_abstract_map(excel_path: str) -> Dict[str, str]:
    """
    Build mapping: 'study_06' -> 'Title: ...\\nAbstract: ...'
    from within_scope_records.xlsx columns: study_no, Article Title, Abstract.
    """
    df = pd.read_excel(excel_path)

    def _clean(x) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip()

    title_abs_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        study_no = row.get("study_no")
        if pd.isna(study_no):
            continue
        try:
            n = int(study_no)
        except Exception:
            continue

        title = _clean(row.get("Article Title"))
        abstract = _clean(row.get("Abstract"))
        title_abs = f"Title: {title}\nAbstract: {abstract}".strip()

        title_abs_map[f"study_{n}"] = title_abs
        title_abs_map[f"study_{n:02d}"] = title_abs

    return title_abs_map

def classify_study(
    study_id: str,
    chain,
    llm,
    title_abs_map: Dict[str, str],
    provider: str,
    model_name: str,
    *,
    chunks_override: List[Dict] | None = None,
    output_file_override=None,
    force_rerun: bool = False,
) -> List[Dict]:
    """
    Load chunked JSON for a single study, classify each chunk
    with the LLMChain, collect timing & token usage, and save results.
    """
    paths = get_study_paths(study_id)
    input_file = paths.chunked_input_file
    output_file = output_file_override or paths.classified_file
    title_abstract = title_abs_map.get(study_id, "")

    if output_file.exists() and not force_rerun and not OVERWRITE_EXISTING_OUTPUTS:
        with output_file.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
        print(f"[{study_id}] Reusing existing classification output: {output_file}")
        return existing

    if chunks_override is None:
        check_stage_input(paths, "classification")
        try:
            with input_file.open("r", encoding="utf-8") as fh:
                chunks = json.load(fh)
        except Exception as e:
            logging.error(f"Failed to load chunks for {study_id}: {e}")
            return []
    else:
        chunks = chunks_override

    results = []
    for chunk in chunks:
        section_label = (chunk.get("section_label") or "").strip().lower().replace(" ", "_")
        enriched_text = chunk.get("enriched_text", "").strip()

        if section_label == "abstract":
            results.append({
                "study_folder":          study_id,
                "file_type":             chunk.get("file_type"),
                "chunk_id":              chunk.get("chunk_id"),
                "section_label":         chunk.get("section_label"),
                "enriched_text":         enriched_text,
                "predicted_label":       None,
                "classification_time_s": 0.0,
                "tokens": {
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                    "total_tokens":      0,
                },
            })
            continue

        if not enriched_text:
            chunk["predicted_label"] = "Empty"
            results.append(chunk)
            continue

        # 1) Build & print the prompt so we can inspect it
        prompt_text = chain.prompt.format_prompt(
            title_abstract=title_abstract,
            enriched_text=enriched_text,
        ).to_string()
        print(f"\n[{study_id}] Prompt for Chunk {chunk.get('chunk_id')}:\n{prompt_text}\n")
        # 2) Call the chain and collect provider-agnostic usage metadata
        call = predict_with_usage(
            chain,
            provider=provider,
            model_name=model_name,
            title_abstract=title_abstract,
            enriched_text=enriched_text,
        )
        raw_response = call.text
        usage = call.usage
        label_str = raw_response.strip()

        # 3) Print out the raw LLM response
        print(f"[{study_id}] Raw response for Chunk {chunk.get('chunk_id')}:\n{raw_response}\n")

        # record outputs
        out = {
            "study_folder":          study_id,
            "file_type":             chunk.get("file_type"),
            "chunk_id":              chunk.get("chunk_id"),
            "section_label":         chunk.get("section_label"),
            "enriched_text":         enriched_text,
            "predicted_label":       label_str,
            "classification_time_s": round(usage.elapsed_s, 3),
            "tokens": usage.token_dict(),
        }
        results.append(out)
        print(
            f"[{study_id}] Chunk {chunk.get('chunk_id')} → {label_str} "
            f"(prompt_tokens={usage.prompt_tokens}, "
            f"completion_tokens={usage.completion_tokens}, "
            f"total_tokens={usage.total_tokens})"
        )




    # save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} classified chunks to {output_file}")

    return results


def main():
    # pick LLM
    model_name = OPENAI_MODEL if LLM_PROVIDER == "openai" else TOGETHER_MODEL
    print(f"Using {LLM_PROVIDER} model: {model_name}")
    llm = get_llm(provider=LLM_PROVIDER, model_name=model_name)
    # build our classification chain
    chain = create_classification_chain(PROMPTS_DIR, llm)
    title_abs_map = build_title_abstract_map(EXCEL_PATH)

    # process each study
    all_results = []
    all_results_by_dir: dict[object, list[Dict]] = {}
    for paths in iter_active_study_paths():
        sid = paths.study_id
        print(f"\n=== Classifying study: {sid} ===")
        res = classify_study(
            sid,
            chain,
            llm,
            title_abs_map,
            provider=LLM_PROVIDER,
            model_name=model_name,
        )
        all_results.extend(res)
        all_results_by_dir.setdefault(paths.classification_dir, []).extend(res)

    # Keep mixed active selections separate by their configured study group.
    for output_dir, group_results in all_results_by_dir.items():
        combined_file = output_dir / "all_studies_classified.json"
        with combined_file.open("w", encoding="utf-8") as fh:
            json.dump(group_results, fh, indent=2, ensure_ascii=False)
        print(f"\nGroup combined output saved to {combined_file}")


if __name__ == "__main__":
    main()
