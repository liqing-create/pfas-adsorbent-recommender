# post_process_enumeration.py
# Post-process four-slot Performance_id Differentiating_Condition:
#  - Accept only PFAS|Adsorbent|Test_mode|Differentiating_Condition.
#  - Preserve every enumerated Differentiating_Condition token. A condition can
#    be shared within a table chunk yet distinguish records elsewhere in a study.
#  - Reorder tokens to canonical order.
#  - Preserve PFAS and Adsorbent EXACTLY; canonicalize Test_mode to lower-case; only modify the Differentiating_Condition.
#  - Fix stray unit pipes inside condition (e.g., "_mg|L" -> "_mg/L").
#  - Preserve zero-concentration control conditions (for example,
#    Ca^2+_0_mM and NaCl_0_mM) because they identify an experimental baseline.
#  - Save changes in-place; if a file changes, backup original to BACKUP_DIR.

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Set, Any

try:
    from ..workflow_config import iter_active_study_paths
except ImportError:  # Supports ``python post_process_enumeration.py``.
    import sys

    _DATA_EXTRACTION_DIR = Path(__file__).resolve().parents[1]
    if str(_DATA_EXTRACTION_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_EXTRACTION_DIR))
    from workflow_config import iter_active_study_paths

# ───────────────────────────────────────────────────────────
# 🔧 CONFIG
# ───────────────────────────────────────────────────────────

RECURSIVE = False
DRY_RUN = True
VERBOSE = True

CATEGORY_ORDER = [
    "pH",
    "Time",
    "PFAS_initial_concentration",
    "Adsorbent_dosage",
    "ionic_strength",
    "coexisting_inorganic_matter",
    "coexisting_organic_matter",
    "water_matrix",
    "TOC_or_DOC",
]

VALID_TEST_MODES = {"kinetic", "isotherm", "snapshot"}
ADSORBENT_SCOPE_KEYS = ("in_scope", "out_of_scope")
ADSORBENT_IDENTITY_FIELDS = (
    "Adsorbent_id",
    "Name_Abbreviation",
    "Name_Full",
    "Name_Commercial",
)

# ───────────────────────────────────────────────────────────
# 🧠 Helpers
# ───────────────────────────────────────────────────────────

UNIT_PIPE_FIX = re.compile(r"_(mg|μg|ug|ng|pg|g)\|L\b", flags=re.IGNORECASE)
CONDITION_SEPARATOR = re.compile(r"\s*(?:;|&&)\s*")

def fix_unit_pipes(cond: str) -> str:
    """Convert things like '_mg|L' -> '_mg/L' inside Differentiating_Condition."""
    return UNIT_PIPE_FIX.sub(lambda m: f"_{m.group(1)}/L", cond)

def split_performance_id(pid: str) -> Tuple[str, str, str, str] | None:
    """Return four-slot Performance_id parts, or None for invalid/legacy IDs."""
    text = fix_unit_pipes(str(pid or "").strip())
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 4 or any(not part for part in parts):
        return None
    pfas, ads, test_mode, cond = parts
    test_mode = test_mode.casefold()
    if test_mode not in VALID_TEST_MODES:
        return None
    return pfas, ads, test_mode, cond

def tokenize_differentiating_condition(cond: str) -> List[str]:
    if not cond:
        return []
    # Fix unit pipes before splitting tokens
    cond = fix_unit_pipes(cond)
    return [t.strip() for t in CONDITION_SEPARATOR.split(cond) if t.strip()]

def join_differentiating_condition(tokens: Iterable[str]) -> str:
    return "; ".join(t.strip() for t in tokens if t and t.strip())


def iter_data_source_values(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values: List[str] = []
        for item in value:
            values.extend(iter_data_source_values(item))
        return values
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"\s*(?:;|,)\s*", text) if part.strip()]


def merge_record_data_source(target: Dict[str, object], source: Dict[str, object]) -> None:
    merged: List[str] = []
    seen: Set[str] = set()
    for value in (
        target.get("Data_Source"),
        target.get("data_source"),
        source.get("Data_Source"),
        source.get("data_source"),
    ):
        for item in iter_data_source_values(value):
            key = item.casefold()
            if key not in seen:
                seen.add(key)
                merged.append(item)
    if merged:
        target["Data_Source"] = ", ".join(merged)


def identity_match_keys(value: Any) -> Set[str]:
    text = str(value or "").strip().strip("'\"`")
    if not text:
        return set()
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    text = re.sub(r"\s+", " ", text).strip()
    folded = text.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    return {
        key
        for key in (folded, compact)
        if key and key not in {"na", "none", "null"}
    }


def normalize_adsorbent_groups(raw: Any) -> Dict[str, List[Dict[str, object]]]:
    if isinstance(raw, list):
        return {"in_scope": raw, "out_of_scope": []}
    if isinstance(raw, dict):
        return {
            key: raw.get(key) if isinstance(raw.get(key), list) else []
            for key in ADSORBENT_SCOPE_KEYS
        }
    return {"in_scope": [], "out_of_scope": []}


def adsorbent_identity_keys(
    payload: Dict[str, object],
    scope_keys: Tuple[str, ...] = ADSORBENT_SCOPE_KEYS,
) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    groups = normalize_adsorbent_groups(payload.get("adsorbents"))
    for scope_key in scope_keys:
        for item in groups[scope_key]:
            if not isinstance(item, dict):
                continue
            adsorbent_id = str(item.get("Adsorbent_id") or "").strip()
            for field in ADSORBENT_IDENTITY_FIELDS:
                value = str(item.get(field) or "").strip()
                if not value:
                    continue
                display = f"{field}={value}"
                if adsorbent_id and field != "Adsorbent_id":
                    display = f"{display} (Adsorbent_id={adsorbent_id})"
                for key in identity_match_keys(value):
                    keys.setdefault(key, display)
    return keys


def study_id_from_output_path(fp: Path) -> str:
    stem = fp.stem
    for suffix in ("_enumerated", "_performance"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def load_adsorbent_identity_keys_for_file(fp: Path) -> Dict[str, str]:
    study_id = study_id_from_output_path(fp)
    artifact_name = f"{study_id}_adsorbent.json"
    candidates = [
        fp.parent / "_chain_artifacts" / study_id / artifact_name,
    ]
    if fp.parent.name == study_id and fp.parent.parent.name == "_chain_artifacts":
        candidates.insert(0, fp.parent / artifact_name)
    artifact_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if artifact_path is None:
        return {}
    try:
        with artifact_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        if VERBOSE:
            print(f"  - adsorbent guard unavailable for {fp.name}: {e}")
        return {}
    if not isinstance(payload, dict):
        return {}
    return adsorbent_identity_keys(payload)


def load_adsorbent_identity_key_maps_for_file(
    fp: Path,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    study_id = study_id_from_output_path(fp)
    artifact_name = f"{study_id}_adsorbent.json"
    candidates = [
        fp.parent / "_chain_artifacts" / study_id / artifact_name,
    ]
    if fp.parent.name == study_id and fp.parent.parent.name == "_chain_artifacts":
        candidates.insert(0, fp.parent / artifact_name)
    artifact_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if artifact_path is None:
        return {}, {}
    try:
        with artifact_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        if VERBOSE:
            print(f"  - adsorbent guard unavailable for {fp.name}: {e}")
        return {}, {}
    if not isinstance(payload, dict):
        return {}, {}
    return (
        adsorbent_identity_keys(payload),
        adsorbent_identity_keys(payload, scope_keys=("out_of_scope",)),
    )


def adsorbent_match_for_slot(
    value: str,
    adsorbent_keys: Dict[str, str] | None,
) -> str:
    if not adsorbent_keys:
        return ""
    for key in identity_match_keys(value):
        if key in adsorbent_keys:
            return adsorbent_keys[key]
    return ""


def find_adsorbent_in_pfas_slot(
    records: List[Dict[str, object]],
    adsorbent_identity_key_map: Dict[str, str] | None,
) -> List[Dict[str, str]]:
    if not adsorbent_identity_key_map:
        return []
    violations: List[Dict[str, str]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        performance_id = str(rec.get("Performance_id") or "").strip()
        parts = [part.strip() for part in performance_id.split("|")]
        if len(parts) != 4:
            continue
        pfas_slot = parts[0]
        matched = adsorbent_match_for_slot(pfas_slot, adsorbent_identity_key_map)
        if matched:
            violations.append(
                {
                    "Performance_id": performance_id,
                    "PFAS_slot": pfas_slot,
                    "matched_adsorbent": matched,
                }
            )
    return violations


def find_out_of_scope_adsorbent_slot(
    records: List[Dict[str, object]],
    out_of_scope_adsorbent_identity_key_map: Dict[str, str] | None,
) -> List[Dict[str, str]]:
    if not out_of_scope_adsorbent_identity_key_map:
        return []
    violations: List[Dict[str, str]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        performance_id = str(rec.get("Performance_id") or "").strip()
        parts = [part.strip() for part in performance_id.split("|")]
        if len(parts) != 4:
            continue
        adsorbent_slot = parts[1]
        matched = adsorbent_match_for_slot(
            adsorbent_slot,
            out_of_scope_adsorbent_identity_key_map,
        )
        if matched:
            violations.append(
                {
                    "Performance_id": performance_id,
                    "Adsorbent_slot": adsorbent_slot,
                    "matched_adsorbent": matched,
                }
            )
    return violations


def remove_performance_id_records(
    records: List[Dict[str, object]],
    violations: List[Dict[str, str]],
) -> List[str]:
    invalid_ids = {
        str(violation.get("Performance_id") or "").strip().casefold()
        for violation in violations
    }
    removed: List[str] = []
    kept: List[Dict[str, object]] = []
    for rec in records:
        performance_id = str((rec or {}).get("Performance_id") or "").strip()
        if performance_id.casefold() in invalid_ids:
            removed.append(performance_id)
        else:
            kept.append(rec)
    records[:] = kept
    return removed


# Ordering heuristics
INORG_KEYWORDS = {
    "chloride","sodium","magnesium","calcium","nitrate","sulfate","phosphate",
    "fluoride","bicarbonate","carbonate","aluminum","iron","ferric","ferrous",
    "potassium","sulfide","bromide","iodide","ammonium","ammonia","silica"
}
ORG_KEYWORDS = {
    "humic","fulvic","nom","srfa","ethanol","methanol","acetate","tce",
    "dioxane","suva","et","ha"
}

def is_pfasc_init_conc(t: str) -> bool:
    # <PFASname>_<value>_<unit>/L (rough, case/μ tolerant)
    return bool(re.match(r"^[^_]+_\d+(?:\.\d+)?_?(?:μg|ug|ng|mg|pg)/l$", t.lower()))

def is_adsorbent_dosage(t: str) -> bool:
    tl = t.lower()
    if tl.startswith(("time_", "ph_", "doc_", "toc_", "ionic_strength_")):
        return False
    return bool(re.search(r"_\d+(?:\.\d+)?_?(?:μg|ug|ng|mg|g)/l$", tl))

def token_category(token: str) -> str:
    tl = token.lower()
    if tl.startswith("ph_"): return "pH"
    if tl.startswith("time_"): return "Time"
    if is_pfasc_init_conc(token): return "PFAS_initial_concentration"
    if is_adsorbent_dosage(token): return "Adsorbent_dosage"
    if tl.startswith("ionic_strength_") or re.search(r"_(?:mm|mol/l)$", tl): return "ionic_strength"
    if tl.startswith("toc_") or tl.startswith("doc_"): return "TOC_or_DOC"
    if any(kw in tl for kw in INORG_KEYWORDS): return "coexisting_inorganic_matter"
    if any(kw in tl for kw in ORG_KEYWORDS): return "coexisting_organic_matter"
    if "_" not in token: return "water_matrix"
    return "unknown"

def token_sort_key(token: str):
    try:
        idx = CATEGORY_ORDER.index(token_category(token))
    except ValueError:
        idx = len(CATEGORY_ORDER)
    return (idx, token.lower())

def reorder_tokens(tokens: List[str]) -> List[str]:
    return sorted(tokens, key=token_sort_key)

# ───────────────────────────────────────────────────────────
# 🔁 Core processing
# ───────────────────────────────────────────────────────────
def process_performance_records(
    records: List[Dict[str, object]],
    adsorbent_identity_key_map: Dict[str, str] | None = None,
    out_of_scope_adsorbent_identity_key_map: Dict[str, str] | None = None,
) -> Tuple[bool, List[str], List[str]]:
    normalized_records: List[Tuple[Dict[str, object], str, str, str, List[str]]] = []
    invalid_removed: List[str] = []
    pre_normalized_seen: Dict[str, Dict[str, object]] = {}
    changed = False
    pfas_slot_violations = {
        str(violation.get("Performance_id") or "").strip().casefold(): violation
        for violation in find_adsorbent_in_pfas_slot(records, adsorbent_identity_key_map)
    }
    out_of_scope_violations = {
        str(violation.get("Performance_id") or "").strip().casefold(): violation
        for violation in find_out_of_scope_adsorbent_slot(
            records,
            out_of_scope_adsorbent_identity_key_map,
        )
    }

    for rec in records:
        pid = str(rec.get("Performance_id", ""))
        parsed = split_performance_id(pid)
        if parsed is None:
            if pid.strip():
                invalid_removed.append(pid)
            changed = True
            continue
        pfas, ads, test_mode, cond = parsed
        pfas = pfas.strip()
        ads = ads.strip()
        pid_key = pid.strip().casefold()
        if pid_key in pfas_slot_violations:
            violation = pfas_slot_violations[pid_key]
            invalid_removed.append(
                f"{pid} [PFAS slot matches {violation['matched_adsorbent']}]"
            )
            changed = True
            continue
        if pid_key in out_of_scope_violations:
            violation = out_of_scope_violations[pid_key]
            invalid_removed.append(
                f"{pid} [Adsorbent slot matches out-of-scope {violation['matched_adsorbent']}]"
            )
            changed = True
            continue
        toks = tokenize_differentiating_condition(cond)
        normalized_condition = join_differentiating_condition(reorder_tokens(toks)) if toks else "NA"
        normalized_pid_key = f"{pfas}|{ads}|{test_mode}|{normalized_condition}".strip().lower()
        if normalized_pid_key in pre_normalized_seen:
            merge_record_data_source(pre_normalized_seen[normalized_pid_key], rec)
            changed = True
            continue
        pre_normalized_seen[normalized_pid_key] = rec
        normalized_records.append((rec, pfas, ads, test_mode, toks))
    if not normalized_records:
        records[:] = []
        return changed, [], invalid_removed 
    removed_tokens: Set[str] = set()
    cleaned_records: List[Dict[str, object]] = []
    for rec, pfas, ads, test_mode, toks in normalized_records:
        new_tokens = reorder_tokens(toks)
        new_cond = join_differentiating_condition(new_tokens) if new_tokens else "NA"
        new_pid = f"{pfas}|{ads}|{test_mode}|{new_cond}"
        if new_pid != rec.get("Performance_id"):
            rec["Performance_id"] = new_pid
            changed = True
        cleaned_records.append(rec)

    deduped: List[Dict[str, object]] = []
    seen: Set[str] = set()
    deduped_by_key: Dict[str, Dict[str, object]] = {}
    for rec in cleaned_records:
        pid_key = str(rec.get("Performance_id", "")).strip().lower()
        if not pid_key or pid_key in seen:
            if pid_key:
                merge_record_data_source(deduped_by_key[pid_key], rec)
            changed = True
            continue
        seen.add(pid_key)
        deduped_by_key[pid_key] = rec
        deduped.append(rec)
    records[:] = deduped
    return changed, sorted(removed_tokens), invalid_removed

def process_file_payload(
    payload: list,
    adsorbent_identity_key_map: Dict[str, str] | None = None,
    out_of_scope_adsorbent_identity_key_map: Dict[str, str] | None = None,
) -> Tuple[list, bool, List[Dict[str, object]]]:
    new_payload = deepcopy(payload)
    changed_any = False
    audit: List[Dict[str, object]] = []

    for chunk in new_payload:
        enums = chunk.get("enumerations", [])
        for enum in enums:
            if enum.get("task") != "performance":
                continue
            recs = enum.get("records", [])
            count_updated = False
            block_changed, removed_conditions, invalid_removed = process_performance_records(
                recs,
                adsorbent_identity_key_map,
                out_of_scope_adsorbent_identity_key_map,
            )
            new_count = len(recs)
            if enum.get("extracted_count") != new_count:
                enum["extracted_count"] = new_count
                count_updated = True
            if "total_count" in enum and enum.get("total_count") != new_count:
                enum["total_count"] = new_count
                count_updated = True
            if block_changed or count_updated:
                changed_any = True
                audit.append({
                    "study_folder": chunk.get("study_folder"),
                    "chunk_id": chunk.get("chunk_id"),
                    "removed_conditions": removed_conditions,
                    "invalid_removed": invalid_removed,
                    "count_updated": count_updated,
                    "num_records": len(recs),
                })
    return new_payload, changed_any, audit


def count_performance_records(outputs: List[Dict[str, object]]) -> int:
    total = 0
    for chunk in outputs:
        if not isinstance(chunk, dict):
            continue
        for enum in chunk.get("enumerations", []):
            if isinstance(enum, dict) and enum.get("task") == "performance":
                records = enum.get("records", [])
                if isinstance(records, list):
                    total += len(records)
    return total


def process_payload(
    payload,
    adsorbent_identity_key_map: Dict[str, str] | None = None,
    out_of_scope_adsorbent_identity_key_map: Dict[str, str] | None = None,
) -> Tuple[object, bool, List[Dict[str, object]]]:
    if isinstance(payload, list):
        return process_file_payload(
            payload,
            adsorbent_identity_key_map,
            out_of_scope_adsorbent_identity_key_map,
        )
    if isinstance(payload, dict) and isinstance(payload.get("outputs"), list):
        new_payload = deepcopy(payload)
        outputs, changed, audit = process_file_payload(
            new_payload["outputs"],
            adsorbent_identity_key_map,
            out_of_scope_adsorbent_identity_key_map,
        )
        new_payload["outputs"] = outputs
        if changed:
            new_payload["record_count"] = count_performance_records(outputs)
            new_payload["output_count"] = len(outputs)
        return new_payload, changed, audit
    return payload, False, []

# ───────────────────────────────────────────────────────────
# 🏃 Main
# ───────────────────────────────────────────────────────────

def find_input_files() -> List[Path]:
    """Return only the selected studies' canonical performance artifacts."""
    return [
        paths.enumeration_task_artifact("performance")
        for paths in iter_active_study_paths()
        if paths.enumeration_task_artifact("performance").exists()
    ]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    files = find_input_files()
    if VERBOSE:
        print(f"Found {len(files)} selected enumeration artifact(s).")

    changed_count = 0
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"⚠️  Skip (failed to read JSON): {fp} → {e}")
            continue

        if not (
            isinstance(payload, list)
            or (isinstance(payload, dict) and isinstance(payload.get("outputs"), list))
        ):
            if VERBOSE:
                print(f"• Not a supported payload, skipping: {fp.name}")
            continue

        (
            adsorbent_identity_key_map,
            out_of_scope_adsorbent_identity_key_map,
        ) = load_adsorbent_identity_key_maps_for_file(fp)
        if VERBOSE and adsorbent_identity_key_map:
            print(f"  - adsorbent guard loaded {len(adsorbent_identity_key_map)} key(s) for {fp.name}")

        new_payload, is_changed, audit = process_payload(
            payload,
            adsorbent_identity_key_map,
            out_of_scope_adsorbent_identity_key_map,
        )

        if not is_changed:
            if VERBOSE:
                print(f"✓ No change: {fp.name}")
            continue

        changed_count += 1
        print(f"\n✳ Changed: {fp.name}")
        for row in audit:
            print(f"  - chunk {row['chunk_id']}: removed conditions {row['removed_conditions']} (n={row['num_records']})")
            if row.get("invalid_removed"):
                print(f"    removed invalid IDs: {row['invalid_removed']}")
        if DRY_RUN:
            print("  [dry-run] would backup + overwrite file")
            continue

        try:
            backup_dir = fp.parent / "Preprocess_Backups"
            ensure_dir(backup_dir)
            backup_path = backup_dir / fp.name
            shutil.copy2(fp, backup_path)
            if VERBOSE:
                print(f"  → backup saved: {backup_path}")
        except Exception as e:
            print(f"  ⚠️ backup failed: {e} (continuing)")

        try:
            with fp.open("w", encoding="utf-8") as f:
                json.dump(new_payload, f, ensure_ascii=False, indent=2)
            print(f"  → updated file written: {fp}")
        except Exception as e:
            print(f"  ❌ failed to write updated file: {fp} → {e}")

    print(f"\nDone. {changed_count} file(s) modified.")

if __name__ == "__main__":
    main()
