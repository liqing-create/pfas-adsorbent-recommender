import copy
import html
import json
import os
import re

import streamlit as st

try:
    from .workflow_config import ACTIVE_STUDIES, get_study_paths
except ImportError:  # Streamlit executes this file directly.
    from workflow_config import ACTIVE_STUDIES, get_study_paths

try:
    from .classification_chunk_matching import (
        MIN_CONTENT_SIMILARITY,
        duplicate_normalized_content_count,
        match_chunks,
    )
except ImportError:
    from classification_chunk_matching import (
        MIN_CONTENT_SIMILARITY,
        duplicate_normalized_content_count,
        match_chunks,
    )


LABEL_OPTIONS = ["performance", "adsorbent", "experiment", "irrelevant"]
ANNOTATION_FIELDS = (
    "study_folder",
    "file_type",
    "chunk_id",
    "section_label",
    "enriched_text",
    "annotation",
)


def parse_predicted_labels(value):
    """Return labels as a clean list, accepting JSON lists and CSV strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(label).strip() for label in value if str(label).strip()]
    return [label.strip() for label in str(value).split(",") if label.strip()]


def parse_ground_truth_labels(value):
    """Normalize legacy empty GT annotations to the explicit negative label."""
    labels = parse_predicted_labels(value)
    return labels or ["irrelevant"]


def is_abstract_chunk(chunk):
    section_label = chunk.get("section_label")
    return isinstance(section_label, str) and section_label.strip().lower() == "abstract"


def format_chunk_text_for_display(text):
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n[ \t]*\n+", "\n", text)


def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def ground_truth_record_from_source(chunk, study_id):
    """Keep the GT schema small; model metadata belongs in classified.json."""
    record = {field: chunk.get(field) for field in ANNOTATION_FIELDS}
    if not record.get("study_folder"):
        record["study_folder"] = study_id
    record["annotation"] = parse_predicted_labels(chunk.get("predicted_label"))
    return record


def merge_classified_with_ground_truth(classified_chunks, existing_chunks, study_id):
    """
    Merge current classified chunks with an existing annotation file.

    Exact and high-confidence fuzzy content matches keep the existing GT
    annotation, while the current classified record supplies current metadata.
    Unmatched source content is new. Unmatched existing GT is retained.
    """
    matches = match_chunks(classified_chunks, existing_chunks)
    match_by_source = {match.source_index: match for match in matches}
    matched_existing_indices = {match.target_index for match in matches}
    records_by_id = {}
    source_predictions = {}
    origins = {}
    display_ids = []
    source_ids = []
    new_ids = []
    matched_source_ids_by_existing_index = {}
    exact_match_count = 0
    fuzzy_match_count = 0

    for source_index, source_record in enumerate(classified_chunks):
        if is_abstract_chunk(source_record):
            continue

        entry_id = f"source:{source_index}"
        match = match_by_source.get(source_index)

        record = ground_truth_record_from_source(source_record, study_id)
        if match is None:
            origins[entry_id] = "NEW"
            new_ids.append(entry_id)
        else:
            existing_index = match.target_index
            matched_source_ids_by_existing_index.setdefault(existing_index, []).append(
                entry_id
            )
            existing_annotation = existing_chunks[existing_index].get("annotation")
            record["annotation"] = parse_ground_truth_labels(existing_annotation)
            if match.method == "exact":
                exact_match_count += 1
                origins[entry_id] = "GT exact"
            elif match.method == "table_containment":
                fuzzy_match_count += 1
                origins[entry_id] = f"GT fuzzy {match.similarity:.0%}"
            else:
                fuzzy_match_count += 1
                origins[entry_id] = f"GT fuzzy {match.similarity:.0%}"

        records_by_id[entry_id] = record
        source_predictions[entry_id] = parse_predicted_labels(
            source_record.get("predicted_label")
        )
        display_ids.append(entry_id)
        source_ids.append(entry_id)

    gt_only_ids = []
    for existing_index, existing_record in enumerate(existing_chunks):
        if existing_index in matched_existing_indices or is_abstract_chunk(existing_record):
            continue
        entry_id = f"gt:{existing_index}"
        record = copy.deepcopy(existing_record)
        record["annotation"] = parse_ground_truth_labels(record.get("annotation"))
        records_by_id[entry_id] = record
        origins[entry_id] = "GT ONLY"
        display_ids.append(entry_id)
        gt_only_ids.append(entry_id)

    # Keep the original GT order and append only unmatched, non-abstract source
    # chunks. Matched entries use current source metadata but old GT labels.
    save_order = []
    for existing_index, existing_record in enumerate(existing_chunks):
        if existing_index in matched_source_ids_by_existing_index:
            save_order.extend(matched_source_ids_by_existing_index[existing_index])
        else:
            entry_id = f"gt:{existing_index}"
            if entry_id not in records_by_id:
                records_by_id[entry_id] = copy.deepcopy(existing_record)
            save_order.append(entry_id)
    save_order.extend(entry_id for entry_id in source_ids if entry_id in new_ids)

    return {
        "records_by_id": records_by_id,
        "source_predictions": source_predictions,
        "origins": origins,
        "display_ids": display_ids,
        "save_order": save_order,
        "new_ids": new_ids,
        "gt_only_ids": gt_only_ids,
        "source_abstract_count": sum(
            1 for record in classified_chunks if is_abstract_chunk(record)
        ),
        "matched_count": len(source_ids) - len(new_ids),
        "exact_match_count": exact_match_count,
        "fuzzy_match_count": fuzzy_match_count,
        "duplicate_existing": duplicate_normalized_content_count(existing_chunks),
        "duplicate_classified": duplicate_normalized_content_count(classified_chunks),
    }


def records_for_save(records_by_id, save_order):
    """Return merged GT records with UI-only state removed."""
    output = []
    for entry_id in save_order:
        record = copy.deepcopy(records_by_id[entry_id])
        if not is_abstract_chunk(record):
            record["annotation"] = parse_predicted_labels(record.get("annotation"))
        output.append(record)
    return output


def has_conflicting_labels(annotation):
    labels = set(parse_predicted_labels(annotation))
    return "irrelevant" in labels and bool(
        labels & (set(LABEL_OPTIONS) - {"irrelevant"})
    )


def main():
    st.title("LLM-Predicted Label Correction")

    study_choice = st.selectbox("Select Study ID:", ACTIVE_STUDIES)
    paths = get_study_paths(study_choice)
    input_file = paths.classified_file
    output_file = paths.classification_annotation_file

    if os.path.exists(output_file):
        st.info(
            f"Existing ground truth found at {output_file}. "
            "Loading will open it for revision and merge source chunks by content."
        )

    if st.button("Load / merge chunks"):
        if not os.path.exists(input_file):
            st.error(f"Could not find classified input: {input_file}")
            return
        try:
            classified_chunks = load_json_records(input_file)
            existing_chunks = (
                load_json_records(output_file) if os.path.exists(output_file) else []
            )
            merged = merge_classified_with_ground_truth(
                classified_chunks,
                existing_chunks,
                study_id=study_choice,
            )
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
            return

        if not merged["display_ids"]:
            st.warning("No non-abstract chunks found to annotate.")
            return

        load_nonce = st.session_state.get("annotation_load_nonce", 0) + 1
        st.session_state.annotation_load_nonce = load_nonce
        st.session_state.annotation_records = merged["records_by_id"]
        st.session_state.annotation_display_ids = merged["display_ids"]
        st.session_state.annotation_save_order = merged["save_order"]
        st.session_state.annotation_source_predictions = merged["source_predictions"]
        st.session_state.annotation_origins = merged["origins"]
        st.session_state.annotation_new_ids = set(merged["new_ids"])
        st.session_state.annotation_gt_only_ids = set(merged["gt_only_ids"])
        st.session_state.annotation_loaded_study = study_choice
        st.session_state.annotation_loaded_input = str(input_file)
        st.session_state.annotation_loaded_output = str(output_file)

        st.success(
            f"Loaded {len(merged['display_ids'])} labelable chunks: "
            f"{merged['exact_match_count']} exact + "
            f"{merged['fuzzy_match_count']} fuzzy GT matches, and "
            f"{len(merged['new_ids'])} new from classified.json."
        )
        st.caption(
            f"Fuzzy matches require at least {MIN_CONTENT_SIMILARITY:.0%} "
            "content similarity and an unambiguous one-to-one best match."
        )
        if merged["gt_only_ids"]:
            st.info(
                f"Preserved {len(merged['gt_only_ids'])} non-abstract GT chunk(s) "
                "whose content is absent from the current classified file."
            )
        if merged["source_abstract_count"]:
            st.caption(
                f"Kept {merged['source_abstract_count']} abstract chunk(s) out of the annotation view."
            )
        if merged["duplicate_existing"] or merged["duplicate_classified"]:
            st.warning(
                "Duplicate content was found. Matches are paired in file order; "
                "please review duplicate chunks carefully."
            )

    records_by_id = st.session_state.get("annotation_records")
    loaded_study = st.session_state.get("annotation_loaded_study")
    if not records_by_id or loaded_study != study_choice:
        return

    display_ids = st.session_state.annotation_display_ids
    source_predictions = st.session_state.annotation_source_predictions
    origins = st.session_state.annotation_origins
    new_ids = st.session_state.annotation_new_ids
    gt_only_ids = st.session_state.annotation_gt_only_ids
    load_nonce = st.session_state.annotation_load_nonce

    st.markdown("#### Annotation view")
    filter_mode = st.selectbox(
        "Show chunks",
        ["All chunks", "New from classified.json", "Existing GT chunks"],
    )
    search_text = st.text_input(
        "Search chunk text or ID",
        placeholder="e.g., 26, adsorption, supplementary",
    ).strip().lower()

    visible_ids = []
    for entry_id in display_ids:
        record = records_by_id[entry_id]
        annotation = parse_predicted_labels(record.get("annotation"))
        if filter_mode == "New from classified.json" and entry_id not in new_ids:
            continue
        if filter_mode == "Existing GT chunks" and entry_id in new_ids:
            continue
        searchable = " ".join(
            [
                str(record.get("chunk_id") or ""),
                str(record.get("file_type") or ""),
                str(record.get("section_label") or ""),
                str(record.get("enriched_text") or ""),
            ]
        ).lower()
        if search_text and search_text not in searchable:
            continue
        visible_ids.append(entry_id)

    st.caption(f"Showing {len(visible_ids)} of {len(display_ids)} labelable chunks")

    for index, entry_id in enumerate(visible_ids):
        record = records_by_id[entry_id]
        predicted = source_predictions.get(entry_id, [])
        origin = origins[entry_id]
        predicted_text = ", ".join(predicted) if predicted else "none"
        title = (
            f"{index + 1}. Chunk {record.get('chunk_id')} [{record.get('file_type')}] "
            f"- {origin} - model: {predicted_text}"
        )
        with st.expander(title, expanded=(index == 0)):
            st.markdown(
                f"""
                <div style="
                    white-space: pre-wrap;
                    line-height: 1.35;
                    font-size: 0.95rem;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    padding: 0.75rem;
                    background: #fafafa;
                ">{html.escape(format_chunk_text_for_display(record.get("enriched_text", "")))}</div>
                """,
                unsafe_allow_html=True,
            )
            if record.get("section_label"):
                st.caption(f"Section: {record['section_label']}")

            columns = st.columns(len(LABEL_OPTIONS))
            selected = []
            for column, label in zip(columns, LABEL_OPTIONS):
                widget_key = f"annotation_{load_nonce}_{label}_{entry_id}"
                checked = label in parse_predicted_labels(record.get("annotation"))
                if column.checkbox(label, key=widget_key, value=checked):
                    selected.append(label)
            record["annotation"] = selected

            if has_conflicting_labels(selected):
                st.warning("Select irrelevant alone.")

    if st.button("Save GT revisions", type="primary"):
        conflicts = [
            entry_id
            for entry_id in display_ids
            if has_conflicting_labels(records_by_id[entry_id].get("annotation"))
        ]
        if conflicts:
            st.error(
                f"Resolve conflicting labels in {len(conflicts)} chunk(s): "
                "irrelevant cannot be combined with another label."
            )
            return

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_records = records_for_save(
                records_by_id,
                st.session_state.annotation_save_order,
            )
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(output_records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return

        st.success(
            f"Saved {len(output_records)} GT records to {output_file}. "
            "Existing labels were preserved unless you changed them."
        )


if __name__ == "__main__":
    main()
