# Preprocessing workflow

This directory turns already converted Docling JSON into JSON-first records
with relevant sections, figures, and repaired tables, then writes the chunk
files consumed by `Data_Extraction`.

[`workflow_config.py`](workflow_config.py) is this module's single editable
source for `ACTIVE_STUDIES`, the local Training/Validation/Application registry,
raw-source paths, processed paths, and rerun settings.  It intentionally does
not import `Data_Extraction/workflow_config.py`; configure the two modules
separately when they should run different study batches.

By default `OVERWRITE_EXISTING_OUTPUTS = False`.  Existing Docling JSON,
JSON-first records, and chunk files are retained.  Enable it only for an
intentional regeneration of the currently selected studies.  `FORCE_RERUN_LLM`
separately controls cached table-repair decisions in the JSON-first stage.

Only these two scripts are workflow entry points. Run them in order from this
directory:

```powershell
python docling_json_preprocess.py --check
python docling_json_preprocess.py
python chunk_json_input.py --check
python chunk_json_input.py
```

`--check` validates only the configured study paths and never writes files or
loads the Docling/model runtime. The JSON-first stage requires at least one of
`main_paper.docling.json` or `supplementary_material.docling.json`; the chunk
stage requires `Processed/<study>/chunk_input.jsonl`.

## Refresh downstream artifacts after a small upstream correction

If a deterministic correction changes only source text (for example, adding a
missing table unit) and cannot change the existing LLM classifications or
enumerated IDs, refresh their source links here instead of rerunning those LLM
stages:

```powershell
python refresh_downstream_artifacts.py --dry-run
python refresh_downstream_artifacts.py
python refresh_downstream_artifacts.py --classification-only
```

The first command is optional and validates source lineage without refreshing
any artifact. The plain command reads this directory's `ACTIVE_STUDIES` and
updates the classified and existing enumeration artifacts for those studies
directly, without creating backups. It never creates an LLM client or makes an
LLM call. After a successful refresh, run `Data_Extraction/extract.py` for the
same study selection.

`--classification-only` updates the per-study and combined classification JSON
files but leaves all enumeration artifacts unchanged. It is useful when only
the classified source text needs to be current. Do not use it when performance
extraction must see the correction: that stage reads table text and source
metadata from the performance enumeration artifact, so use the plain refresh
command in that case.

If the dry run reports an unmatched source, or if the correction changed the
meaning or logical boundaries of a chunk, run classification and enumeration
normally instead.

The `helpers/` directory contains library modules that these entry points
import; do not run them directly. Tests live in `tests/`.

## WSL mounts

When running the chunk stage in WSL, its processed-data mount must expose
`study_58/chunk_input.jsonl` directly beneath `~/docling/processed` (the
current Windows directory is `.../2. Adsorbent Recommender/Processed`, not
`Adsorbent_Processed`).  The WSL default is therefore
`~/docling/processed`. For a different mount point, set it only for that run:

```bash
ADSORBENT_PROCESSED_DIR="$HOME/docling/processed" python chunk_json_input.py
```

`docling_json_preprocess.py` additionally needs mounted source documents and
the records workbook; use `ADSORBENT_SOURCE_STUDIES_DIR` and
`ADSORBENT_STUDY_RECORDS_XLSX` if those are outside the default mount layout.

`convert_to_docling_json.py` is the upstream step that converts the retrieved
article and supplementary PDF/DOCX files into `main_paper.docling.json` and
`supplementary_material.docling.json` (plus a Markdown companion for review).
Run it before the two stages above:

```powershell
python convert_to_docling_json.py --check
python convert_to_docling_json.py
```
