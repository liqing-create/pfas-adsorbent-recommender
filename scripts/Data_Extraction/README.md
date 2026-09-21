# Data-extraction workflow

`workflow_config.py` is the only file to edit when selecting a normal run.
Set `ACTIVE_STUDIES` to the study IDs to process.  The file maps each ID to
exactly one of Training, Validation, or Application and resolves the correct
classification, enumeration, extraction, annotation, and artifact locations.

The current run directories are deliberately explicit because the historical
Training and Application layouts do not share one folder convention.  Do not
add paths or study lists to individual stage scripts.

## Run the workflow

From this folder, run the stages in order:

```powershell
python classify.py
python enumerate.py
python extract.py
```

Existing completed outputs are reused by default.  Set
`OVERWRITE_EXISTING_OUTPUTS = True` in `workflow_config.py` only for a
deliberate rerun.  The enumeration and extraction scripts retain their
existing task toggles; those control LLM work, while `ACTIVE_STUDIES` controls
the study batch.

Performance extraction uses enumeration provenance to choose its input
sources. Tables not cited by any remaining enumeration record's `Data_Source`
are excluded, while uncited non-table chunks are retained as context. If a
`Data_Source` value is missing, invalid, or cannot be mapped through
`enriched_text_metadata`, extraction falls back to the complete original
source group. Each performance extraction output records the decision in its
`source_routing` field.

For a small deterministic correction to preprocessing output, use
`../Preprocessing/refresh_downstream_artifacts.py` from the Preprocessing
directory before running extraction. That utility deliberately uses the
preprocessing workflow's `ACTIVE_STUDIES`; `extract.py` continues to use this
directory's selected studies.

All internal support modules are kept in `helpers/`; the root folder contains
the runnable stages, evaluators, review UIs, configuration, and documentation.

## Debugging evaluator rows

Use `debug.py` to rerun only problematic chunks. Choose one stage at a time;
the selected studies still come from `ACTIVE_STUDIES`, while the CSV selects
the chunks within those studies. After the corresponding evaluator has run,
the runner automatically finds its CSV in the configured stage output folder.

```powershell
python debug.py --classify
python debug.py --enumerate
python debug.py --extract
```

Use `--csv "C:\path\to\report.csv"` only for an alternate or historical
report. Enumeration and extraction can consume their respective mismatch CSVs
directly; rows with `source_chunk_ids` or `source_id` are also understood,
including grouped study-level entries. If a CSV contains several task types,
use `--task-filter performance` (or set `DEBUG_TASK_FILTER` in `debug.py`).

Debug runs read the normal production JSON files in memory and write all
results, chain artifacts, and error logs to sibling directories ending in
`_debug`.  They never overwrite production inputs or outputs and always rerun
the selected LLM work instead of reusing production artifacts. The default
suffix can be changed with `--output-suffix` or `DEBUG_OUTPUT_SUFFIX` in
`debug.py`; it must not be empty.

## Promoting a focused debug result

After reviewing a debug result, use `promote_debug.py` to preview a focused
replacement in the normal output. The first command is always a dry run:

```powershell
python promote_debug.py --classify
python promote_debug.py --enumerate
python promote_debug.py --extract
```

If the displayed source chunks and files are correct, repeat the chosen command
with `--apply`, for example:

```powershell
python promote_debug.py --enumerate --apply
```

Promotion matches the underlying source chunks rather than generated batch
names. It refuses partial or ambiguous overlaps, updates the corresponding
chain artifact, and backs up every changed production JSON under
`_focused_replacement_backups/`. Evaluation CSV/workbook files are not edited;
rerun the corresponding evaluator after promotion.

## Review and evaluation

```powershell
streamlit run classify_UI.py
streamlit run enumeration_UI.py
streamlit run extraction_UI.py

python evaluation_classification.py
python evaluation_enumeration.py
python evaluation_extraction.py
```

The review apps show only `ACTIVE_STUDIES` and save ground truth in the same
study group.  Evaluators write separate reports into each selected group’s
prediction directory; studies without human ground truth are skipped where a
metric cannot be calculated.

## Supporting files

- Upstream conversion, JSON-first preprocessing, and chunking live in
  `../Preprocessing`.  It has its own `workflow_config.py` and independent
  `ACTIVE_STUDIES`; update both configs only when the same studies should run
  through both workflows.  Its figure/table CSV remains the shared
  caption-context source for enumeration and extraction.
- `helpers/figure_caption_context.py` and `helpers/extract_by_chunk.py` are internal helpers
  used by enumeration/extraction, not standalone workflows.
- `helpers/post_process_enumeration.py` is safe by default: its standalone mode uses
  selected studies and `DRY_RUN = True`; enumeration already applies its rules
  during the normal pipeline.
- `screen.py` is upstream abstract screening.  It runs on the Web of Science
  export before any study folder exists, so it is not part of the ordered
  stage list above and does not read `ACTIVE_STUDIES`.
- `prompts/extract_template_without_enumeration.j2` is the single-step
  (no-enumeration) extraction prompt used only for the with/without-enumeration
  comparison in the manuscript (Fig. 2f); the current pipeline does not load it.
