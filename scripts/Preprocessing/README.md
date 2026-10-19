# Docling workflow

Only two files need your attention:

- `workflow_config.py` — change `ACTIVE_STUDIES` to select studies.
- `run_docling_stage.sh` — run every workflow stage.

The implementation is under `helpers/`; do not run those files directly.

## Run

First mount the `Data`, `Processed`, and `scripts` shares, then:

```bash
cd ~/docling/scripts

./run_docling_stage.sh convert
./run_docling_stage.sh preprocess --check
./run_docling_stage.sh preprocess
./run_docling_stage.sh chunk --check
./run_docling_stage.sh chunk
```

All stages use `ACTIVE_STUDIES` from `workflow_config.py` and the Python at
`/home/liqing/miniconda3/envs/docling/bin/python`.

Conversion skips existing `*.docling.json` files. To intentionally regenerate
the selected studies:

```bash
FORCE_RERUN=true ./run_docling_stage.sh convert
```

If the records workbook is not at `~/docling/within_scope_records.xlsx`, set
its WSL path when preprocessing:

```bash
ADSORBENT_STUDY_RECORDS_XLSX=/path/to/within_scope_records.xlsx \
  ./run_docling_stage.sh preprocess
```
