#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="${SCRIPTS_ROOT:-$SCRIPT_DIR}"
WORKFLOW_DIR="$SCRIPTS_ROOT/helpers/workflow"
DOCLING_PYTHON="${DOCLING_PYTHON:-/home/liqing/miniconda3/envs/docling/bin/python}"

export PYTHONPATH="$SCRIPTS_ROOT:$SCRIPTS_ROOT/helpers/backend${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export ADSORBENT_SOURCE_STUDIES_DIR="${ADSORBENT_SOURCE_STUDIES_DIR:-$HOME/docling/data}"
export ADSORBENT_PROCESSED_DIR="${ADSORBENT_PROCESSED_DIR:-$HOME/docling/processed}"

# The workbook is not part of the Data, Processed, or scripts mounts. Use a
# caller-provided path when available, otherwise recognize the conventional
# local location without silently copying or duplicating the workbook.
if [[ -z "${ADSORBENT_STUDY_RECORDS_XLSX:-}" ]] \
  && [[ -f "$HOME/docling/within_scope_records.xlsx" ]]; then
  export ADSORBENT_STUDY_RECORDS_XLSX="$HOME/docling/within_scope_records.xlsx"
fi

usage() {
  cat <<'EOF'
Usage: run_docling_stage.sh {convert|preprocess|chunk} [stage options]

  convert      Run the selected-study Docling conversion wrapper.
  preprocess   Build JSON-first records from *.docling.json files.
  chunk        Build final chunks from Processed/<study>/chunk_input.jsonl.

Examples:
  ./run_docling_stage.sh convert
  ./run_docling_stage.sh preprocess --check
  ./run_docling_stage.sh preprocess
  ./run_docling_stage.sh chunk --check
  ./run_docling_stage.sh chunk
EOF
}

if [[ ! -x "$DOCLING_PYTHON" ]]; then
  echo "[ERROR] Docling Python is not executable: $DOCLING_PYTHON" >&2
  exit 2
fi

stage="${1:-}"
if (( $# > 0 )); then
  shift
fi

case "$stage" in
  convert)
    if (( $# > 0 )); then
      echo "[ERROR] The conversion selection is configured in docling_all_studies.sh; no positional options are accepted." >&2
      exit 2
    fi
    exec "$WORKFLOW_DIR/docling_all_studies.sh"
    ;;
  preprocess)
    exec "$DOCLING_PYTHON" "$WORKFLOW_DIR/docling_json_preprocess.py" "$@"
    ;;
  chunk)
    exec "$DOCLING_PYTHON" "$WORKFLOW_DIR/chunk_json_input.py" "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "[ERROR] Unknown stage: $stage" >&2
    usage >&2
    exit 2
    ;;
esac
