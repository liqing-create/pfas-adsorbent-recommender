#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/docling/data}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="${WORKFLOW_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
DOCLING_PYTHON="${DOCLING_PYTHON:-/home/liqing/miniconda3/envs/docling/bin/python}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/_logs}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -x "$DOCLING_PYTHON" ]]; then
  echo "[ERROR] Docling Python is not executable: $DOCLING_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$WORKFLOW_ROOT/convert_to_docling_json.py" ]]; then
  echo "[ERROR] Converter not found: $WORKFLOW_ROOT/convert_to_docling_json.py" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"

# Resume behavior is the default:
# - Existing *.docling.json files are skipped.
# - Missing outputs are converted.
#
# Set FORCE_RERUN=true to rerun every study included in this run.
FORCE_RERUN="${FORCE_RERUN:-false}"

# PDF page orientation preprocessing:
# - Uses Docling's existing pypdfium2 dependency; no extra package is needed.
# - Only embedded-text pages with a strong non-upright orientation are rotated.
# - Mixed-orientation table pages can be rescued by an auxiliary rotated conversion.
# - Set PDF_ROTATION_REPORT_ONLY=true to audit recommendations without converting.
AUTO_ROTATE_PDF_PAGES="${AUTO_ROTATE_PDF_PAGES:-true}"
MIXED_ORIENTATION_TABLE_RESCUE="${MIXED_ORIENTATION_TABLE_RESCUE:-true}"
PDF_ROTATION_MIN_CHARS="${PDF_ROTATION_MIN_CHARS:-500}"
PDF_ROTATION_MIN_SHARE="${PDF_ROTATION_MIN_SHARE:-0.70}"
PDF_ROTATION_MIN_DOMINANCE_RATIO="${PDF_ROTATION_MIN_DOMINANCE_RATIO:-2.5}"
PDF_ROTATION_REPORT_ONLY="${PDF_ROTATION_REPORT_ONLY:-false}"

# Usually this is safer than FORCE_RERUN=true. Add only the studies whose
# existing outputs should be backed up and regenerated. This works whether
# USE_SELECTION is true or false.
FORCE_RERUN_STUDIES=(
  # "study_01"
)

mapfile -t STUDIES < <(
  "$DOCLING_PYTHON" -c \
    'import workflow_config as config; [print(paths.study_id) for paths in config.iter_active_study_paths()]'
)
if (( ${#STUDIES[@]} == 0 )) || [[ -z "${STUDIES[0]}" ]]; then
  echo "[ERROR] ACTIVE_STUDIES did not select any studies." >&2
  exit 2
fi

VALID=()
for study in "${STUDIES[@]}"; do
  if [[ -d "$DATA_ROOT/$study" ]]; then
    VALID+=( "$study" )
  else
    echo "[WARN] Study directory not found: $DATA_ROOT/$study"
  fi
done
if (( ${#VALID[@]} == 0 )); then
  echo "[ERROR] No selected study directories exist under $DATA_ROOT."
  exit 1
fi

ARGS=(
  "$WORKFLOW_ROOT/convert_to_docling_json.py"
  --data-root "$DATA_ROOT"
  --studies "${VALID[@]}"
  --rotation-min-chars "$PDF_ROTATION_MIN_CHARS"
  --rotation-min-share "$PDF_ROTATION_MIN_SHARE"
  --rotation-min-dominance-ratio "$PDF_ROTATION_MIN_DOMINANCE_RATIO"
)

case "${FORCE_RERUN,,}" in
  true|1|yes|y) ARGS+=( --force ) ;;
esac
case "${AUTO_ROTATE_PDF_PAGES,,}" in
  false|0|no|n) ARGS+=( --no-auto-rotate-pdf-pages ) ;;
esac
case "${MIXED_ORIENTATION_TABLE_RESCUE,,}" in
  false|0|no|n) ARGS+=( --no-mixed-orientation-table-rescue ) ;;
esac
case "${PDF_ROTATION_REPORT_ONLY,,}" in
  true|1|yes|y) ARGS+=( --rotation-report-only ) ;;
esac

for study in "${FORCE_RERUN_STUDIES[@]}"; do
  [[ -n "$study" ]] && ARGS+=( --force-study "$study" )
done

echo "[INFO] Checking ${#VALID[@]} study/studies (force_rerun=$FORCE_RERUN)."
echo "[INFO] PDF auto-rotation: $AUTO_ROTATE_PDF_PAGES " \
  "(min_chars=$PDF_ROTATION_MIN_CHARS, min_share=$PDF_ROTATION_MIN_SHARE, " \
  "min_ratio=$PDF_ROTATION_MIN_DOMINANCE_RATIO, " \
  "mixed_table_rescue=$MIXED_ORIENTATION_TABLE_RESCUE)"  
if [[ "${PDF_ROTATION_REPORT_ONLY,,}" =~ ^(true|1|yes|y)$ ]]; then
  echo "[INFO] Rotation report-only mode: Docling conversion will not run."
fi
if (( ${#FORCE_RERUN_STUDIES[@]} > 0 )); then
  echo "[INFO] Force-rerun studies: ${FORCE_RERUN_STUDIES[*]}"
fi
"$DOCLING_PYTHON" "${ARGS[@]}" 2>&1 | tee -a "$LOG_ROOT/docling_json_conversion.log"
if [[ "${PDF_ROTATION_REPORT_ONLY,,}" =~ ^(true|1|yes|y)$ ]]; then
  echo "[DONE] PDF rotation recommendations were written to $LOG_ROOT/pdf_rotation_report.json."
else
  echo "[DONE] Authoritative conversion files are now *.docling.json."
fi
