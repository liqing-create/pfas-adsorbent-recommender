"""Explicit, performance-scoped decisions for value normalization.

Unlike adsorbent properties, experimental conditions belong to individual
performance records.  Decisions therefore use the extraction dataset, study,
and extracted differentiating condition rather than regenerated row IDs.
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from pathlib import Path


REQUIRED_COLUMNS = (
    "extraction_dataset",
    "study_no",
    "differentiating_condition",
    "property",
    "action",
    "expected_raw",
)
RANGE_AVERAGE_PROPERTIES = frozenset({"toc_(mg/l)", "doc_(mg/l)"})
AVERAGE_REPORTED_RANGE_ACTION = "average_reported_range"
ADSORBENT_SCALAR_PROPERTIES = frozenset({"phpzc"})
SET_MANUAL_SCALAR_ACTION = "set_manual_scalar"


def normalized_performance_normalization_key(
    extraction_dataset: object,
    study_no: object,
    differentiating_condition: object,
    property_name: object,
) -> tuple[str, str, str, str]:
    """Return the stable, case-insensitive identity for one condition field."""
    return tuple(
        str(value or "").strip().casefold()
        for value in (
            extraction_dataset,
            study_no,
            differentiating_condition,
            property_name,
        )
    )


def normalized_adsorbent_scalar_normalization_key(
    extraction_dataset: object,
    study_no: object,
    adsorbent_id: object,
    water_type: object,
    property_name: object,
) -> tuple[str, str, str, str, str]:
    """Return the stable identity for a manually selected adsorbent scalar."""
    return tuple(
        str(value or "").strip().casefold()
        for value in (
            extraction_dataset,
            study_no,
            adsorbent_id,
            water_type,
            property_name,
        )
    )


def normalized_expected_raw(value: object) -> str:
    """Compare source expressions while tolerating spacing and dash glyphs."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", "", text)


def _load_manual_normalization_overrides(
    path: str | Path | None,
) -> tuple[
    dict[tuple[str, str, str, str], str],
    dict[tuple[str, str, str, str, str], tuple[str, float]],
]:
    """Load approved manual scalar decisions, failing closed on malformed rows."""
    if not path:
        return {}, {}
    decision_path = Path(path)
    if not decision_path.exists() or not decision_path.stat().st_size:
        return {}, {}

    with decision_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"Performance-normalization decisions file {decision_path} is missing "
                f"required columns {missing}. Expected at least {list(REQUIRED_COLUMNS)}."
            )

        range_average_overrides: dict[tuple[str, str, str, str], str] = {}
        adsorbent_scalar_overrides: dict[
            tuple[str, str, str, str, str], tuple[str, float]
        ] = {}
        for row_number, row in enumerate(reader, start=2):
            property_name = str(row.get("property", "") or "").strip().casefold()
            action = str(row.get("action", "") or "").strip().casefold()
            if not property_name and not action:
                continue
            expected_raw = normalized_expected_raw(row.get("expected_raw"))
            if property_name in RANGE_AVERAGE_PROPERTIES:
                key = normalized_performance_normalization_key(
                    row.get("extraction_dataset"),
                    row.get("study_no"),
                    row.get("differentiating_condition"),
                    property_name,
                )
                if not all(key) or not expected_raw:
                    raise ValueError(
                        f"Incomplete performance-normalization decision in {decision_path} "
                        f"row {row_number}"
                    )
                if action != AVERAGE_REPORTED_RANGE_ACTION:
                    raise ValueError(
                        f"Unsupported decision action {action!r} in {decision_path} row "
                        f"{row_number}. Use {AVERAGE_REPORTED_RANGE_ACTION!r}."
                    )
                if key in range_average_overrides:
                    raise ValueError(
                        f"Duplicate performance-normalization decision for {key!r} in "
                        f"{decision_path} row {row_number}"
                    )
                range_average_overrides[key] = expected_raw
                continue

            if property_name in ADSORBENT_SCALAR_PROPERTIES:
                key = normalized_adsorbent_scalar_normalization_key(
                    row.get("extraction_dataset"),
                    row.get("study_no"),
                    row.get("adsorbent_id"),
                    row.get("water_type"),
                    property_name,
                )
                normalized_value = str(row.get("normalized_value", "") or "").strip()
                try:
                    scalar_value = float(normalized_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid normalized_value {normalized_value!r} in {decision_path} "
                        f"row {row_number}"
                    ) from exc
                if not math.isfinite(scalar_value):
                    raise ValueError(
                        f"Invalid normalized_value {normalized_value!r} in {decision_path} "
                        f"row {row_number}"
                    )
                if not all(key) or not expected_raw:
                    raise ValueError(
                        f"Incomplete adsorbent-scalar decision in {decision_path} row "
                        f"{row_number}"
                    )
                if action != SET_MANUAL_SCALAR_ACTION:
                    raise ValueError(
                        f"Unsupported adsorbent-scalar action {action!r} in {decision_path} "
                        f"row {row_number}. Use {SET_MANUAL_SCALAR_ACTION!r}."
                    )
                if key in adsorbent_scalar_overrides:
                    raise ValueError(
                        f"Duplicate adsorbent-scalar decision for {key!r} in {decision_path} "
                        f"row {row_number}"
                    )
                adsorbent_scalar_overrides[key] = (expected_raw, scalar_value)
                continue

            supported = sorted(RANGE_AVERAGE_PROPERTIES | ADSORBENT_SCALAR_PROPERTIES)
            raise ValueError(
                f"Unsupported decision property {property_name!r} in {decision_path} "
                f"row {row_number}. Use one of {supported!r}."
            )
    return range_average_overrides, adsorbent_scalar_overrides


def load_manual_range_average_overrides(
    path: str | Path | None,
) -> dict[tuple[str, str, str, str], str]:
    """Load approved water-quality range averages from the decisions CSV."""
    range_average_overrides, _adsorbent_scalar_overrides = (
        _load_manual_normalization_overrides(path)
    )
    return range_average_overrides


def load_manual_adsorbent_scalar_overrides(
    path: str | Path | None,
) -> dict[tuple[str, str, str, str, str], tuple[str, float]]:
    """Load exact, reviewer-selected adsorbent scalar values."""
    _range_average_overrides, adsorbent_scalar_overrides = (
        _load_manual_normalization_overrides(path)
    )
    return adsorbent_scalar_overrides
