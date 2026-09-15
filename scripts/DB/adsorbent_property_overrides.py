"""Explicit, adsorbent-scoped decisions for property normalization.

These decisions are deliberately separate from performance-record review
decisions.  An adsorbent property is copied to several performance records, so
its stable identity is the extracted dataset, study, and adsorbent identifier.
"""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = (
    "extraction_dataset",
    "study_no",
    "adsorbent_id",
    "property",
    "action",
)
ELEMENTAL_COMPOSITION_PROPERTY = "elemental_composition"
AVERAGE_REPORTED_VALUES_ACTION = "average_reported_values"
ADSORBENT_PROPERTIES_PROPERTY = "adsorbent_properties"
NO_PROPERTIES_EXPECTED_ACTION = "no_properties_expected"


def normalized_adsorbent_property_key(
    extraction_dataset: object,
    study_no: object,
    adsorbent_id: object,
) -> tuple[str, str, str]:
    """Return the exact, case-insensitive study-local adsorbent identity."""
    return tuple(
        str(value or "").strip().casefold()
        for value in (extraction_dataset, study_no, adsorbent_id)
    )


def normalized_adsorbent_property_study_key(
    extraction_dataset: object,
    study_no: object,
) -> tuple[str, str]:
    """Return the exact, case-insensitive identity for a reviewed study."""
    return tuple(
        str(value or "").strip().casefold()
        for value in (extraction_dataset, study_no)
    )


def load_elemental_average_overrides(path: str | Path | None) -> set[tuple[str, str, str]]:
    """Load approved elemental-composition averages from a dedicated CSV.

    Unknown actions and incomplete identities fail closed so an edit to the CSV
    cannot silently alter the wrong adsorbent property.
    """
    if not path:
        return set()
    override_path = Path(path)
    if not override_path.exists() or not override_path.stat().st_size:
        return set()

    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"Adsorbent-property decisions file {override_path} is missing required "
                f"columns {missing}. Expected at least {list(REQUIRED_COLUMNS)}."
            )

        approved: set[tuple[str, str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            property_name = str(row.get("property", "") or "").strip().casefold()
            action = str(row.get("action", "") or "").strip().casefold()
            if not property_name and not action:
                continue
            if property_name == ADSORBENT_PROPERTIES_PROPERTY:
                study_key = normalized_adsorbent_property_study_key(
                    row.get("extraction_dataset"),
                    row.get("study_no"),
                )
                adsorbent_id = str(row.get("adsorbent_id", "") or "").strip()
                if not all(study_key) or adsorbent_id:
                    raise ValueError(
                        f"A no-properties decision in {override_path} row {row_number} must "
                        "identify a dataset and study, with a blank adsorbent_id"
                    )
                if action != NO_PROPERTIES_EXPECTED_ACTION:
                    raise ValueError(
                        f"Unsupported no-properties action {action!r} in {override_path} row "
                        f"{row_number}. Use {NO_PROPERTIES_EXPECTED_ACTION!r}."
                    )
                continue
            key = normalized_adsorbent_property_key(
                row.get("extraction_dataset"),
                row.get("study_no"),
                row.get("adsorbent_id"),
            )
            if not all(key):
                raise ValueError(
                    f"Incomplete adsorbent identity for override in {override_path} row {row_number}"
                )
            if property_name != ELEMENTAL_COMPOSITION_PROPERTY:
                raise ValueError(
                    f"Unsupported override property {property_name!r} in {override_path} row "
                    f"{row_number}. Use {ELEMENTAL_COMPOSITION_PROPERTY!r}."
                )
            if action != AVERAGE_REPORTED_VALUES_ACTION:
                raise ValueError(
                    f"Unsupported override action {action!r} in {override_path} row {row_number}. "
                    f"Use {AVERAGE_REPORTED_VALUES_ACTION!r}."
                )
            approved.add(key)
    return approved


def load_expected_missing_adsorbent_property_studies(
    path: str | Path | None,
) -> set[tuple[str, str]]:
    """Load study-level decisions that no adsorbent properties were extractable.

    These decisions are intentionally study scoped: they suppress an otherwise
    misleading unmatched-property warning only when the reviewer has confirmed
    that no source adsorbent-property record exists for the study.
    """
    if not path:
        return set()
    override_path = Path(path)
    if not override_path.exists() or not override_path.stat().st_size:
        return set()

    # Validate all decision types before applying this subset.
    load_elemental_average_overrides(override_path)
    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected: set[tuple[str, str]] = set()
        for row in reader:
            property_name = str(row.get("property", "") or "").strip().casefold()
            if property_name != ADSORBENT_PROPERTIES_PROPERTY:
                continue
            expected.add(
                normalized_adsorbent_property_study_key(
                    row.get("extraction_dataset"),
                    row.get("study_no"),
                )
            )
    return expected
