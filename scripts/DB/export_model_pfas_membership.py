"""Export per-model PFAS membership for the chemical-space map.

The TMAP figure in PFAS/viz_pfas_database.py colors the database by chemical
class. That map cannot also show which model cohort each compound belongs to,
because color is already spent on the class hierarchy. This script writes the
membership table that PFAS/viz_model_chemical_space.py needs to draw the same
map once per model.

Model row sets come from backend.logkd_data.load_model_rows with the same
arguments data_audit.py uses, so the compound counts here match the PFAS entity
counts on the audit workbook's Evidence Coverage sheet.

Every modeled PFAS is resolved to exactly one row of mine_classified.csv, in
this key order: RDKit canonical SMILES, DTXSID, CAS, then abbreviation. The
resolution used for each compound is recorded so the join stays auditable.

Run from the repository's normal Python environment (not the pfas-viz
container, which has no access to the performance database):
  python export_model_pfas_membership.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent / "ML_model_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from backend import logkd_config as cfg
from backend.logkd_data import load_model_rows


# Global is listed last so the per-model columns are written in the order the
# figure panels use, and so the union check below runs against a set that is
# already populated.
MODELS = ("AC", "Resin", "CDP", "Global")
CLASS_MODELS = ("AC", "Resin", "CDP")

PFAS_NAME_COLUMN = "PFAS_name"
MODEL_KEY_COLUMNS = ("Canonical SMILES", "DSSTox Substance ID", "CAS Number", "Abbreviation")

ATLAS_SMILES_COLUMN = "RDKIT_SMILES"
ATLAS_CARRY_COLUMNS = (
    "Abbreviation",
    "Compound Full Name",
    "CAS Number",
    "DSSTox Substance ID",
    "First_Class",
    "Second_Class",
)

OUTPUT_NAME = "model_pfas_membership.csv"


def normalized_dtxsid(value: object) -> str:
    """Return the bare DTXSID accession, or an empty string when absent."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    marker = "DTXSID"
    position = text.find(marker)
    if position < 0:
        return ""
    digits = []
    for character in text[position + len(marker):]:
        if not character.isdigit():
            break
        digits.append(character)
    return marker + "".join(digits) if digits else ""


def normalized_cas(value: object) -> str:
    """Return a CAS registry number, or an empty string for anything else.

    Some rows carry a lookup URL rather than a registry number, so the shape is
    checked rather than trusted.
    """

    if pd.isna(value):
        return ""
    text = "".join(str(value).split())
    parts = text.split("-")
    if len(parts) != 3:
        return ""
    if not all(part.isdigit() for part in parts):
        return ""
    if not 2 <= len(parts[0]) <= 7 or len(parts[1]) != 2 or len(parts[2]) != 1:
        return ""
    return text


def normalized_abbreviation(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(character for character in str(value).casefold() if character.isalnum())


def unique_value_index(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Map a key to a row position, keeping only unambiguous, non-empty keys."""

    counts = frame.loc[frame[column].ne(""), column].value_counts()
    unambiguous = set(counts[counts.eq(1)].index)
    return {
        value: position
        for position, value in enumerate(frame[column])
        if value in unambiguous
    }


def load_atlas_table(atlas_csv: Path) -> pd.DataFrame:
    atlas = pd.read_csv(atlas_csv)
    missing = [
        column
        for column in (ATLAS_SMILES_COLUMN,) + ATLAS_CARRY_COLUMNS
        if column not in atlas.columns
    ]
    if missing:
        raise ValueError(f"{atlas_csv} is missing required columns: {missing}")
    atlas["_dtxsid_key"] = atlas["DSSTox Substance ID"].map(normalized_dtxsid)
    atlas["_cas_key"] = atlas["CAS Number"].map(normalized_cas)
    atlas["_abbreviation_key"] = atlas["Abbreviation"].map(normalized_abbreviation)
    return atlas


def model_pfas_table(model: str, args: argparse.Namespace) -> pd.DataFrame:
    frame, _info, _catalog = load_model_rows(
        input_path=args.input_path,
        sheet_name=args.sheet_name,
        model=model,
        target=cfg.TARGET,
        pfas_features_path=args.pfas_features_path,
        pfas_features_sheet=args.pfas_features_sheet,
        data_mode=args.data_mode,
    )
    missing = [column for column in MODEL_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Model {model} rows lack the PFAS identity columns {missing}. "
            "The PFAS feature join is required for this export."
        )
    unique = frame.drop_duplicates(PFAS_NAME_COLUMN)
    return unique[[PFAS_NAME_COLUMN, *MODEL_KEY_COLUMNS]].reset_index(drop=True)


def resolve_to_atlas(
    pfas_table: pd.DataFrame,
    atlas: pd.DataFrame,
    smiles_index: dict[str, int],
    dtxsid_index: dict[str, int],
    cas_index: dict[str, int],
    abbreviation_index: dict[str, int],
) -> pd.DataFrame:
    """Attach the atlas row position and the key that produced each match."""

    positions: list[object] = []
    matched_by: list[str] = []

    for smiles, dtxsid, cas, abbreviation in zip(
        pfas_table["Canonical SMILES"],
        pfas_table["DSSTox Substance ID"],
        pfas_table["CAS Number"],
        pfas_table["Abbreviation"],
    ):
        candidates = (
            ("RDKit canonical SMILES", smiles_index.get("" if pd.isna(smiles) else str(smiles))),
            ("DTXSID", dtxsid_index.get(normalized_dtxsid(dtxsid))),
            ("CAS", cas_index.get(normalized_cas(cas))),
            ("Abbreviation", abbreviation_index.get(normalized_abbreviation(abbreviation))),
        )
        position = None
        key_name = ""
        for name, candidate in candidates:
            if candidate is not None:
                position = candidate
                key_name = name
                break
        positions.append(position)
        matched_by.append(key_name)

    resolved = pfas_table.copy()
    resolved["_atlas_position"] = positions
    resolved["_matched_by"] = matched_by
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=cfg.DEFAULT_INPUT)
    parser.add_argument("--sheet-name", default=cfg.SHEET_NAME)
    parser.add_argument("--pfas-features-path", type=Path, default=cfg.DEFAULT_PFAS_FEATURES)
    parser.add_argument("--pfas-features-sheet", default=cfg.DEFAULT_PFAS_FEATURES_SHEET)
    parser.add_argument(
        "--data-mode",
        choices=["baseline", "drop_unreliable"],
        default="drop_unreliable",
        help="Must match the mode used for data_audit.py so the counts agree.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=cfg.DEFAULT_PFAS_FEATURES.parent / "pfas-atlas output",
        help="Folder holding mine_classified.csv and mine.npz.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=f"Defaults to {OUTPUT_NAME} inside --atlas-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atlas_csv = args.atlas_dir / "mine_classified.csv"
    if not atlas_csv.exists():
        raise FileNotFoundError(f"{atlas_csv} is missing.")
    output_path = args.output_path or args.atlas_dir / OUTPUT_NAME

    atlas = load_atlas_table(atlas_csv)
    smiles_index = unique_value_index(atlas, ATLAS_SMILES_COLUMN)
    dtxsid_index = unique_value_index(atlas, "_dtxsid_key")
    cas_index = unique_value_index(atlas, "_cas_key")
    abbreviation_index = unique_value_index(atlas, "_abbreviation_key")

    membership = atlas[[ATLAS_SMILES_COLUMN, *ATLAS_CARRY_COLUMNS]].copy()
    model_names: dict[str, list[str]] = {position: [] for position in range(len(atlas))}
    match_keys: dict[int, str] = {}
    model_sets: dict[str, set[str]] = {}

    for model in MODELS:
        pfas_table = model_pfas_table(model, args)
        resolved = resolve_to_atlas(
            pfas_table,
            atlas,
            smiles_index,
            dtxsid_index,
            cas_index,
            abbreviation_index,
        )

        unresolved = resolved.loc[resolved["_atlas_position"].isna()]
        if not unresolved.empty:
            names = ", ".join(str(name) for name in unresolved[PFAS_NAME_COLUMN])
            raise ValueError(
                f"{len(unresolved)} {model} PFAS could not be located in {atlas_csv}: {names}. "
                "Regenerate the atlas inputs, or extend the key order in resolve_to_atlas."
            )

        positions = resolved["_atlas_position"].astype(int)
        if positions.duplicated().any():
            duplicated = resolved.loc[positions.duplicated(keep=False), PFAS_NAME_COLUMN].tolist()
            raise ValueError(
                f"Several {model} PFAS resolved to the same atlas compound: {duplicated}."
            )

        flags = pd.Series(False, index=membership.index)
        flags.iloc[positions] = True
        membership[f"in_{model}"] = flags
        model_sets[model] = set(resolved[PFAS_NAME_COLUMN])

        for position, name, key in zip(positions, resolved[PFAS_NAME_COLUMN], resolved["_matched_by"]):
            match_keys.setdefault(position, key)
            if model != "Global":
                model_names[position].append(str(name))

    membership["n_class_models"] = membership[[f"in_{model}" for model in CLASS_MODELS]].sum(axis=1)
    membership["matched_by"] = [match_keys.get(position, "") for position in range(len(atlas))]
    membership["PFAS_name"] = [
        model_names[position][0] if model_names[position] else ""
        for position in range(len(atlas))
    ]
    membership.insert(0, "atlas_row", range(len(atlas)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_path, index=False)

    union_of_classes = set().union(*(model_sets[model] for model in CLASS_MODELS))
    print(f"Wrote: {output_path}")
    print(f"Atlas compounds: {len(atlas)}")
    for model in MODELS:
        print(f"  {model:<7} {len(model_sets[model]):>4} PFAS")
    print(f"Class-model union equals the Global cohort: {union_of_classes == model_sets['Global']}")
    for model in CLASS_MODELS:
        others = set().union(*(model_sets[other] for other in CLASS_MODELS if other != model))
        print(f"  {model:<7} unique to this cohort: {len(model_sets[model] - others)}")
    shared_by_all = set.intersection(*(model_sets[model] for model in CLASS_MODELS))
    print(f"  Shared by all three cohorts: {len(shared_by_all)}")
    print(f"  Modeled by no adsorbent class (atlas only): {int((membership['n_class_models'] == 0).sum())}")
    print(membership["matched_by"].value_counts().to_string())


if __name__ == "__main__":
    main()
