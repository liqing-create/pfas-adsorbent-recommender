"""Generate and screen the PFAS candidate-feature bank from the curated identity list.

Mental model:
  - pfas.xlsx is the manually curated PFAS identity table: names, IDs, and SMILES.
  - pfas_features.xlsx is the generated bank of reproducible candidate values:
    PFAS-Atlas classes, OPERA predictions, Abraham descriptors, scalar RDKit/PFAS
    descriptors, and raw sparse count fingerprints.
  - The same workbook also reports scalar-feature missingness, low variation, and
    strong pairwise collinearity for downstream model-input screening. It does not
    make final feature-selection decisions or assign model, coverage, or
    applicability-domain roles.

Fingerprints are stored in sparse long format outside Sheet1. The raw bank contains
an unfolded count Morgan fingerprint (default radius 2) and an unfolded topological
count atom-pair fingerprint (default distance 1-30). Binary fingerprints and folded
vectors can be derived later without losing the count information.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors


SCRIPT_DIR = Path(__file__).resolve().parent
# Repository root (scripts/PFAS -> scripts -> root). PFAS_DATA_DIR holds pfas.xlsx
# (curated PFAS names and SMILES) and receives pfas_features.xlsx.
AD_ROOT = SCRIPT_DIR.parents[1]
PFAS_DIR = Path(os.getenv("PFAS_DATA_DIR", str(AD_ROOT / "data" / "PFAS")))

DEFAULT_INPUT = PFAS_DIR / "pfas.xlsx"
DEFAULT_OUTPUT = PFAS_DIR / "pfas_features.xlsx"
DEFAULT_WORK_DIR = PFAS_DIR / "opera_work"
DEFAULT_DATABASE = AD_ROOT / "output" / "Merge" / "pfas_adsorption_performance_database.xlsx"
DEFAULT_PFAS_ATLAS_SOURCE = Path(
    os.getenv("PFAS_ATLAS_SOURCE", str(AD_ROOT / "external" / "PFAS-atlas-source"))
)

DEFAULT_OPERA_EXE = Path(os.getenv("OPERA_EXE", r"C:\Program Files\OPERA\application\OPERA.exe"))
DEFAULT_SHEET = "Sheet1"
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_ATOM_PAIR_MAX_DISTANCE = 30

OPERA_ENDPOINTS = ["pKa", "logD", "logP", "KOA", "WS"]
OPERA_COLUMNS = [
    "LogP_pred",
    "LogP_predRange",
    "AD_LogP",
    "AD_index_LogP",
    "Conf_index_LogP",
    "LogWS_pred",
    "WS_predRange",
    "AD_WS",
    "AD_index_WS",
    "Conf_index_WS",
    "LogKOA_pred",
    "KOA_predRange",
    "AD_KOA",
    "AD_index_KOA",
    "Conf_index_KOA",
    "ionization",
    "pKa_a_pred",
    "pKa_a_predRange",
    "pKa_b_pred",
    "pKa_b_predRange",
    "AD_pKa",
    "AD_index_pKa",
    "Conf_index_pKa",
    "LogD55_pred",
    "LogD55_predRange",
    "LogD74_pred",
    "LogD74_predRange",
    "AD_LogD",
    "AD_index_LogD",
    "Conf_index_LogD",
]
OPERA_TEXT_COLUMNS = [col for col in OPERA_COLUMNS if col.endswith("Range")]

OPERA_CRITICAL_COLUMNS = [
    "LogP_pred",
    "LogP_predRange",
    "LogWS_pred",
    "WS_predRange",
    "LogKOA_pred",
    "KOA_predRange",
    "LogD55_pred",
    "LogD55_predRange",
    "LogD74_pred",
    "LogD74_predRange",
]
OPERA_OPTIONAL_PKA_COLUMNS = {
    "pKa_a_pred",
    "pKa_a_predRange",
    "pKa_b_pred",
    "pKa_b_predRange",
}
OPERA_MISSING_STRINGS = {"", "na", "nan", "n/a", "none", "null"}

# Cached OPERA/Abraham rows are matched to input rows by chemical identity, not by
# row position. Positional matching silently misaligns every downstream feature as
# soon as a PFAS is inserted, removed, or reordered in the manual workbook.
CACHE_KEY_COLUMN = "InChIKey"

# Scalar candidates screened after generation. Only actual OPERA endpoint
# predictions are included; ranges, AD metadata, confidence indices, and
# ionization metadata are deliberately excluded.
SCREENED_OPERA_PREDICTION_COLUMNS = (
    "LogP_pred",
    "LogWS_pred",
    "LogKOA_pred",
    "pKa_a_pred",
    "pKa_b_pred",
    "LogD55_pred",
    "LogD74_pred",
)
EXACT_SCALAR_SCREENING_COLUMNS = (*SCREENED_OPERA_PREDICTION_COLUMNS, "Mw (g/mol)")
SCALAR_SCREENING_PREFIXES = ("abraham_", "rdkit_")

# PaDEL/CDK molecular linear free-energy-relation descriptors. PaDEL exposes
# two hydrogen-bond basicity scales (BH and BO); McGowan volume is a separate
# descriptor rather than a member of the MLFER six-column block.
ABRAHAM_SOURCE_TO_OUTPUT = {
    "MLFER_E": "abraham_E",
    "MLFER_S": "abraham_S",
    "MLFER_A": "abraham_A",
    "MLFER_BH": "abraham_BH",
    "MLFER_BO": "abraham_BO",
    "McGowan_Volume": "abraham_V",
    "MLFER_L": "abraham_L",
}
ABRAHAM_COLUMNS = tuple(ABRAHAM_SOURCE_TO_OUTPUT.values())
ABRAHAM_SOURCE_COLUMNS = tuple(ABRAHAM_SOURCE_TO_OUTPUT)

CLASS_COLUMNS = ["First_Class", "Second_Class"]
CLASSIFICATION_AUDIT_COLUMNS = [
    "excel_row",
    "pfas_key",
    "smiles",
    "classification_smiles",
    "classification_status",
    "classification_error",
    "First_Class",
    "Second_Class",
]

GENERATED_EXACT_COLUMNS = {
    "Molecular Formula",
    "Canonical SMILES",
    "Classification",
    *CLASS_COLUMNS,
    *OPERA_COLUMNS,
    *ABRAHAM_COLUMNS,
}
GENERATED_PREFIXES = ("rdkit_", "morgan_r", "atom_pair_", "abraham_")
OVERLAPPING_MANUAL_COLUMNS = {
    "Molecular Formula",
    "Canonical SMILES",
    "# of C",
    "# of F",
    "# of flurinated C",
    "# of fluorinated C",
    "Mw (g/mol)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pfas_features.xlsx from curated PFAS SMILES."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Manual PFAS workbook.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated feature workbook.")
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET)
    parser.add_argument("--key-col", default="Abbreviation")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument(
        "--morgan-radius",
        type=int,
        default=DEFAULT_MORGAN_RADIUS,
        help="Radius for the unfolded sparse-count Morgan raw representation.",
    )
    parser.add_argument(
        "--atom-pair-min-distance",
        type=int,
        default=1,
        help="Minimum topological distance for the unfolded sparse-count atom-pair representation.",
    )
    parser.add_argument(
        "--atom-pair-max-distance",
        type=int,
        default=DEFAULT_ATOM_PAIR_MAX_DISTANCE,
        help="Maximum topological distance for the unfolded sparse-count atom-pair representation.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility argument. Similarity fingerprints are now unfolded "
            "sparse counts and are not folded into a fixed number of bits."
        ),
    )
    parser.add_argument(
        "--classification-mode",
        choices=["run", "skip"],
        default="run",
        help=(
            "run calculates PFAS-Atlas classes by default. Use skip only when you want "
            "blank class columns."
        ),
    )
    parser.add_argument(
        "--pfas-atlas-source",
        type=Path,
        default=DEFAULT_PFAS_ATLAS_SOURCE,
        help="PFAS-Atlas source directory containing classification_helper.",
    )
    parser.add_argument(
        "--classification-verbose",
        action="store_true",
        help="Show PFAS-Atlas per-molecule classification output.",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--opera-mode",
        choices=["auto", "run", "reuse", "preserve", "skip"],
        default="auto",
        help=(
            "auto (default) reuses cached OPERA predictions matched by chemical identity "
            "and calculates only the molecules the cache is missing, because a full OPERA "
            "run is very slow. run forces a recalculation of every row, reuse never "
            "calculates, preserve keeps existing input columns, and skip writes blanks."
        ),
    )
    parser.add_argument("--opera-exe", type=Path, default=DEFAULT_OPERA_EXE)
    parser.add_argument(
        "--opera-csv",
        type=Path,
        default=None,
        help=(
            "Additional OPERA cache CSV to read before the standard work-dir cache "
            f"({DEFAULT_WORK_DIR}). Rows are matched by {CACHE_KEY_COLUMN}, or by "
            "MoleculeID for caches written before keyed caching."
        ),
    )
    parser.add_argument(
        "--abraham-mode",
        choices=["auto", "run", "reuse", "preserve", "skip"],
        default="auto",
        help=(
            "auto (default) reuses cached Abraham descriptors matched by chemical identity "
            "and extracts only the missing molecules from the OPERA/PaDEL sidecars. "
            "run forces a recalculation of every row, reuse never calculates, preserve "
            "keeps existing abraham_* input columns, and skip writes blanks."
        ),
    )
    parser.add_argument(
        "--abraham-csv",
        type=Path,
        default=None,
        help=(
            "Additional Abraham cache CSV to read before the standard work-dir cache "
            f"({DEFAULT_WORK_DIR})."
        ),
    )
    parser.add_argument(
        "--opera-standardize",
        action="store_true",
        help=(
            "Pass -st to OPERA to generate QSAR-ready structures. "
            "Disabled by default because OPERA 2.9 can fail on batch PFAS SMILES with saltID/structure mismatch."
        ),
    )
    parser.add_argument(
        "--opera-batch-size",
        type=int,
        default=10,
        help="Number of molecules per OPERA run. Default 10 because OPERA/PaDEL may only emit 10 rows for larger PFAS batches.",
    )
    parser.add_argument(
        "--no-opera-singleton-fallback",
        action="store_true",
        help=(
            "Do not rerun rows with missing critical OPERA predictions one-by-one. "
            "By default, the script retries those rows because OPERA/PaDEL can silently "
            "emit partial predictions for some PFAS batches."
        ),
    )
    parser.add_argument(
        "--opera-singleton-retries",
        type=int,
        default=3,
        help=(
            "Maximum OPERA attempts for each row in singleton fallback. "
            "Default 3 because OPERA/PaDEL can transiently emit blank endpoint predictions."
        ),
    )
    parser.add_argument(
        "--keep-opera-intermediates",
        action="store_true",
        help="Keep generated OPERA .smi and CSV files in --work-dir.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Deprecated compatibility argument; database coverage is outside this generator.",
    )
    parser.add_argument("--database-sheet", default="Equilibrium_Data", help="Deprecated compatibility argument; ignored.")
    parser.add_argument("--target-col", default="Kd_final_log10(L/g)", help="Deprecated compatibility argument; ignored.")
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.20,
        help="Missing-rate threshold used to flag high-missing scalar candidates.",
    )
    parser.add_argument(
        "--dominant-fraction-threshold",
        type=float,
        default=0.95,
        help="Observed-value fraction used to flag near-constant scalar candidates.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Absolute Pearson or Spearman threshold used to report collinear pairs.",
    )
    parser.add_argument(
        "--min-pairwise-n",
        type=int,
        default=20,
        help="Minimum pairwise-complete observations required for correlation screening.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda series: series.map(clean_text))


def normalize_key(value: Any) -> str:
    text = clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def clean_smiles(value: Any) -> str:
    return clean_text(value)


_STANDARDIZERS: dict[str, Any] = {}


def _standardizers() -> dict[str, Any]:
    """Build the standardizer objects once; construction is not cheap."""
    if not _STANDARDIZERS:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        _STANDARDIZERS["normalizer"] = rdMolStandardize.Normalizer()
        _STANDARDIZERS["reionizer"] = rdMolStandardize.Reionizer()
        _STANDARDIZERS["uncharger"] = rdMolStandardize.Uncharger()
    return _STANDARDIZERS


def standardize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Put every structure in one protonation convention before describing it.

    Whether a carboxylic acid was drawn as -C(=O)OH or as -C(=O)[O-] is a
    drawing convention rather than a structural difference, but it changes
    formal-charge counts, TPSA and molecular weight.  An inconsistently drawn
    bank therefore makes part of the structural block noise, and lets two
    records of one compound look like two compounds.

    Speciation at the experimental pH is carried separately, by pKa together
    with the measured pH, and a model can combine those itself.  The structural
    descriptors should not also encode a guess about it.

    Uncharger neutralizes groups that have a neutral form and leaves permanently
    charged ones alone: a deprotonated sulfonate regains its proton, while a
    quaternary ammonium keeps its charge because there is no proton to move.
    The same Normalizer/Reionizer/Uncharger order is used by
    ``1. Transformation Simulator/AD/scripts/pathway_figures/4_rxnscribe_merge.py``.
    """
    tools = _standardizers()
    try:
        standardized = tools["normalizer"].normalize(mol)
        standardized = tools["reionizer"].reionize(standardized)
        standardized = tools["uncharger"].uncharge(standardized)
        Chem.SanitizeMol(standardized)
    except Exception:
        # A structure that will not standardize is still describable; keeping the
        # parsed molecule is better than dropping the compound.
        return mol
    return standardized


def report_unstandardized_smiles(
    df: pd.DataFrame,
    smiles_col: str,
    key_col: str,
) -> pd.DataFrame:
    """List source rows that are not in the normalized form, without changing them.

    Normalization is a one-time edit to pfas.xlsx, not something this script
    performs.  Reporting drift keeps the maintained list authoritative while
    still surfacing an entry that was pasted in another convention.
    """
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = clean_smiles(row.get(smiles_col, ""))
        parsed = Chem.MolFromSmiles(raw) if raw else None
        if parsed is None:
            continue
        expected = Chem.MolToSmiles(standardize_mol(parsed), canonical=True, isomericSmiles=True)
        if Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True) != expected:
            records.append(
                {"pfas_key": clean_text(row.get(key_col, "")), "found": raw, "normalized": expected}
            )
    return pd.DataFrame(records)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse only.

    pfas.xlsx is the maintained list and is normalized there, once.  This script
    consumes it, so it must not quietly rewrite structures: if it did, the file
    and the descriptors could disagree without anyone seeing it.
    """
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def chemical_cache_key(smiles: Any) -> str:
    """Stable chemical identity used to match cached OPERA/Abraham rows.

    InChIKey is preferred because it is independent of SMILES writing style.
    Canonical SMILES is the fallback for the rare molecule whose InChI cannot be
    generated; an empty string means the row cannot participate in caching.
    """
    mol = mol_from_smiles(clean_smiles(smiles))
    if mol is None:
        return ""
    try:
        key = clean_text(Chem.MolToInchiKey(mol))
    except Exception:
        key = ""
    if key:
        return key
    try:
        return "SMILES:" + Chem.MolToSmiles(mol)
    except Exception:
        return ""


def chemical_cache_keys(df: pd.DataFrame, smiles_col: str) -> pd.Series:
    if smiles_col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[smiles_col].map(chemical_cache_key).astype(object)


def atom_count(mol: Chem.Mol, symbol: str) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


SMARTS_PATTERNS = {
    "carboxyl_group": "[CX3](=O)[OX2H1,OX1-]",
    "sulfonic_acid_group": "[SX4](=O)(=O)[OX2H1,OX1-]",
    "sulfonamide_group": "[SX4](=O)(=O)[NX3]",
    "phosphonic_acid_group": "[PX4](=O)([OX2H1,OX1-])([OX2H1,OX1-])",
    "carbonyl_group": "[CX3]=[OX1]",
}
SMARTS_MOLS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS_PATTERNS.items()}


def smarts_count(mol: Chem.Mol, pattern_name: str) -> int:
    patt = SMARTS_MOLS[pattern_name]
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def _longest_simple_path(mol: Chem.Mol, atom_indices: set[int]) -> int:
    """Return the longest simple path within an induced atom subgraph."""

    if not atom_indices:
        return 0

    def walk(idx: int, seen: set[int]) -> int:
        best = len(seen)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in atom_indices and nidx not in seen:
                best = max(best, walk(nidx, seen | {nidx}))
        return best

    return max(walk(idx, {idx}) for idx in atom_indices)


def _connected_component_count(mol: Chem.Mol, atom_indices: set[int]) -> int:
    remaining = set(atom_indices)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            idx = stack.pop()
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in remaining:
                    remaining.remove(nidx)
                    stack.append(nidx)
    return components


def atom_annotations(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Return one structural-label record per atom, in atom-index order.

    This is the single definition of terms like "terminal CF3" or
    "perfluoroether oxygen".  The scalar ``rdkit_*`` counts below are derived
    from these records, and ``pfas_atom_map.py`` projects the same records onto
    canonical-SMILES token positions.  Keeping both on one definition is what
    lets an atom-level attention map be compared against the matching scalar
    descriptor without the two silently meaning different things.
    """
    group_members: dict[str, set[int]] = {}
    for name, patt in SMARTS_MOLS.items():
        members: set[int] = set()
        if patt is not None:
            for match in mol.GetSubstructMatches(patt):
                members.update(match)
        group_members[name] = members

    records: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        symbol = atom.GetSymbol()
        neighbors = list(atom.GetNeighbors())
        f_neighbors = sum(1 for nbr in neighbors if nbr.GetSymbol() == "F")
        h_count = atom.GetTotalNumHs()
        is_carbon = symbol == "C"
        is_fluorinated_carbon = is_carbon and f_neighbors > 0
        # Heavy-atom degree of exactly two carbons; implicit hydrogens are not
        # neighbors here, so a terminal -OH or a carbonyl O is excluded.
        is_ether_oxygen = (
            symbol == "O"
            and atom.GetFormalCharge() == 0
            and len(neighbors) == 2
            and all(nbr.GetSymbol() == "C" for nbr in neighbors)
        )
        record: dict[str, Any] = {
            "atom_index": index,
            "symbol": symbol,
            "formal_charge": atom.GetFormalCharge(),
            "fluorine_neighbor_count": f_neighbors,
            "hydrogen_count": h_count,
            "is_fluorinated_C": is_fluorinated_carbon,
            "is_nonfluorinated_C": is_carbon and f_neighbors == 0,
            "is_hydrogenated_C": is_carbon and h_count > 0,
            "is_partially_fluorinated_C": is_fluorinated_carbon and h_count > 0,
            "is_terminal_CF3": is_fluorinated_carbon and f_neighbors >= 3,
            "is_ether_O": is_ether_oxygen,
            "is_perfluoroether_O": is_ether_oxygen
            and all(
                any(nn.GetSymbol() == "F" for nn in nbr.GetNeighbors()) for nbr in neighbors
            ),
        }
        for name, members in group_members.items():
            record[f"in_{name}"] = index in members
        records.append(record)
    return records


def fluorinated_carbon_summary(
    mol: Chem.Mol,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    fluorinated_carbons = {r["atom_index"] for r in records if r["is_fluorinated_C"]}

    longest_chain = _longest_simple_path(mol, fluorinated_carbons)
    branch_points = sum(
        sum(1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() in fluorinated_carbons) >= 3
        for idx in fluorinated_carbons
    )
    fluorinated_count = len(fluorinated_carbons)
    return {
        "rdkit_F_count": atom_count(mol, "F"),
        "rdkit_fluorinated_C_count": fluorinated_count,
        "rdkit_nonfluorinated_C_count": sum(1 for r in records if r["is_nonfluorinated_C"]),
        "rdkit_hydrogenated_C_count": sum(1 for r in records if r["is_hydrogenated_C"]),
        "rdkit_partially_fluorinated_C_count": sum(1 for r in records if r["is_partially_fluorinated_C"]),
        "rdkit_terminal_CF3_count": sum(1 for r in records if r["is_terminal_CF3"]),
        "rdkit_longest_fluorinated_C_chain": longest_chain,
        "rdkit_fluorinated_segment_count": _connected_component_count(mol, fluorinated_carbons),
        "rdkit_fluorinated_branch_point_count": branch_points,
        "rdkit_fluorinated_chain_compactness": (
            np.nan if fluorinated_count == 0 else longest_chain / fluorinated_count
        ),
    }


def charge_summary(mol: Chem.Mol) -> dict[str, int]:
    return {
        "rdkit_positive_charge_count": sum(max(atom.GetFormalCharge(), 0) for atom in mol.GetAtoms()),
        "rdkit_negative_charge_count": sum(max(-atom.GetFormalCharge(), 0) for atom in mol.GetAtoms()),
    }


def pfas_headgroup_summary(mol: Chem.Mol) -> dict[str, int]:
    carboxyl = smarts_count(mol, "carboxyl_group")
    carbonyl = smarts_count(mol, "carbonyl_group")
    return {
        "rdkit_carboxyl_group_count": carboxyl,
        "rdkit_sulfonic_acid_group_count": smarts_count(mol, "sulfonic_acid_group"),
        "rdkit_sulfonamide_group_count": smarts_count(mol, "sulfonamide_group"),
        "rdkit_phosphonic_acid_group_count": smarts_count(mol, "phosphonic_acid_group"),
        "rdkit_noncarboxyl_carbonyl_count": max(carbonyl - carboxyl, 0),
    }


def descriptor_row(mol: Chem.Mol) -> dict[str, float | str]:
    # Label the atoms once and reuse; every count below is a view of the same
    # records, so they cannot drift apart and the SMARTS matching runs once.
    records = atom_annotations(mol)
    ether_count = sum(1 for record in records if record["is_ether_O"])
    perfluoroether_count = sum(1 for record in records if record["is_perfluoroether_O"])
    row: dict[str, float | str] = {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Canonical SMILES": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "Mw (g/mol)": Descriptors.MolWt(mol),
        # Include polar sulfur and phosphorus contributions because PFSA and
        # phosphorus-containing PFAS are part of the candidate bank.
        "rdkit_tpsa": rdMolDescriptors.CalcTPSA(mol, includeSandP=True),
        "rdkit_num_h_acceptors": Lipinski.NumHAcceptors(mol),
        "rdkit_num_h_donors": Lipinski.NumHDonors(mol),
        "rdkit_num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rdkit_ring_count": rdMolDescriptors.CalcNumRings(mol),
        "rdkit_num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rdkit_fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "rdkit_N_count": atom_count(mol, "N"),
        "rdkit_Cl_count": atom_count(mol, "Cl"),
        "rdkit_perfluoroether_O_count": perfluoroether_count,
        "rdkit_nonperfluoroether_O_count": max(ether_count - perfluoroether_count, 0),
    }
    row.update(fluorinated_carbon_summary(mol, records))
    row.update(pfas_headgroup_summary(mol))
    row.update(charge_summary(mol))
    return row



def build_rdkit_features(
    df: pd.DataFrame,
    smiles_col: str,
    key_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Chem.Mol | None]]:
    rows: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    mols: list[Chem.Mol | None] = []
    for idx, smiles_value in df[smiles_col].items():
        smiles = clean_smiles(smiles_value)
        mol = mol_from_smiles(smiles)
        mols.append(mol)
        if mol is None:
            rows.append({})
            problems.append(
                {
                    "excel_row": idx + 2,
                    "pfas_key": clean_text(df.at[idx, key_col]) if key_col in df.columns else "",
                    "smiles": smiles,
                    "problem": "RDKit could not parse SMILES",
                }
            )
            continue
        rows.append(descriptor_row(mol))
    return pd.DataFrame(rows), pd.DataFrame(problems), mols



def _sparse_fingerprint_rows(
    df: pd.DataFrame,
    mols: list[Chem.Mol | None],
    key_col: str,
    generator: Any,
    representation: str,
    radius: int | None = None,
    min_distance: int | None = None,
    max_distance: int | None = None,
) -> pd.DataFrame:
    """Return an unfolded sparse count fingerprint in long format.

    Feature identifiers are stored as strings because RDKit sparse identifiers can
    exceed Excel's exact integer precision. Counts remain integers.
    """
    rows: list[dict[str, Any]] = []
    for position, mol in enumerate(mols):
        if mol is None:
            continue
        key = clean_text(df.iloc[position][key_col]) or f"row_{position + 2}"
        fingerprint = generator.GetSparseCountFingerprint(mol)
        for feature_id, count in sorted(fingerprint.GetNonzeroElements().items()):
            row = {
                "PFAS_key": key,
                "excel_row": position + 2,
                "representation": representation,
                "feature_id": str(feature_id),
                "count": int(count),
            }
            if radius is not None:
                row["radius"] = int(radius)
            if min_distance is not None:
                row["min_distance"] = int(min_distance)
            if max_distance is not None:
                row["max_distance"] = int(max_distance)
            rows.append(row)
    return pd.DataFrame(rows)


def build_raw_fingerprint_outputs(
    df: pd.DataFrame,
    mols: list[Chem.Mol | None],
    key_col: str,
    morgan_radius: int,
    atom_pair_min_distance: int,
    atom_pair_max_distance: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if morgan_radius < 0:
        raise ValueError("Morgan radius must be nonnegative.")
    if atom_pair_min_distance < 1:
        raise ValueError("Atom-pair minimum distance must be at least 1.")
    if atom_pair_max_distance < atom_pair_min_distance:
        raise ValueError("Atom-pair maximum distance must be >= minimum distance.")

    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=morgan_radius,
        countSimulation=False,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
        includeRedundantEnvironments=False,
    )
    atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=atom_pair_min_distance,
        maxDistance=atom_pair_max_distance,
        includeChirality=False,
        use2D=True,
        countSimulation=True,
    )
    morgan_sparse = _sparse_fingerprint_rows(
        df,
        mols,
        key_col,
        morgan_generator,
        representation="morgan_count_sparse",
        radius=morgan_radius,
    )
    atom_pair_sparse = _sparse_fingerprint_rows(
        df,
        mols,
        key_col,
        atom_pair_generator,
        representation="atom_pair_count_sparse",
        min_distance=atom_pair_min_distance,
        max_distance=atom_pair_max_distance,
    )
    parameters = pd.DataFrame(
        [
            {
                "representation": "morgan_count_sparse",
                "algorithm": "RDKit Morgan",
                "storage": "unfolded sparse count",
                "radius": int(morgan_radius),
                "min_distance": np.nan,
                "max_distance": np.nan,
                "include_chirality": False,
                "use_bond_types": True,
                "include_ring_membership": True,
                "include_redundant_environments": False,
                "use_2D_topological_distance": np.nan,
                "count_simulation": False,
                "folded_length": np.nan,
                "rdkit_version": rdBase.rdkitVersion,
            },
            {
                "representation": "atom_pair_count_sparse",
                "algorithm": "RDKit atom-pair",
                "storage": "unfolded sparse count",
                "radius": np.nan,
                "min_distance": int(atom_pair_min_distance),
                "max_distance": int(atom_pair_max_distance),
                "include_chirality": False,
                "use_bond_types": np.nan,
                "include_ring_membership": np.nan,
                "include_redundant_environments": np.nan,
                "use_2D_topological_distance": True,
                "count_simulation": True,
                "folded_length": np.nan,
                "rdkit_version": rdBase.rdkitVersion,
            },
        ]
    )
    return morgan_sparse, atom_pair_sparse, parameters

def remove_generated_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        col
        for col in df.columns
        if col not in GENERATED_EXACT_COLUMNS
        and col not in OVERLAPPING_MANUAL_COLUMNS
        and not any(str(col).startswith(prefix) for prefix in GENERATED_PREFIXES)
    ]
    return df[keep_cols].copy()


def blank_classification_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series([""] * n_rows, dtype="object") for col in CLASS_COLUMNS})


def pfas_atlas_smiles(value: Any) -> tuple[str | None, str]:
    smiles = clean_smiles(value)
    if not smiles or smiles == "-":
        return None, "blank_smiles"
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, "rdkit_parse_failed"

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    def looks_like_pfas_fragment(fragment: Chem.Mol) -> bool:
        symbols = {atom.GetSymbol() for atom in fragment.GetAtoms()}
        return "C" in symbols and "F" in symbols

    pfas_fragments = [fragment for fragment in fragments if looks_like_pfas_fragment(fragment)]
    if not pfas_fragments:
        return None, "no_C_and_F_fragment"
    if len(pfas_fragments) > 1:
        return None, "multiple_C_and_F_fragments"
    return Chem.MolToSmiles(pfas_fragments[0], canonical=True, isomericSmiles=False), "ok"


def load_pfas_atlas_classifier(source_dir: Path):
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"PFAS-Atlas source directory not found: {source_dir}")
    if not (source_dir / "classification_helper").exists():
        raise FileNotFoundError(f"PFAS-Atlas classification_helper not found under: {source_dir}")
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import classification_helper.classify_pfas as classify_module  # type: ignore

    original_calculate_mhfp = classify_module.calculate_MHFP
    if not getattr(original_calculate_mhfp, "_pfas_features_overflow_safe", False):

        def calculate_mhfp_overflow_safe(smiles: str):
            try:
                return original_calculate_mhfp(smiles)
            except OverflowError:
                # PFAS-Atlas only checks MHFP length here; let valid molecules continue to rule classification.
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return [-1]
                return [0, 0]

        calculate_mhfp_overflow_safe._pfas_features_overflow_safe = True
        classify_module.calculate_MHFP = calculate_mhfp_overflow_safe

    return classify_module.classify_pfas_molecule


def run_pfas_atlas_classification(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    classify_pfas_molecule = load_pfas_atlas_classifier(args.pfas_atlas_source)
    rows = []
    audit_rows = []
    classified_rows = 0
    failed_rows = 0
    unclassifiable_rows = 0

    for idx, source_row in df.reset_index(drop=True).iterrows():
        raw_smiles = source_row[args.smiles_col]
        class_smiles, status = pfas_atlas_smiles(raw_smiles)
        first_class = ""
        second_class = ""
        error = ""

        if class_smiles is None:
            unclassifiable_rows += 1
        else:
            try:
                if args.classification_verbose:
                    classification = classify_pfas_molecule(class_smiles)
                else:
                    sink = io.StringIO()
                    with contextlib.redirect_stdout(sink):
                        classification = classify_pfas_molecule(class_smiles)
                first_class = clean_text(classification[0])
                second_class = clean_text(classification[1])
                status = "classified"
                classified_rows += 1
            except Exception as exc:
                status = "classification_failed"
                error = f"{type(exc).__name__}: {exc}"
                failed_rows += 1

        rows.append({"First_Class": first_class, "Second_Class": second_class})
        audit_rows.append(
            {
                "excel_row": idx + 2,
                "pfas_key": clean_text(source_row.get(args.key_col, "")),
                "smiles": clean_text(raw_smiles),
                "classification_smiles": class_smiles or "",
                "classification_status": status,
                "classification_error": error,
                "First_Class": first_class,
                "Second_Class": second_class,
            }
        )

    info = {
        "classification_source": "pfas_atlas",
        "classification_mode": args.classification_mode,
        "classification_pfas_atlas_source": str(args.pfas_atlas_source),
        "classification_mhfp_overflow_guard": "enabled",
        "classification_note": "",
        "classification_attempted_rows": int(len(df)),
        "classification_classified_rows": classified_rows,
        "classification_failed_rows": failed_rows,
        "classification_unclassifiable_rows": unclassifiable_rows,
    }
    return pd.DataFrame(rows), pd.DataFrame(audit_rows), info


def resolve_classification_features(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mode = args.classification_mode

    if mode == "skip":
        info = {
            "classification_source": "skipped",
            "classification_mode": mode,
            "classification_pfas_atlas_source": str(args.pfas_atlas_source),
            "classification_mhfp_overflow_guard": "not_used",
            "classification_note": "",
            "classification_attempted_rows": 0,
            "classification_classified_rows": 0,
            "classification_failed_rows": 0,
            "classification_unclassifiable_rows": 0,
        }
        return blank_classification_frame(len(df)), pd.DataFrame(columns=CLASSIFICATION_AUDIT_COLUMNS), info

    try:
        return run_pfas_atlas_classification(df, args)
    except Exception as exc:
        raise RuntimeError(
            "PFAS-Atlas classification failed. Run this script inside the pfas-atlas "
            "environment or use --classification-mode skip.\n"
            f"PFAS-Atlas source: {args.pfas_atlas_source}\n"
            f"Error: {type(exc).__name__}: {exc}"
        ) from exc


def blank_abraham_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series([np.nan] * n_rows) for col in ABRAHAM_COLUMNS})


def abraham_from_existing(df: pd.DataFrame) -> pd.DataFrame:
    out = blank_abraham_frame(len(df))
    for col in ABRAHAM_COLUMNS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").reset_index(drop=True)
    return out


def abraham_missing_mask(features: pd.DataFrame) -> pd.Series:
    if features.empty:
        return pd.Series([], dtype=bool)
    return features.reindex(columns=ABRAHAM_COLUMNS).apply(pd.to_numeric, errors="coerce").isna().any(axis=1)


def build_abraham_missing_report(
    args: argparse.Namespace,
    source: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    source = source.reset_index(drop=True)
    features = features.reset_index(drop=True).reindex(columns=ABRAHAM_COLUMNS)
    rows: list[dict[str, Any]] = []
    for idx, row in features.iterrows():
        missing = [col for col in ABRAHAM_COLUMNS if pd.isna(pd.to_numeric(row.get(col), errors="coerce"))]
        if not missing:
            continue
        rows.append({
            "excel_row": idx + 2,
            "pfas_key": clean_text(source.at[idx, args.key_col]) if args.key_col in source.columns else "",
            "compound_name": clean_text(source.at[idx, "Compound Full Name"]) if "Compound Full Name" in source.columns else "",
            "smiles": clean_text(source.at[idx, args.smiles_col]) if args.smiles_col in source.columns else "",
            "status": "all_abraham_descriptors_missing" if len(missing) == len(ABRAHAM_COLUMNS) else "partial_abraham_descriptors_missing",
            "missing_abraham_count": len(missing),
            "missing_abraham_columns": "; ".join(missing),
        })
    return pd.DataFrame(rows)


def _normalized_column_lookup(columns: Any) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]+", "", str(col).lower()): str(col) for col in columns}


def _align_descriptor_rows(
    raw: pd.DataFrame,
    n_rows: int,
    expected_ids: pd.Series | None,
) -> pd.DataFrame:
    if expected_ids is None:
        if len(raw) != n_rows:
            raise ValueError(f"PaDEL descriptor row count {len(raw)} does not match expected rows {n_rows}.")
        return raw.reset_index(drop=True)
    expected = expected_ids.map(clean_text).reset_index(drop=True)
    id_col = next((col for col in ("Name", "MoleculeID", "ID", "Compound") if col in raw.columns), None)
    if id_col is not None:
        observed = raw[id_col].map(clean_text)
        if expected.ne("").all() and expected.is_unique and observed.is_unique and set(expected) == set(observed):
            aligned = raw.assign(__molecule_id=observed).set_index("__molecule_id").reindex(expected)
            return aligned.reset_index(drop=True)
    if len(raw) != n_rows:
        raise ValueError(
            f"PaDEL descriptor row count {len(raw)} does not match expected rows {n_rows}, "
            "and molecule IDs could not be aligned."
        )
    return raw.reset_index(drop=True)


def extract_abraham_columns(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Map PaDEL/MLFER source columns onto the stable abraham_* output names."""
    lookup = _normalized_column_lookup(raw.columns)
    out = blank_abraham_frame(len(raw))
    out.index = raw.index
    missing_sources: list[str] = []
    for source_col, output_col in ABRAHAM_SOURCE_TO_OUTPUT.items():
        actual = lookup.get(re.sub(r"[^a-z0-9]+", "", source_col.lower()))
        if actual is None:
            # A merged reuse file may already contain the stable output names.
            actual = lookup.get(re.sub(r"[^a-z0-9]+", "", output_col.lower()))
        if actual is None:
            missing_sources.append(source_col)
            continue
        out[output_col] = pd.to_numeric(raw[actual], errors="coerce")
    if len(missing_sources) == len(ABRAHAM_SOURCE_COLUMNS):
        raise ValueError(
            f"No Abraham/MLFER columns were found in {source_label}. Expected source columns: "
            + ", ".join(ABRAHAM_SOURCE_COLUMNS)
        )
    return out


def read_abraham_csv(
    path: Path,
    n_rows: int,
    expected_ids: pd.Series | None = None,
) -> pd.DataFrame:
    raw = _align_descriptor_rows(pd.read_csv(path), n_rows, expected_ids)
    return extract_abraham_columns(raw, str(path)).reset_index(drop=True)


def candidate_padel_descriptor_paths(smi_path: Path) -> list[Path]:
    stems = [smi_path.stem]
    if smi_path.stem.endswith("_QSAR-ready_smi"):
        stems.append(smi_path.stem.removesuffix("_QSAR-ready_smi"))
    candidates: list[Path] = []
    for stem in stems:
        candidates.extend([
            smi_path.parent / f"{stem}_PadelDesc.csv",
            smi_path.parent / f"{stem}_QSAR-ready_smi_PadelDesc.csv",
        ])
        candidates.extend(sorted(smi_path.parent.glob(f"{stem}*PadelDesc.csv")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def read_abraham_from_opera_sidecars(
    smi_path: Path,
    n_rows: int,
    expected_ids: pd.Series,
) -> tuple[pd.DataFrame, Path]:
    errors: list[str] = []
    for path in candidate_padel_descriptor_paths(smi_path):
        if not path.exists():
            continue
        try:
            return read_abraham_csv(path, n_rows, expected_ids), path
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    detail = " | ".join(errors) if errors else "no PaDEL descriptor sidecar was created"
    raise FileNotFoundError(f"Could not extract Abraham descriptors for {smi_path}: {detail}")


def candidate_abraham_csv_paths(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.abraham_csv is not None:
        candidates.append(args.abraham_csv)
    candidates.extend([
        # The work directory is listed first because run_opera writes the
        # authoritative merged cache there.
        args.work_dir / f"{args.output.stem}_abraham.csv",
        args.work_dir / "pfas_abraham.csv",
        args.work_dir / f"{args.input.stem}_abraham.csv",
        args.output.with_name(f"{args.output.stem}_abraham.csv"),
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def blank_opera_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series([np.nan] * n_rows) for col in OPERA_COLUMNS})


def is_missing_opera_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in OPERA_MISSING_STRINGS:
        return True
    return False


def build_opera_missing_report(
    args: argparse.Namespace,
    source: pd.DataFrame,
    opera_features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    source = source.reset_index(drop=True)
    opera_features = opera_features.reset_index(drop=True)
    for idx, row in opera_features.iterrows():
        missing_cols = [
            col
            for col in OPERA_COLUMNS
            if col in opera_features.columns and is_missing_opera_value(row[col])
        ]
        if not missing_cols:
            continue
        critical_cols = [
            col
            for col in OPERA_CRITICAL_COLUMNS
            if col in opera_features.columns and is_missing_opera_value(row[col])
        ]
        if critical_cols:
            status = "critical_prediction_missing"
        elif set(missing_cols).issubset(OPERA_OPTIONAL_PKA_COLUMNS):
            status = "pka_endpoint_not_predicted"
        else:
            status = "noncritical_missing"
        rows.append(
            {
                "excel_row": idx + 2,
                "pfas_key": clean_text(source.at[idx, args.key_col])
                if args.key_col in source.columns
                else "",
                "compound_name": clean_text(source.at[idx, "Compound Full Name"])
                if "Compound Full Name" in source.columns
                else "",
                "smiles": clean_text(source.at[idx, args.smiles_col])
                if args.smiles_col in source.columns
                else "",
                "status": status,
                "missing_opera_count": len(missing_cols),
                "missing_critical_count": len(critical_cols),
                "missing_opera_columns": "; ".join(missing_cols),
                "missing_critical_columns": "; ".join(critical_cols),
            }
        )
    return pd.DataFrame(rows)


def critical_opera_missing_mask(opera_features: pd.DataFrame) -> pd.Series:
    if opera_features.empty:
        return pd.Series([], dtype=bool)
    missing = pd.DataFrame(
        {
            col: opera_features[col].map(is_missing_opera_value)
            for col in OPERA_CRITICAL_COLUMNS
            if col in opera_features.columns
        },
        index=opera_features.index,
    )
    if missing.empty:
        return pd.Series([False] * len(opera_features), index=opera_features.index)
    return missing.any(axis=1)


def opera_from_existing(df: pd.DataFrame) -> pd.DataFrame:
    out = blank_opera_frame(len(df))
    for col in OPERA_COLUMNS:
        if col in df.columns:
            out[col] = df[col].reset_index(drop=True)
    return out


def molecule_id_order(value: Any) -> int | None:
    match = re.fullmatch(r"Molecule_(\d+)", clean_text(value))
    return int(match.group(1)) if match else None


def read_opera_csv(path: Path, n_rows: int, expected_ids: pd.Series | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if expected_ids is not None and "MoleculeID" in raw.columns:
        expected = expected_ids.map(clean_text).reset_index(drop=True)
        observed = raw["MoleculeID"].map(clean_text)
        if (
            expected.ne("").all()
            and expected.is_unique
            and observed.is_unique
            and set(expected) == set(observed)
        ):
            raw = raw.copy()
            raw["__molecule_id"] = observed
            raw = raw.set_index("__molecule_id").reindex(expected).reset_index(drop=True)
    if "MoleculeID" in raw.columns:
        raw = raw.copy()
        raw["__molecule_order"] = raw["MoleculeID"].map(molecule_id_order)
        if raw["__molecule_order"].notna().all():
            expected = set(range(1, n_rows + 1))
            observed = set(raw["__molecule_order"].astype(int))
            if observed == expected:
                raw = raw.sort_values("__molecule_order")
        raw = raw.drop(columns=["__molecule_order"], errors="ignore")
    if len(raw) != n_rows:
        raise ValueError(f"OPERA CSV row count {len(raw)} does not match PFAS rows {n_rows}: {path}")
    out = blank_opera_frame(n_rows)
    for col in OPERA_COLUMNS:
        if col in raw.columns:
            out[col] = raw[col].reset_index(drop=True)
    return out


def candidate_opera_csv_paths(args: argparse.Namespace) -> list[Path]:
    candidates = []
    if args.opera_csv is not None:
        candidates.append(args.opera_csv)
    candidates.extend(
        [
            # The work directory is listed first because run_opera writes the
            # authoritative merged cache there.
            args.work_dir / f"{args.output.stem}_opera.csv",
            args.work_dir / "pfas_opera.csv",
            args.work_dir / f"{args.input.stem}_opera.csv",
            args.output.with_name(f"{args.output.stem}_opera.csv"),
        ]
    )
    unique = []
    seen = set()
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique

def cache_row_keys(raw: pd.DataFrame, legacy_key_to_chem: dict[str, str]) -> pd.Series | None:
    """Chemical identity of each cached row.

    Caches written by this script carry the key directly. Legacy caches predate the
    key column and are migrated by matching their MoleculeID against the manual
    workbook's key column, then taking that row's chemical identity.
    """
    if CACHE_KEY_COLUMN in raw.columns:
        keys = raw[CACHE_KEY_COLUMN].map(clean_text)
        if keys.ne("").any():
            return keys
    if "MoleculeID" in raw.columns and legacy_key_to_chem:
        keys = raw["MoleculeID"].map(lambda value: legacy_key_to_chem.get(normalize_key(value), ""))
        if keys.ne("").any():
            return keys
    return None


def align_cache_by_key(
    raw: pd.DataFrame,
    cache_keys: pd.Series,
    wanted_keys: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reindex a cache frame onto the wanted chemical keys.

    Returns the aligned rows (blank where the cache has no entry) and a boolean
    hit mask, both indexed like wanted_keys.
    """
    frame = raw.copy()
    frame["__chem_key"] = pd.Series(cache_keys.to_numpy(), index=frame.index)
    frame = frame[frame["__chem_key"].ne("")]
    frame = frame.drop_duplicates(subset="__chem_key", keep="last").set_index("__chem_key")
    hit = wanted_keys.ne("") & wanted_keys.isin(frame.index)
    aligned = frame.reindex(wanted_keys.where(hit).to_numpy())
    aligned.index = wanted_keys.index
    return aligned, hit


def legacy_key_map(df: pd.DataFrame, args: argparse.Namespace, chem_keys: pd.Series) -> dict[str, str]:
    if args.key_col not in df.columns:
        return {}
    return {
        normalize_key(value): chem
        for value, chem in zip(df[args.key_col], chem_keys, strict=True)
        if normalize_key(value) and chem
    }


def load_opera_cache(
    df: pd.DataFrame,
    args: argparse.Namespace,
    chem_keys: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Fill OPERA columns from cached CSVs, matched by chemical identity."""
    out = blank_opera_frame(len(df))
    out.index = df.index
    for col in OPERA_TEXT_COLUMNS:
        out[col] = out[col].astype("object")
    filled = pd.Series(False, index=df.index)
    legacy_map = legacy_key_map(df, args, chem_keys)
    used: list[str] = []
    errors: list[str] = []

    for path in candidate_opera_csv_paths(args):
        if not path.exists() or filled.all():
            continue
        try:
            raw = pd.read_csv(path)
            cache_keys = cache_row_keys(raw, legacy_map)
            if cache_keys is None:
                errors.append(
                    f"{path}: no {CACHE_KEY_COLUMN} column, and MoleculeID values did not "
                    f"match the input {args.key_col} column"
                )
                continue
            aligned, hit = align_cache_by_key(raw, cache_keys, chem_keys)
            new_hits = hit & ~filled
            if not new_hits.any():
                continue
            for col in OPERA_COLUMNS:
                if col in aligned.columns:
                    out.loc[new_hits, col] = aligned.loc[new_hits, col]
            filled = filled | new_hits
            used.append(f"{path} ({int(new_hits.sum())} rows)")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    # A cached row is only usable if its non-optional endpoints are present.
    # Blank pKa_a/pKa_b is expected for non-ionizable PFAS and never a cache miss.
    usable = filled & ~critical_opera_missing_mask(out)
    if (~usable).any():
        out.loc[~usable, OPERA_COLUMNS] = np.nan
    info = {
        "opera_cache_files": "; ".join(used),
        "opera_cache_hits": int(usable.sum()),
        "opera_cache_incomplete_rows": int((filled & ~usable).sum()),
        "opera_cache_misses": int((~usable).sum()),
        "opera_cache_errors": " | ".join(errors),
    }
    return out, usable, info


def load_abraham_cache(
    df: pd.DataFrame,
    args: argparse.Namespace,
    chem_keys: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Fill Abraham columns from cached CSVs, matched by chemical identity."""
    out = blank_abraham_frame(len(df))
    out.index = df.index
    filled = pd.Series(False, index=df.index)
    legacy_map = legacy_key_map(df, args, chem_keys)
    used: list[str] = []
    errors: list[str] = []

    for path in candidate_abraham_csv_paths(args):
        if not path.exists() or filled.all():
            continue
        try:
            raw = pd.read_csv(path)
            cache_keys = cache_row_keys(raw, legacy_map)
            if cache_keys is None:
                errors.append(
                    f"{path}: no {CACHE_KEY_COLUMN} column, and MoleculeID values did not "
                    f"match the input {args.key_col} column"
                )
                continue
            aligned, hit = align_cache_by_key(raw, cache_keys, chem_keys)
            values = extract_abraham_columns(aligned, str(path))
            complete = hit & values.notna().any(axis=1)
            new_hits = complete & ~filled
            if not new_hits.any():
                continue
            for col in ABRAHAM_COLUMNS:
                out.loc[new_hits, col] = values.loc[new_hits, col]
            filled = filled | new_hits
            used.append(f"{path} ({int(new_hits.sum())} rows)")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    info = {
        "abraham_cache_files": "; ".join(used),
        "abraham_cache_hits": int(filled.sum()),
        "abraham_cache_misses": int((~filled).sum()),
        "abraham_cache_errors": " | ".join(errors),
    }
    return out, filled, info


def make_safe_opera_ids(index: pd.Index) -> pd.Series:
    """Generate OPERA-safe temporary molecule IDs.

    Do not pass curated PFAS abbreviations directly to OPERA/PaDEL because
    names such as "6:2 FTSA", "H-UPFPr-O/OH", or unicode-containing IDs can
    break descriptor calculation on Windows. These IDs are only used inside
    OPERA; the final workbook keeps the original Abbreviation column.
    """
    return pd.Series([f"PFAS_{int(i) + 1:04d}" for i in index], index=index)


def write_opera_smi(
    df: pd.DataFrame,
    smiles_col: str,
    key_col: str,
    path: Path,
    molecule_ids: pd.Series | None = None,
) -> None:
    blanks = df[smiles_col].map(clean_smiles).eq("")
    if blanks.any():
        rows = ", ".join(str(i + 2) for i in df.index[blanks][:10])
        raise ValueError(f"Cannot run OPERA with blank SMILES. First blank Excel rows: {rows}")

    if molecule_ids is None:
        keys = df[key_col].map(clean_text)
        id_source = key_col
    else:
        keys = molecule_ids.reindex(df.index).map(clean_text)
        id_source = "generated OPERA-safe molecule IDs"

    if keys.eq("").any() or not keys.is_unique:
        raise ValueError(f"Cannot run OPERA with blank or duplicate molecule IDs in {id_source!r}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{smiles}\t{mol_id}"
        for smiles, mol_id in zip(df[smiles_col].map(clean_smiles), keys, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def build_opera_command(args: argparse.Namespace, smi_path: Path, csv_path: Path) -> list[str]:
    cmd = [
        str(args.opera_exe),
        "--SMI",
        str(smi_path),
        "-o",
        str(csv_path),
        "-e",
        *OPERA_ENDPOINTS,
        "-v",
        "2",
    ]
    if args.opera_standardize:
        cmd.insert(-2, "-st")
    return cmd


def remove_stale_opera_sidecars(smi_path: Path, csv_path: Path) -> None:
    """Delete known OPERA/PaDEL sidecars before rerun so stale partial files cannot be reused."""
    csv_path.unlink(missing_ok=True)
    stems = [smi_path.stem]
    if smi_path.stem.endswith("_QSAR-ready_smi"):
        stems.append(smi_path.stem.removesuffix("_QSAR-ready_smi"))
    suffixes = [
        "_PadelDesc.csv",
        "_PadelFP.csv",
        "_Summary_file.csv",
        "_QSAR-ready_kek.sdf",
        "_QSAR-ready_saltInfo.csv",
        "_QSAR-ready_smi.smi",
        "_QSAR-ready_smi_PadelDesc.csv",
        "_QSAR-ready_smi_PadelFP.csv",
    ]
    for stem in stems:
        for suffix in suffixes:
            (smi_path.parent / f"{stem}{suffix}").unlink(missing_ok=True)


def cleanup_opera_run_files(smi_path: Path, csv_path: Path) -> None:
    remove_stale_opera_sidecars(smi_path, csv_path)
    smi_path.unlink(missing_ok=True)


def cleanup_previous_opera_intermediates(args: argparse.Namespace) -> int:
    if args.keep_opera_intermediates or not args.work_dir.exists():
        return 0
    stem = args.output.stem
    removed = 0
    batch_csv_pattern = re.compile(rf"^{re.escape(stem)}_opera_\d{{4}}_\d{{4}}\.csv$")
    prefixes = (
        f"{stem}_opera_input",
        f"{stem}_opera_singleton",
        f"{stem}_opera_singleton_input",
    )
    for path in args.work_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name == f"{stem}_opera.csv":
            continue
        if batch_csv_pattern.fullmatch(name) or name.startswith(prefixes):
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def run_opera_command(cmd: list[str], smi_path: Path, csv_path: Path) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "OPERA failed.\n"
            f"Exit code: {exc.returncode}\n"
            f"Command: {' '.join(map(str, cmd))}\n\n"
            f"stdout:\n{(exc.stdout or '').strip() or '[empty]'}\n\n"
            f"stderr:\n{(exc.stderr or '').strip() or '[empty]'}\n\n"
            f"Input SMI kept at: {smi_path}\n"
            f"Output CSV target: {csv_path}"
        ) from exc

    if not csv_path.exists():
        raise RuntimeError(
            "OPERA finished without creating the expected CSV.\n"
            f"Command: {' '.join(map(str, cmd))}\n"
            f"Input SMI: {smi_path}\n"
            f"Expected CSV: {csv_path}\n\n"
            f"stdout:\n{(completed.stdout or '').strip() or '[empty]'}\n\n"
            f"stderr:\n{(completed.stderr or '').strip() or '[empty]'}"
        )

    return completed


def retry_critical_opera_rows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    opera_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    retry_indices = list(opera_features.index[critical_opera_missing_mask(opera_features)])
    max_attempts = max(1, int(args.opera_singleton_retries))
    info: dict[str, Any] = {
        "opera_singleton_fallback": "enabled",
        "opera_singleton_fallback_attempted": len(retry_indices),
        "opera_singleton_fallback_succeeded": 0,
        "opera_singleton_fallback_failed": 0,
        "opera_singleton_fallback_max_attempts": max_attempts,
        "opera_singleton_fallback_total_attempts": 0,
        "opera_singleton_fallback_excel_rows": ", ".join(str(i + 2) for i in retry_indices),
        "opera_singleton_fallback_failed_rows": "",
    }
    stdout_tails: list[str] = []
    stderr_tails: list[str] = []
    failed_rows: list[str] = []

    if not retry_indices:
        return opera_features, info, stdout_tails, stderr_tails

    print(
        "OPERA critical predictions are missing for "
        f"{len(retry_indices)} rows; rerunning those rows one-by-one."
    )
    fixed = opera_features.copy()
    for col in OPERA_TEXT_COLUMNS:
        if col in fixed.columns:
            fixed[col] = fixed[col].astype("object")
    for retry_number, idx in enumerate(retry_indices, start=1):
        singleton = df.loc[[idx]].copy()
        singleton_ids = make_safe_opera_ids(singleton.index)
        excel_row = idx + 2
        pfas_key = clean_text(singleton.iloc[0].get(args.key_col, ""))
        fixed_this_row = False
        last_problem = "missing critical OPERA fields"

        for attempt in range(1, max_attempts + 1):
            tag = f"{idx + 1:04d}_try{attempt:02d}"
            smi_path = args.work_dir / f"{args.output.stem}_opera_singleton_input_{tag}.smi"
            csv_path = args.work_dir / f"{args.output.stem}_opera_singleton_{tag}.csv"
            info["opera_singleton_fallback_total_attempts"] += 1

            attempt_suffix = "" if max_attempts == 1 else f" attempt {attempt}/{max_attempts}"
            print(
                f"Running OPERA singleton fallback {retry_number}/{len(retry_indices)}"
                f"{attempt_suffix}: Excel row {excel_row} {pfas_key}"
            )
            try:
                remove_stale_opera_sidecars(smi_path, csv_path)
                write_opera_smi(
                    singleton,
                    args.smiles_col,
                    args.key_col,
                    smi_path,
                    molecule_ids=singleton_ids,
                )
                cmd = build_opera_command(args, smi_path, csv_path)
                completed = run_opera_command(cmd, smi_path=smi_path, csv_path=csv_path)
                singleton_frame = read_opera_csv(
                    csv_path,
                    1,
                    expected_ids=singleton_ids,
                )

                stdout_lines = (completed.stdout or "").strip().splitlines()
                stderr_lines = (completed.stderr or "").strip().splitlines()
                stdout_tails.append(
                    f"singleton Excel row {excel_row} attempt {attempt}: "
                    + " | ".join(stdout_lines[-5:])
                )
                if stderr_lines:
                    stderr_tails.append(
                        f"singleton Excel row {excel_row} attempt {attempt}: "
                        + " | ".join(stderr_lines[-5:])
                    )

                if bool(critical_opera_missing_mask(singleton_frame).iloc[0]):
                    last_problem = "missing critical OPERA fields"
                    if attempt < max_attempts:
                        print(
                            f"  Singleton attempt {attempt} still has missing critical "
                            f"OPERA fields for Excel row {excel_row}; retrying."
                        )
                    continue

                for col in OPERA_COLUMNS:
                    fixed.at[idx, col] = singleton_frame.at[0, col]
                info["opera_singleton_fallback_succeeded"] += 1
                fixed_this_row = True
                break
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
                stderr_tails.append(
                    f"singleton Excel row {excel_row} attempt {attempt}: {last_problem}"
                )
                if attempt < max_attempts:
                    print(
                        f"  Singleton attempt {attempt} failed for Excel row {excel_row}: "
                        f"{last_problem}; retrying."
                    )
            finally:
                if not args.keep_opera_intermediates:
                    cleanup_opera_run_files(smi_path, csv_path)

        if not fixed_this_row:
            failed_rows.append(str(excel_row))
            info["opera_singleton_fallback_failed"] += 1
            print(
                f"  Singleton fallback still has {last_problem} "
                f"for Excel row {excel_row} after {max_attempts} attempt(s)."
            )

    info["opera_singleton_fallback_failed_rows"] = ", ".join(failed_rows)
    return fixed, info, stdout_tails, stderr_tails


def retry_missing_abraham_rows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = features.reindex(columns=ABRAHAM_COLUMNS).apply(pd.to_numeric, errors="coerce")
    retry_indices = list(numeric.index[numeric.isna().all(axis=1)])
    max_attempts = max(1, int(args.opera_singleton_retries))
    info: dict[str, Any] = {
        "abraham_singleton_fallback_attempted": len(retry_indices),
        "abraham_singleton_fallback_succeeded": 0,
        "abraham_singleton_fallback_failed": 0,
        "abraham_singleton_fallback_total_attempts": 0,
        "abraham_singleton_fallback_excel_rows": ", ".join(str(i + 2) for i in retry_indices),
        "abraham_singleton_fallback_failed_rows": "",
    }
    if not retry_indices:
        return features, info

    print(
        f"Abraham descriptors are wholly missing for {len(retry_indices)} rows; "
        "rerunning those rows one-by-one."
    )
    fixed = features.copy()
    failed_rows: list[str] = []
    for retry_number, idx in enumerate(retry_indices, start=1):
        singleton = df.loc[[idx]].copy()
        singleton_ids = make_safe_opera_ids(singleton.index)
        excel_row = idx + 2
        pfas_key = clean_text(singleton.iloc[0].get(args.key_col, ""))
        recovered = False
        last_problem = "PaDEL Abraham descriptor sidecar unavailable"
        for attempt in range(1, max_attempts + 1):
            info["abraham_singleton_fallback_total_attempts"] += 1
            tag = f"{idx + 1:04d}_abraham_try{attempt:02d}"
            smi_path = args.work_dir / f"{args.output.stem}_abraham_singleton_input_{tag}.smi"
            csv_path = args.work_dir / f"{args.output.stem}_abraham_singleton_{tag}.csv"
            try:
                remove_stale_opera_sidecars(smi_path, csv_path)
                write_opera_smi(
                    singleton, args.smiles_col, args.key_col, smi_path, molecule_ids=singleton_ids
                )
                run_opera_command(
                    build_opera_command(args, smi_path, csv_path),
                    smi_path=smi_path,
                    csv_path=csv_path,
                )
                singleton_features, _sidecar = read_abraham_from_opera_sidecars(
                    smi_path, 1, singleton_ids
                )
                if singleton_features.reindex(columns=ABRAHAM_COLUMNS).isna().all(axis=1).iloc[0]:
                    last_problem = "all Abraham descriptor values remained missing"
                    continue
                for col in ABRAHAM_COLUMNS:
                    fixed.at[idx, col] = singleton_features.at[0, col]
                recovered = True
                info["abraham_singleton_fallback_succeeded"] += 1
                break
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            finally:
                if not args.keep_opera_intermediates:
                    cleanup_opera_run_files(smi_path, csv_path)
        if not recovered:
            failed_rows.append(str(excel_row))
            info["abraham_singleton_fallback_failed"] += 1
            print(
                f"  Abraham singleton fallback failed for Excel row {excel_row} "
                f"({pfas_key}): {last_problem}"
            )
        elif len(retry_indices) > 1:
            print(
                f"  Recovered Abraham descriptors {retry_number}/{len(retry_indices)}: "
                f"Excel row {excel_row} {pfas_key}"
            )
    info["abraham_singleton_fallback_failed_rows"] = ", ".join(failed_rows)
    return fixed, info


def run_opera(
    df: pd.DataFrame,
    args: argparse.Namespace,
    run_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run OPERA for the requested rows.

    run_index selects which rows of df to calculate; the returned frames carry
    those original row labels so the caller can merge them with cached rows.
    """
    if not args.opera_exe.exists():
        raise FileNotFoundError(f"OPERA executable not found: {args.opera_exe}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    subset = df if run_index is None else df.loc[run_index]
    calculate_abraham = args.abraham_mode in {"auto", "run"}
    batch_size = max(1, int(args.opera_batch_size))
    n_rows = len(subset)
    total_batches = (n_rows + batch_size - 1) // batch_size
    pre_run_cleanup_files = cleanup_previous_opera_intermediates(args)

    batch_frames: list[pd.DataFrame] = []
    abraham_batch_frames: list[pd.DataFrame] = []
    abraham_sidecars: list[str] = []
    abraham_errors: list[str] = []
    stdout_tails: list[str] = []
    stderr_tails: list[str] = []

    for batch_number, start in enumerate(range(0, n_rows, batch_size), start=1):
        stop = min(start + batch_size, n_rows)
        batch = subset.iloc[start:stop].copy()
        batch_opera_ids = make_safe_opera_ids(batch.index)
        batch_tag = f"{int(batch.index[0]) + 1:04d}_{int(batch.index[-1]) + 1:04d}"
        smi_path = args.work_dir / f"{args.output.stem}_opera_input_{batch_tag}.smi"
        csv_path = args.work_dir / f"{args.output.stem}_opera_{batch_tag}.csv"

        excel_rows = f"{int(batch.index[0]) + 2}-{int(batch.index[-1]) + 2}"
        remove_stale_opera_sidecars(smi_path, csv_path)
        print(
            f"Running OPERA batch {batch_number}/{total_batches}: "
            f"{len(batch)} molecules, Excel rows {excel_rows}"
        )
        write_opera_smi(batch, args.smiles_col, args.key_col, smi_path, molecule_ids=batch_opera_ids)
        completed = run_opera_command(
            build_opera_command(args, smi_path, csv_path),
            smi_path=smi_path,
            csv_path=csv_path,
        )
        batch_frames.append(
            read_opera_csv(csv_path, len(batch), expected_ids=batch_opera_ids).set_axis(batch.index)
        )

        if calculate_abraham:
            try:
                abraham_batch, sidecar = read_abraham_from_opera_sidecars(
                    smi_path, len(batch), batch_opera_ids
                )
                abraham_batch_frames.append(abraham_batch.set_axis(batch.index))
                abraham_sidecars.append(str(sidecar))
            except Exception as exc:
                message = f"batch {batch_number} Excel rows {excel_rows}: {type(exc).__name__}: {exc}"
                abraham_errors.append(message)
                abraham_batch_frames.append(blank_abraham_frame(len(batch)).set_axis(batch.index))
        else:
            abraham_batch_frames.append(blank_abraham_frame(len(batch)).set_axis(batch.index))

        stdout_lines = (completed.stdout or "").strip().splitlines()
        stderr_lines = (completed.stderr or "").strip().splitlines()
        stdout_tails.append(f"batch {batch_number}: " + " | ".join(stdout_lines[-5:]))
        if stderr_lines:
            stderr_tails.append(f"batch {batch_number}: " + " | ".join(stderr_lines[-5:]))

        if not args.keep_opera_intermediates:
            cleanup_opera_run_files(smi_path, csv_path)

    if batch_frames:
        opera_features = pd.concat(batch_frames, axis=0)
        abraham_features = pd.concat(abraham_batch_frames, axis=0)
    else:
        opera_features = blank_opera_frame(0).set_axis(subset.index)
        abraham_features = blank_abraham_frame(0).set_axis(subset.index)
    if len(opera_features) != n_rows or len(abraham_features) != n_rows:
        raise ValueError("Batched external-feature output row count does not match PFAS rows.")

    abraham_fallback_info: dict[str, Any] = {
        "abraham_singleton_fallback_attempted": 0,
        "abraham_singleton_fallback_succeeded": 0,
        "abraham_singleton_fallback_failed": 0,
        "abraham_singleton_fallback_total_attempts": 0,
        "abraham_singleton_fallback_excel_rows": "",
        "abraham_singleton_fallback_failed_rows": "",
    }
    if calculate_abraham:
        abraham_features, abraham_fallback_info = retry_missing_abraham_rows(
            df, args, abraham_features
        )
        still_all_missing = (
            abraham_features.reindex(columns=ABRAHAM_COLUMNS)
            .apply(pd.to_numeric, errors="coerce")
            .isna()
            .all(axis=1)
        )
        if args.abraham_mode == "run" and still_all_missing.any():
            rows = ", ".join(str(index + 2) for index in still_all_missing.index[still_all_missing])
            raise RuntimeError(
                "Required Abraham descriptors remain wholly missing after singleton fallback "
                f"for Excel rows: {rows}. Inspect OPERA/PaDEL sidecar generation."
            )

    fallback_info: dict[str, Any] = {
        "opera_singleton_fallback": "disabled" if args.no_opera_singleton_fallback else "enabled",
        "opera_singleton_fallback_attempted": 0,
        "opera_singleton_fallback_succeeded": 0,
        "opera_singleton_fallback_failed": 0,
        "opera_singleton_fallback_max_attempts": max(1, int(args.opera_singleton_retries)),
        "opera_singleton_fallback_total_attempts": 0,
        "opera_singleton_fallback_excel_rows": "",
        "opera_singleton_fallback_failed_rows": "",
    }
    if not args.no_opera_singleton_fallback:
        opera_features, fallback_info, fallback_stdout, fallback_stderr = retry_critical_opera_rows(
            df, args, opera_features
        )
        stdout_tails.extend(fallback_stdout)
        stderr_tails.extend(fallback_stderr)

    if args.abraham_mode == "skip":
        abraham_source = "skipped"
    elif args.abraham_mode in {"preserve", "reuse"}:
        abraham_source = "not_resolved_during_opera_run"
    elif abraham_errors:
        abraham_source = "partial_opera_padel_sidecars"
    else:
        abraham_source = "opera_padel_sidecars"

    info = {
        "opera_source": "run_batched_with_singleton_fallback" if fallback_info["opera_singleton_fallback_attempted"] else ("run_batched" if n_rows > batch_size else "run"),
        "opera_exe": str(args.opera_exe),
        "opera_calculated_rows": n_rows,
        "opera_batch_size": batch_size,
        "opera_batches": len(batch_frames),
        "opera_pre_run_intermediate_files_removed": pre_run_cleanup_files,
        **fallback_info,
        "opera_stdout": "\n".join(stdout_tails),
        "opera_stderr": "\n".join(stderr_tails),
        "abraham_source": abraham_source,
        "abraham_padel_sidecar_count": len(abraham_sidecars),
        "abraham_padel_sidecars": "\n".join(abraham_sidecars),
        "abraham_extraction_errors": "\n".join(abraham_errors),
        **abraham_fallback_info,
    }
    return opera_features, abraham_features, info


def update_feature_cache(
    path: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
    chem_keys: pd.Series,
    features: pd.DataFrame,
    value_columns: Sequence[str],
) -> dict[str, Any]:
    """Merge the current results into the on-disk cache, keyed by chemical identity.

    Entries for molecules that are no longer in the manual workbook are retained so
    that removing and later re-adding a PFAS does not force a recalculation.
    """
    value_columns = list(value_columns)
    header = pd.DataFrame(
        {
            CACHE_KEY_COLUMN: chem_keys.to_numpy(),
            "MoleculeID": (
                df[args.key_col].map(clean_text).to_numpy() if args.key_col in df.columns else ""
            ),
            "SMILES": (
                df[args.smiles_col].map(clean_smiles).to_numpy()
                if args.smiles_col in df.columns
                else ""
            ),
        },
        index=df.index,
    )
    current = pd.concat([header, features.reindex(columns=value_columns)], axis=1)
    keep = current[CACHE_KEY_COLUMN].ne("") & current[value_columns].notna().any(axis=1)
    current = current[keep]

    carried = 0
    dropped = 0
    if path.exists():
        try:
            existing = pd.read_csv(path)
            existing_keys = cache_row_keys(existing, legacy_key_map(df, args, chem_keys))
            if existing_keys is not None:
                existing = existing.copy()
                existing[CACHE_KEY_COLUMN] = existing_keys.to_numpy()
                dropped = int(existing[CACHE_KEY_COLUMN].eq("").sum())
                existing = existing[
                    existing[CACHE_KEY_COLUMN].ne("")
                    & ~existing[CACHE_KEY_COLUMN].isin(set(current[CACHE_KEY_COLUMN]))
                ]
                existing = existing.reindex(columns=current.columns)
                carried = len(existing)
                current = pd.concat([current, existing], axis=0, ignore_index=True)
        except Exception as exc:
            return {"cache_write_note": f"{path}: could not merge existing cache: {exc}"}

    path.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(path, index=False)
    return {
        "cache_path": str(path),
        "cache_rows_written": len(current),
        "cache_rows_carried_over": carried,
        "cache_rows_dropped_unkeyed": dropped,
    }


def resolve_external_features(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve OPERA and Abraham features, calculating only what the cache lacks."""
    work = df.reset_index(drop=True)
    n_rows = len(work)
    chem_keys = chemical_cache_keys(work, args.smiles_col)
    info: dict[str, Any] = {
        "cache_match": f"chemical identity ({CACHE_KEY_COLUMN})",
        "cache_key_unresolved_rows": int(chem_keys.eq("").sum()),
    }

    no_hits = pd.Series(False, index=work.index)

    if args.opera_mode == "skip":
        opera, opera_hit = blank_opera_frame(n_rows).set_axis(work.index), no_hits.copy()
        info["opera_source"] = "skipped"
    elif args.opera_mode == "preserve":
        opera = opera_from_existing(work).set_axis(work.index)
        opera_hit = ~critical_opera_missing_mask(opera)
        info["opera_source"] = "preserved_input_columns"
    elif args.opera_mode == "run":
        opera, opera_hit = blank_opera_frame(n_rows).set_axis(work.index), no_hits.copy()
        info["opera_source"] = "recalculated_all_rows"
    else:
        opera, opera_hit, cache_info = load_opera_cache(work, args, chem_keys)
        info.update(cache_info)
        info["opera_source"] = "cache" if args.opera_mode == "reuse" else "cache_plus_calculation"
        if args.opera_mode == "reuse" and not opera_hit.any():
            raise FileNotFoundError(
                "No reusable OPERA cache matched the input molecules. "
                + (cache_info["opera_cache_errors"] or "Checked: ")
                + "; ".join(str(p) for p in candidate_opera_csv_paths(args))
            )

    if args.abraham_mode == "skip":
        abraham, abraham_hit = blank_abraham_frame(n_rows).set_axis(work.index), no_hits.copy()
        info["abraham_source"] = "skipped"
    elif args.abraham_mode == "preserve":
        abraham = abraham_from_existing(work).set_axis(work.index)
        abraham_hit = abraham.notna().any(axis=1)
        info["abraham_source"] = "preserved_input_columns"
    elif args.abraham_mode == "run":
        abraham, abraham_hit = blank_abraham_frame(n_rows).set_axis(work.index), no_hits.copy()
        info["abraham_source"] = "recalculated_all_rows"
    else:
        abraham, abraham_hit, cache_info = load_abraham_cache(work, args, chem_keys)
        info.update(cache_info)
        info["abraham_source"] = (
            "cache" if args.abraham_mode == "reuse" else "cache_plus_calculation"
        )
        if args.abraham_mode == "reuse" and not abraham_hit.any():
            raise FileNotFoundError(
                "No reusable Abraham cache matched the input molecules. "
                + (cache_info["abraham_cache_errors"] or "Checked: ")
                + "; ".join(str(p) for p in candidate_abraham_csv_paths(args))
            )

    opera_pending = (
        ~opera_hit if args.opera_mode in {"auto", "run"} else no_hits.copy()
    )
    abraham_pending = (
        ~abraham_hit if args.abraham_mode in {"auto", "run"} else no_hits.copy()
    )
    run_index = work.index[opera_pending | abraham_pending]
    info["opera_rows_to_calculate"] = int(opera_pending.sum())
    info["abraham_rows_to_calculate"] = int(abraham_pending.sum())

    if len(run_index):
        forced = args.opera_mode == "run" or args.abraham_mode == "run"
        if not args.opera_exe.exists() and not forced:
            info["opera_note"] = (
                f"{len(run_index)} rows are not cached, but OPERA is unavailable at "
                f"{args.opera_exe}; those rows are left blank."
            )
            print(f"OPERA not found at {args.opera_exe}; leaving {len(run_index)} uncached rows blank.")
        else:
            print(
                f"Reusing cached predictions for {n_rows - len(run_index)}/{n_rows} molecules; "
                f"calculating {len(run_index)}."
            )
            run_opera_frame, run_abraham_frame, run_info = run_opera(work, args, run_index)
            info.update(run_info)
            if opera_pending.any():
                target = work.index[opera_pending]
                opera.loc[target, OPERA_COLUMNS] = run_opera_frame.loc[target, OPERA_COLUMNS]
            if abraham_pending.any() and args.abraham_mode in {"auto", "run"}:
                target = work.index[abraham_pending]
                abraham.loc[target, ABRAHAM_COLUMNS] = run_abraham_frame.loc[target, ABRAHAM_COLUMNS]
    else:
        print(f"All {n_rows} molecules resolved from cache; no OPERA run needed.")

    if args.opera_mode != "skip":
        opera_cache_path = args.work_dir / f"{args.output.stem}_opera.csv"
        cache_write = update_feature_cache(
            opera_cache_path, work, args, chem_keys, opera, OPERA_COLUMNS
        )
        info["opera_csv"] = str(opera_cache_path)
        info.update({f"opera_{k}": v for k, v in cache_write.items()})
    if args.abraham_mode != "skip":
        abraham_cache_path = args.work_dir / f"{args.output.stem}_abraham.csv"
        cache_write = update_feature_cache(
            abraham_cache_path, work, args, chem_keys, abraham, ABRAHAM_COLUMNS
        )
        info["abraham_csv"] = str(abraham_cache_path)
        info.update({f"abraham_{k}": v for k, v in cache_write.items()})

    return opera.reset_index(drop=True), abraham.reset_index(drop=True), info



def validate_screening_args(args: argparse.Namespace) -> None:
    for name in (
        "missing_threshold",
        "dominant_fraction_threshold",
        "correlation_threshold",
    ):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1; received {value}.")
    if args.min_pairwise_n < 3:
        raise ValueError("min_pairwise_n must be at least 3.")


def scalar_candidate_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        str(column)
        for column in df.columns
        if str(column) in EXACT_SCALAR_SCREENING_COLUMNS
        or str(column).startswith(SCALAR_SCREENING_PREFIXES)
    ]
    if not columns:
        raise ValueError("No scalar candidate columns were found after feature generation.")
    return columns


def scalar_screening_profile(
    df: pd.DataFrame,
    columns: list[str],
    missing_threshold: float,
    dominant_threshold: float,
) -> pd.DataFrame:
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    records: list[dict[str, Any]] = []
    n_rows = len(numeric)

    for feature in columns:
        series = numeric[feature]
        observed = series.dropna()
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / n_rows if n_rows else np.nan
        unique_count = int(observed.nunique())
        counts = observed.value_counts()
        dominant_value = counts.index[0] if not counts.empty else np.nan
        dominant_fraction = float(counts.iloc[0] / len(observed)) if len(observed) else np.nan

        missing_status = (
            ""
            if missing_count == 0
            else "high_missing"
            if missing_rate >= missing_threshold
            else "some_missing"
        )
        variation_status = (
            "constant"
            if unique_count <= 1
            else "near_constant"
            if dominant_fraction >= dominant_threshold
            else ""
        )
        quantiles = (
            observed.quantile([0.05, 0.50, 0.95])
            if len(observed)
            else pd.Series(dtype=float)
        )
        records.append(
            {
                "feature": feature,
                "missing_status": missing_status,
                "variation_status": variation_status,
                "nonmissing_count": int(len(observed)),
                "missing_count": missing_count,
                "missing_rate": float(missing_rate),
                "unique_count": unique_count,
                "dominant_value": dominant_value,
                "dominant_fraction": dominant_fraction,
                "min": float(observed.min()) if len(observed) else np.nan,
                "p05": float(quantiles.get(0.05, np.nan)),
                "median": float(quantiles.get(0.50, np.nan)),
                "p95": float(quantiles.get(0.95, np.nan)),
                "max": float(observed.max()) if len(observed) else np.nan,
                "standard_deviation": float(observed.std()) if len(observed) > 1 else 0.0,
                "iqr": (
                    float(observed.quantile(0.75) - observed.quantile(0.25))
                    if len(observed)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def scalar_missing_results(profile: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature",
        "missing_status",
        "nonmissing_count",
        "missing_count",
        "missing_rate",
    ]
    return (
        profile.loc[profile["missing_count"].gt(0), columns]
        .sort_values(["missing_rate", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def scalar_low_variation_results(profile: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature",
        "variation_status",
        "nonmissing_count",
        "missing_count",
        "missing_rate",
        "unique_count",
        "dominant_value",
        "dominant_fraction",
        "min",
        "p05",
        "median",
        "p95",
        "max",
        "standard_deviation",
        "iqr",
    ]
    result = profile.loc[profile["variation_status"].ne(""), columns].copy()
    result["_order"] = result["variation_status"].map({"constant": 0, "near_constant": 1})
    return (
        result.sort_values(
            ["_order", "dominant_fraction", "feature"],
            ascending=[True, False, True],
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def high_scalar_correlations(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    threshold: float,
    missing_threshold: float,
    min_n: int,
) -> pd.DataFrame:
    eligible = profile.loc[
        profile["variation_status"].eq("")
        & profile["missing_rate"].lt(missing_threshold)
        & profile["unique_count"].ge(2),
        "feature",
    ].tolist()
    numeric = df[eligible].apply(pd.to_numeric, errors="coerce")
    records: list[dict[str, Any]] = []

    for i, left in enumerate(eligible):
        for right in eligible[i + 1 :]:
            pair = numeric[[left, right]].dropna()
            if len(pair) < min_n or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            pearson = float(pair[left].corr(pair[right], method="pearson"))
            spearman = float(pair[left].corr(pair[right], method="spearman"))
            strongest = max(abs(pearson), abs(spearman))
            if strongest < threshold:
                continue
            method = "pearson" if abs(pearson) >= abs(spearman) else "spearman"
            records.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "pairwise_complete_n": int(len(pair)),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "max_absolute_correlation": strongest,
                    "strongest_method": method,
                    "strongest_correlation": pearson if method == "pearson" else spearman,
                }
            )

    columns = [
        "feature_1",
        "feature_2",
        "pairwise_complete_n",
        "pearson_r",
        "spearman_rho",
        "max_absolute_correlation",
        "strongest_method",
        "strongest_correlation",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(records, columns=columns)
        .sort_values("max_absolute_correlation", ascending=False)
        .reset_index(drop=True)
    )


def build_audit(
    args: argparse.Namespace,
    source: pd.DataFrame,
    enriched: pd.DataFrame,
    rdkit_problems: pd.DataFrame,
    classification_info: dict[str, Any],
    opera_info: dict[str, Any],
    opera_missing_report: pd.DataFrame,
    abraham_missing_report: pd.DataFrame,
    morgan_sparse: pd.DataFrame,
    atom_pair_sparse: pd.DataFrame,
    scalar_candidate_count: int,
    scalar_missing_values: pd.DataFrame,
    scalar_low_variation: pd.DataFrame,
    high_correlations: pd.DataFrame,
) -> pd.DataFrame:
    blank_smiles = int(source[args.smiles_col].map(clean_smiles).eq("").sum())
    duplicate_keys = 0
    if args.key_col in source.columns:
        keys = source[args.key_col].map(normalize_key)
        duplicate_keys = int(keys[keys != ""].duplicated(keep=False).sum())
    classification_cols_present = [col for col in CLASS_COLUMNS if col in enriched.columns]
    if all(col in enriched.columns for col in CLASS_COLUMNS):
        class_values = clean_text_frame(enriched[CLASS_COLUMNS])
        classification_complete_rows = int(class_values.ne("").all(axis=1).sum())
        classification_missing_rows = int(class_values.eq("").any(axis=1).sum())
    else:
        classification_complete_rows = 0
        classification_missing_rows = int(len(enriched))

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(enriched)),
        "key_col": args.key_col,
        "smiles_col": args.smiles_col,
        "blank_smiles": blank_smiles,
        "duplicate_normalized_key_rows": duplicate_keys,
        "rdkit_problem_rows": int(len(rdkit_problems)),
        "rdkit_descriptor_columns": int(sum(str(c).startswith("rdkit_") for c in enriched.columns)),
        "molecular_weight_column": "Mw (g/mol)",
        "molecular_weight_method": "RDKit Descriptors.MolWt; average molecular weight for mol-to-mass conversion",
        "rdkit_tpsa_method": "RDKit rdMolDescriptors.CalcTPSA(includeSandP=True); 2D fragment contributions including N, O, S, and P",
        "expanded_fingerprint_columns_on_Sheet1": 0,
        "raw_fingerprint_storage": "sparse long-format workbook sheets",
        "morgan_representation": "unfolded sparse count Morgan",
        "morgan_radius": int(args.morgan_radius),
        "morgan_sparse_rows": int(len(morgan_sparse)),
        "morgan_unique_feature_ids": int(morgan_sparse["feature_id"].nunique()) if not morgan_sparse.empty else 0,
        "atom_pair_representation": "unfolded sparse count topological atom-pair",
        "atom_pair_min_distance": int(args.atom_pair_min_distance),
        "atom_pair_max_distance": int(args.atom_pair_max_distance),
        "atom_pair_sparse_rows": int(len(atom_pair_sparse)),
        "atom_pair_unique_feature_ids": int(atom_pair_sparse["feature_id"].nunique()) if not atom_pair_sparse.empty else 0,
        **classification_info,
        "classification_columns_present": ", ".join(classification_cols_present),
        "classification_complete_rows": classification_complete_rows,
        "classification_missing_rows": classification_missing_rows,
        "opera_rows_with_missing_values": int(len(opera_missing_report)),
        "opera_missing_value_cells": int(opera_missing_report["missing_opera_count"].sum()) if not opera_missing_report.empty else 0,
        "opera_rows_with_critical_missing": int((opera_missing_report["missing_critical_count"] > 0).sum()) if not opera_missing_report.empty else 0,
        "opera_critical_missing_cells": int(opera_missing_report["missing_critical_count"].sum()) if not opera_missing_report.empty else 0,
        "opera_mode": args.opera_mode,
        "abraham_mode": args.abraham_mode,
        "abraham_descriptor_columns": int(sum(str(c).startswith("abraham_") for c in enriched.columns)),
        "abraham_rows_with_missing_values": int(len(abraham_missing_report)),
        "abraham_missing_value_cells": int(abraham_missing_report["missing_abraham_count"].sum()) if not abraham_missing_report.empty else 0,
        "abraham_descriptor_method": "PaDEL/CDK MLFER_E,S,A,BH,BO,L plus McGowan_Volume; group-contribution estimates from the supplied molecular representation",
        "scalar_candidate_count": int(scalar_candidate_count),
        "scalar_missing_feature_count": int(len(scalar_missing_values)),
        "scalar_low_variation_feature_count": int(len(scalar_low_variation)),
        "high_correlation_pair_count": int(len(high_correlations)),
        "scalar_missing_threshold": float(args.missing_threshold),
        "scalar_dominant_fraction_threshold": float(args.dominant_fraction_threshold),
        "scalar_correlation_threshold": float(args.correlation_threshold),
        "scalar_min_pairwise_n": int(args.min_pairwise_n),
        "scalar_candidate_rule": "actual OPERA predictions + abraham_* + Mw (g/mol) + rdkit_*",
        "scalar_screening_scope": "missingness, low variation, and pairwise collinearity only; no final inclusion/exclusion decision",
        **opera_info,
    }
    return pd.DataFrame([{"field": key, "value": value} for key, value in audit.items()])

def write_outputs(
    args: argparse.Namespace,
    enriched: pd.DataFrame,
    audit: pd.DataFrame,
    rdkit_problems: pd.DataFrame,
    classification_audit: pd.DataFrame,
    opera_missing_report: pd.DataFrame,
    abraham_missing_report: pd.DataFrame,
    morgan_sparse: pd.DataFrame,
    atom_pair_sparse: pd.DataFrame,
    fingerprint_parameters: pd.DataFrame,
    scalar_missing_values: pd.DataFrame,
    scalar_low_variation: pd.DataFrame,
    high_correlations: pd.DataFrame,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        enriched.to_excel(writer, sheet_name=args.sheet_name, index=False)
        morgan_sparse.to_excel(writer, sheet_name="morgan_count_sparse", index=False)
        atom_pair_sparse.to_excel(writer, sheet_name="atom_pair_count_sparse", index=False)
        fingerprint_parameters.to_excel(writer, sheet_name="fingerprint_parameters", index=False)
        audit.to_excel(writer, sheet_name="generation_audit", index=False)
        scalar_missing_values.to_excel(writer, sheet_name="scalar_missing_values", index=False)
        scalar_low_variation.to_excel(writer, sheet_name="scalar_constant_values", index=False)
        high_correlations.to_excel(writer, sheet_name="high_correlations", index=False)
        if not rdkit_problems.empty:
            rdkit_problems.to_excel(writer, sheet_name="rdkit_problems", index=False)
        if not classification_audit.empty:
            classification_audit.to_excel(writer, sheet_name="classification_audit", index=False)
        if not opera_missing_report.empty:
            opera_missing_report.to_excel(writer, sheet_name="opera_missing_values", index=False)
        if not abraham_missing_report.empty:
            abraham_missing_report.to_excel(writer, sheet_name="abraham_missing_values", index=False)

def main() -> None:
    args = parse_args()
    validate_screening_args(args)
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    source = pd.read_excel(args.input, sheet_name=args.sheet_name)
    source.columns = [c.strip() if isinstance(c, str) else c for c in source.columns]
    if args.smiles_col not in source.columns:
        raise ValueError(f"SMILES column {args.smiles_col!r} not found in {args.input}")
    if args.key_col not in source.columns:
        raise ValueError(f"Key column {args.key_col!r} not found in {args.input}")

    # pfas.xlsx is normalized in place, once, and consumed as-is here. Drift is
    # reported rather than corrected, so the maintained list stays authoritative.
    drift = report_unstandardized_smiles(source, args.smiles_col, args.key_col)
    if not drift.empty:
        print(f"WARNING: {len(drift)} SMILES in {args.input.name} are not in normalized form.")
        print("They are used as written. Normalize them in the workbook to silence this.")
        for _, row in drift.head(10).iterrows():
            print(f"  {row['pfas_key']}: {row['found'][:56]}  ->  {row['normalized'][:56]}")

    classification_features, classification_audit, classification_info = resolve_classification_features(source, args)
    opera_features, abraham_features, opera_info = resolve_external_features(source, args)
    manual = remove_generated_columns(source)
    rdkit_features, rdkit_problems, mols = build_rdkit_features(
        source,
        smiles_col=args.smiles_col,
        key_col=args.key_col,
    )
    enriched = pd.concat(
        [
            manual.reset_index(drop=True),
            classification_features.reset_index(drop=True),
            opera_features.reset_index(drop=True),
            abraham_features.reset_index(drop=True),
            rdkit_features.reset_index(drop=True),
        ],
        axis=1,
    )
    morgan_sparse, atom_pair_sparse, fingerprint_parameters = build_raw_fingerprint_outputs(
        source,
        mols,
        key_col=args.key_col,
        morgan_radius=args.morgan_radius,
        atom_pair_min_distance=args.atom_pair_min_distance,
        atom_pair_max_distance=args.atom_pair_max_distance,
    )
    opera_missing_report = build_opera_missing_report(args, source, opera_features)
    abraham_missing_report = build_abraham_missing_report(args, source, abraham_features)

    scalar_candidates = scalar_candidate_columns(enriched)
    scalar_profile = scalar_screening_profile(
        enriched,
        scalar_candidates,
        missing_threshold=args.missing_threshold,
        dominant_threshold=args.dominant_fraction_threshold,
    )
    scalar_missing_values = scalar_missing_results(scalar_profile)
    scalar_low_variation = scalar_low_variation_results(scalar_profile)
    high_correlations = high_scalar_correlations(
        enriched,
        scalar_profile,
        threshold=args.correlation_threshold,
        missing_threshold=args.missing_threshold,
        min_n=args.min_pairwise_n,
    )

    audit = build_audit(
        args,
        source,
        enriched,
        rdkit_problems,
        classification_info,
        opera_info,
        opera_missing_report,
        abraham_missing_report,
        morgan_sparse,
        atom_pair_sparse,
        len(scalar_candidates),
        scalar_missing_values,
        scalar_low_variation,
        high_correlations,
    )
    write_outputs(
        args,
        enriched,
        audit,
        rdkit_problems,
        classification_audit,
        opera_missing_report,
        abraham_missing_report,
        morgan_sparse,
        atom_pair_sparse,
        fingerprint_parameters,
        scalar_missing_values,
        scalar_low_variation,
        high_correlations,
    )

    print(f"Rows processed: {len(enriched)}")
    print(f"OPERA source: {opera_info.get('opera_source')}")
    print(f"Abraham source: {opera_info.get('abraham_source')}")
    if opera_info.get("opera_note"):
        print(f"OPERA note: {opera_info['opera_note']}")
    print(f"RDKit parse problems: {len(rdkit_problems)}")
    print(f"RDKit descriptor columns: {sum(str(c).startswith('rdkit_') for c in enriched.columns)}")
    print(f"Abraham descriptor columns: {sum(str(c).startswith('abraham_') for c in enriched.columns)}")
    print(f"Scalar candidates screened: {len(scalar_candidates)}")
    print(f"Scalar features with missing values: {len(scalar_missing_values)}")
    print(f"Constant or near-constant scalar features: {len(scalar_low_variation)}")
    print(f"High-correlation scalar pairs: {len(high_correlations)}")
    print("Expanded fingerprint columns on Sheet1: 0")
    print(
        f"Morgan raw fingerprint: {len(morgan_sparse)} sparse rows, "
        f"{morgan_sparse['feature_id'].nunique() if not morgan_sparse.empty else 0} unique feature IDs, "
        f"radius {args.morgan_radius}"
    )
    print(
        f"Atom-pair raw fingerprint: {len(atom_pair_sparse)} sparse rows, "
        f"{atom_pair_sparse['feature_id'].nunique() if not atom_pair_sparse.empty else 0} unique feature IDs, "
        f"distance {args.atom_pair_min_distance}-{args.atom_pair_max_distance}"
    )
    if not abraham_missing_report.empty:
        print(
            "Abraham missing-value audit: "
            f"{len(abraham_missing_report)} rows have at least one missing descriptor."
        )
    print(f"Classification source: {classification_info.get('classification_source')}")
    if classification_info.get("classification_note"):
        print(f"Classification note: {classification_info['classification_note']}")
    if not opera_missing_report.empty:
        critical_rows = int((opera_missing_report["missing_critical_count"] > 0).sum())
        print(
            "OPERA missing-value audit: "
            f"{len(opera_missing_report)} rows have at least one missing OPERA field; "
            f"{critical_rows} rows have missing critical prediction fields."
        )
        if critical_rows:
            examples = opera_missing_report.loc[
                opera_missing_report["missing_critical_count"] > 0,
                ["excel_row", "pfas_key", "missing_critical_columns"],
            ].head(10)
            print("First critical OPERA missing rows:")
            for _, row in examples.iterrows():
                print(
                    f"  Excel row {row['excel_row']}: {row['pfas_key']} "
                    f"missing {row['missing_critical_columns']}"
                )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
