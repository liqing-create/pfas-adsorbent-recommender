"""Versioned RDKit descriptors shared by batch generation and live screening."""
from __future__ import annotations
from typing import Any
from functools import lru_cache
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

DESCRIPTOR_VERSION = "pfas-rdkit-v1"

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
    include_path_descriptors: bool = True,
) -> dict[str, float]:
    fluorinated_carbons = {r["atom_index"] for r in records if r["is_fluorinated_C"]}

    longest_chain = _longest_simple_path(mol, fluorinated_carbons) if include_path_descriptors else np.nan
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


def descriptor_row(mol: Chem.Mol, *, include_path_descriptors: bool = True) -> dict[str, float | str]:
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
    row.update(fluorinated_carbon_summary(mol, records, include_path_descriptors))
    row.update(pfas_headgroup_summary(mol))
    row.update(charge_summary(mol))
    return row




@lru_cache(maxsize=512)
def prepare_structure(smiles: str) -> tuple[str, str]:
    """Return canonical input and the curated acid/base convention, without stripping salts."""
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 4096:
        raise ValueError("Enter a SMILES for one defined PFAS molecule (at most 4096 characters).")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("SMILES could not be parsed as a defined molecule; check it and try again.")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Enter a single PFAS parent structure, without counterions or disconnected mixture components.")
    if mol.GetNumHeavyAtoms() > 256:
        raise ValueError("This structure is too large for small-molecule PFAS screening (maximum 256 heavy atoms).")
    if not any({bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == {"C", "F"} for bond in mol.GetBonds()):
        raise ValueError("The structure has no carbon–fluorine bond. Enter the intended PFAS structure.")
    original = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        mol = rdMolStandardize.Normalizer().normalize(mol)
        mol = rdMolStandardize.Reionizer().reionize(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError("The structure could not be put into the curated protonation convention. Check the parent SMILES.") from exc
    return original, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=512)
def _generated(canonical: str, version: str, rdkit_version: str) -> tuple[tuple[str, Any], ...]:
    # The deployed inputs do not include longest-path descriptors. Avoid an
    # exponential graph search on arbitrary submitted cyclic structures.
    return tuple(descriptor_row(Chem.MolFromSmiles(canonical), include_path_descriptors=False).items())


def generate_features(canonical: str) -> dict[str, Any]:
    return dict(_generated(canonical, DESCRIPTOR_VERSION, rdBase.rdkitVersion))
