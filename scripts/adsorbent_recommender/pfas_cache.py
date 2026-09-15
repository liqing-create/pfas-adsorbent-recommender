"""Read-only cached PFAS feature lookup for recommender requests.

No OPERA calculation is performed here. Uncached structures are returned as
offline-generation requests so the audited batch feature generator remains the
only producer of PFAS properties.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ML_model_training.backend import logkd_config as cfg


GENERATOR_PATH = cfg.BOX_NOTES_ROOT / "PFAS" / "generate_pfas_features.py"
DEFAULT_RDKIT_PYTHON = Path.home() / "anaconda3" / "envs" / "pfas-atlas" / "python.exe"
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null"}
OPERA_CRITICAL_COLUMNS = (
    "LogP_pred", "LogP_predRange", "LogWS_pred", "WS_predRange", "LogKOA_pred", "KOA_predRange",
    "LogD55_pred", "LogD55_predRange", "LogD74_pred", "LogD74_predRange",
)
IDENTIFIER_COLUMNS: dict[str, tuple[str, ...]] = {
    "name": ("PFAS_name", "Compound Full Name", "Name", "Abbreviation"),
    "cas": ("CAS Number", "CAS", "CAS#"),
    "abbreviation": ("Abbreviation",),
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _normalized_identifier(value: Any) -> str:
    return "".join(character for character in _clean(value).casefold() if character.isalnum())


@lru_cache(maxsize=1)
def generator_module():
    """Load the existing feature generator without running its CLI entry point."""
    if not GENERATOR_PATH.exists():
        raise FileNotFoundError(f"PFAS feature generator not found: {GENERATOR_PATH}")
    spec = importlib.util.spec_from_file_location("adsorbent_recommender_pfas_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PFAS feature generator from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonicalize_smiles(smiles: str, rdkit_python: Path | None = None) -> str:
    """Use the generator's RDKit logic in-process, or the PFAS-atlas env as a tiny helper."""
    try:
        generator = generator_module()
        molecule = generator.mol_from_smiles(generator.clean_smiles(smiles))
        if molecule is None:
            raise ValueError("SMILES could not be parsed by RDKit.")
        return str(generator.descriptor_row(molecule)["Canonical SMILES"])
    except (ModuleNotFoundError, FileNotFoundError):
        # A hosted deployment ships RDKit but not the Box-side generator script.
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            pass
        else:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("SMILES could not be parsed by RDKit.")
            return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        helper = Path(rdkit_python or DEFAULT_RDKIT_PYTHON)
        if not helper.exists():
            raise RuntimeError(
                "RDKit is unavailable in the recommender environment and no PFAS-atlas Python helper was found. "
                "Set --rdkit-python to an environment with RDKit."
            )
        code = (
            "from rdkit import Chem; import sys; "
            "m=Chem.MolFromSmiles(sys.argv[1]); "
            "print(Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m else '')"
        )
        completed = subprocess.run([str(helper), "-c", code, smiles], check=True, capture_output=True, text=True)
        canonical = completed.stdout.strip()
        if not canonical:
            raise ValueError("SMILES could not be parsed by RDKit.")
        return canonical


@dataclass(frozen=True)
class PfasLookup:
    status: str
    submitted_smiles: str
    canonical_smiles: str
    feature_row: dict[str, Any] | None
    cache_key: str | None
    warnings: tuple[str, ...]
    offline_request: dict[str, Any] | None


class PfasFeatureStore:
    def __init__(
        self,
        path: Path = cfg.DEFAULT_PFAS_FEATURES,
        sheet_name: str = cfg.DEFAULT_PFAS_FEATURES_SHEET,
        rdkit_python: Path | None = None,
    ):
        self.path, self.sheet_name = Path(path), sheet_name
        if not self.path.exists():
            raise FileNotFoundError(f"PFAS feature cache not found: {self.path}")
        self.features = pd.read_excel(self.path, sheet_name=self.sheet_name).copy()
        self.features["__canonical_smiles"] = self.features.get("Canonical SMILES", pd.Series("", index=self.features.index)).map(_clean)
        self.features["__abbreviation"] = self.features.get("Abbreviation", pd.Series("", index=self.features.index)).map(_clean)
        for identity_type, columns in IDENTIFIER_COLUMNS.items():
            available = next((column for column in columns if column in self.features.columns), None)
            self.features[f"__{identity_type}"] = (
                self.features[available].map(_normalized_identifier)
                if available is not None
                else pd.Series("", index=self.features.index)
            )
        self.rdkit_python = Path(rdkit_python) if rdkit_python else DEFAULT_RDKIT_PYTHON
        self.critical_columns = OPERA_CRITICAL_COLUMNS
        self.cache_version = self._cache_version()

    def _cache_version(self) -> dict[str, Any]:
        created_at = ""
        try:
            audit = pd.read_excel(self.path, sheet_name="generation_audit")
            match = audit.loc[audit["field"].astype(str).eq("created_at"), "value"]
            created_at = _clean(match.iloc[0]) if not match.empty else ""
        except ValueError:
            pass
        return {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(), "created_at": created_at, "rows": int(len(self.features))}

    def _result(self, submitted: str, canonical: str, candidates: pd.DataFrame, unknown_status: str, offline_request: dict[str, Any] | None = None) -> PfasLookup:
        if candidates.empty:
            return PfasLookup(unknown_status, submitted, canonical, None, None, (), offline_request)
        row, warnings = candidates.iloc[0].to_dict(), []
        if len(candidates) > 1:
            warnings.append("Multiple cached PFAS identities matched exactly; the first cache row was used.")
        missing = [column for column in self.critical_columns if column in row and _clean(row[column]).lower() in MISSING_TOKENS]
        cache_key = _clean(row.get("Abbreviation") or row.get("PFAS_name") or row.get("Compound Full Name"))
        if missing:
            return PfasLookup(
                "cache_row_requires_review", submitted, canonical, row, cache_key,
                tuple(warnings + [f"Critical OPERA fields missing: {', '.join(missing)}"]), None,
            )
        return PfasLookup("cached", submitted, canonical, row, cache_key, tuple(warnings), None)

    def lookup_identifier(self, identity: str, identity_type: str) -> PfasLookup:
        """Resolve an exact cached name, CAS number, or abbreviation without live generation."""
        identity_type = _clean(identity_type).casefold()
        if identity_type == "smiles":
            return self.lookup(identity)
        if identity_type not in IDENTIFIER_COLUMNS:
            raise ValueError("identity_type must be one of: smiles, name, cas, abbreviation.")
        submitted = _clean(identity)
        key = _normalized_identifier(submitted)
        if not key:
            return PfasLookup("identifier_not_found", submitted, "", None, None, (), None)
        candidates = self.features[self.features[f"__{identity_type}"].eq(key)]
        canonical = _clean(candidates.iloc[0]["__canonical_smiles"]) if not candidates.empty else ""
        return self._result(submitted, canonical, candidates, "identifier_not_found")

    def lookup(self, smiles: str, abbreviation: str | None = None) -> PfasLookup:
        submitted = _clean(smiles)
        # Avoid spawning an RDKit helper for the common cache-hit path. Exact
        # source or canonical strings are already unambiguous cache keys.
        direct = self.features[
            self.features.get("SMILES", pd.Series("", index=self.features.index)).map(_clean).eq(submitted)
            | self.features["__canonical_smiles"].eq(submitted)
        ]
        if direct.empty:
            canonical = canonicalize_smiles(smiles, self.rdkit_python)
            candidates = self.features[self.features["__canonical_smiles"].eq(canonical)]
        else:
            candidates = direct
            canonical = _clean(candidates.iloc[0]["__canonical_smiles"])
        if abbreviation:
            exact = candidates[candidates["__abbreviation"].str.casefold().eq(_clean(abbreviation).casefold())]
            if not exact.empty:
                candidates = exact
        request = {
            "status": "queued_for_offline_feature_generation", "submitted_smiles": submitted,
            "canonical_smiles": canonical, "requested_abbreviation": _clean(abbreviation),
            "generator": str(GENERATOR_PATH),
            "required_review": "Run the existing audited generator; do not invoke OPERA during recommendation.",
        }
        return self._result(submitted, canonical, candidates, "uncached", request)


def append_offline_request(path: Path, request: dict[str, Any]) -> None:
    """Append a deduplicated JSONL request for the existing batch generator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = _clean(request.get("canonical_smiles"))
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(_clean(json.loads(line).get("canonical_smiles")))
            except json.JSONDecodeError:
                continue
    if canonical not in existing:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
