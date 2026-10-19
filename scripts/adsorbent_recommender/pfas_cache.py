"""Curated PFAS lookup with a separate, versioned RDKit generation fallback."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import pandas as pd
from rdkit import Chem, rdBase
from ML_model_training.backend import logkd_config as cfg
from adsorbent_recommender.pfas_descriptors import DESCRIPTOR_VERSION, generate_features, prepare_structure

# Compatibility exports; OPERA is never run during recommendation.
GENERATOR_PATH = cfg.BOX_NOTES_ROOT / "PFAS" / "generate_pfas_features.py"
OPERA_CRITICAL_COLUMNS = (
    "LogP_pred", "LogP_predRange", "LogWS_pred", "WS_predRange", "LogKOA_pred", "KOA_predRange",
    "LogD55_pred", "LogD55_predRange", "LogD74_pred", "LogD74_predRange",
)
IDENTIFIER_COLUMNS = {
    "name": ("PFAS_name", "Compound Full Name", "Name", "Abbreviation"),
    "cas": ("CAS Number", "CAS", "CAS#"), "abbreviation": ("Abbreviation",),
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null"}

def _clean(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()

def _normalized_identifier(value: Any) -> str:
    return "".join(c for c in _clean(value).casefold() if c.isalnum())

def canonicalize_smiles(smiles: str, rdkit_python: Path | None = None) -> str:
    """Canonicalize without changing protonation; retain the helper argument for compatibility."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES could not be parsed by RDKit.")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

@dataclass(frozen=True)
class PfasLookup:
    status: str
    submitted_smiles: str
    canonical_smiles: str
    feature_row: dict[str, Any] | None
    cache_key: str | None
    warnings: tuple[str, ...] = ()
    offline_request: dict[str, Any] | None = None
    feature_source: str | None = None
    descriptor_version: str | None = None
    rdkit_version: str | None = None
    identity_resolution: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status in {"cached", "generated"}

class PfasFeatureStore:
    def __init__(self, path: Path = cfg.DEFAULT_PFAS_FEATURES,
                 sheet_name: str = cfg.DEFAULT_PFAS_FEATURES_SHEET,
                 rdkit_python: Path | None = None, *, required_features: tuple[str, ...] = (),
                 numeric_features: tuple[str, ...] | None = None, allow_generation: bool = False):
        self.path, self.sheet_name = Path(path), sheet_name
        self.required_features = required_features
        self.numeric_features = set(required_features if numeric_features is None else numeric_features)
        self.allow_generation = allow_generation
        self.features = pd.read_excel(self.path, sheet_name=sheet_name).copy()
        self.features["__canonical_smiles"] = self.features.get("Canonical SMILES", pd.Series("", index=self.features.index)).map(_clean)
        self.features["__abbreviation"] = self.features.get("Abbreviation", pd.Series("", index=self.features.index)).map(_clean)
        self._structure_index: dict[str, list[int]] | None = None
        created_at = ""
        try:
            audit = pd.read_excel(self.path, sheet_name="generation_audit")
            dates = audit.loc[audit["field"].astype(str).eq("created_at"), "value"]
            created_at = _clean(dates.iloc[0]) if not dates.empty else ""
        except (ValueError, KeyError):
            pass
        self.cache_version = {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                              "created_at": created_at, "rows": len(self.features)}

    def _validated(self, lookup: PfasLookup) -> PfasLookup:
        missing = []
        for name in self.required_features:
            value = (lookup.feature_row or {}).get(name)
            if _clean(value).casefold() in MISSING_TOKENS:
                missing.append(name)
            elif name in self.numeric_features:
                try:
                    if not math.isfinite(float(value)):
                        missing.append(name)
                except (ValueError, TypeError):
                    missing.append(name)
        if missing:
            return replace(lookup, status="missing_required_features", warnings=lookup.warnings + (
                "Required PFAS model inputs are missing or nonfinite: " + ", ".join(missing) + ".",
            ))
        return lookup

    def _cached(self, submitted: str, candidates: pd.DataFrame) -> PfasLookup:
        canonical_values = set(candidates["__canonical_smiles"]) - {""}
        if len(canonical_values) > 1:
            return PfasLookup("ambiguous_identity", submitted, "", None, None, (
                "This identifier matches multiple structures. Enter the intended SMILES.",
            ))
        row = candidates.iloc[0].to_dict()
        label = next((_clean(row.get(k)) for k in ("Abbreviation", "PFAS_name", "Compound Full Name") if _clean(row.get(k))), None)
        return self._validated(PfasLookup("cached", submitted, row["__canonical_smiles"], row, label,
            feature_source="reviewed_cache", identity_resolution={"source": "reviewed_cache"}))

    def lookup_identifier(self, identity: str, identity_type: str) -> PfasLookup:
        if identity_type == "smiles":
            return self.lookup(identity)
        if identity_type not in IDENTIFIER_COLUMNS:
            raise ValueError("identity_type must be one of: smiles, name, cas, abbreviation.")
        key = _normalized_identifier(identity)
        mask = pd.Series(False, index=self.features.index)
        for col in IDENTIFIER_COLUMNS[identity_type]:
            if col in self.features:
                mask |= self.features[col].map(_normalized_identifier).eq(key) & bool(key)
        candidates = self.features[mask]
        if not candidates.empty:
            return self._cached(identity, candidates)
        return PfasLookup("identifier_not_found", identity, "", None, None, (
            "No reviewed identity matched. Resolve the name/CAS and confirm its structure, or enter SMILES.",
        ))

    def _normalized_matches(self, original: str, canonical: str) -> pd.DataFrame:
        if self._structure_index is None:
            self._structure_index = {}
            for index, row in self.features.iterrows():
                value = _clean(row.get("Canonical SMILES")) or _clean(row.get("SMILES"))
                try:
                    raw, normalized = prepare_structure(value)
                except ValueError:
                    continue
                for key in {raw, normalized}:
                    self._structure_index.setdefault(key, []).append(index)
        indices = sorted(set(self._structure_index.get(original, []) + self._structure_index.get(canonical, [])))
        return self.features.loc[indices]

    def lookup(self, smiles: str, abbreviation: str | None = None) -> PfasLookup:
        submitted = _clean(smiles)
        direct = self.features[
            self.features.get("SMILES", pd.Series("", index=self.features.index)).map(_clean).eq(submitted)
            | self.features["__canonical_smiles"].eq(submitted)
        ]
        if not direct.empty:
            return self._cached(submitted, direct)
        try:
            original, canonical = prepare_structure(submitted)
        except ValueError as exc:
            return PfasLookup("invalid_structure", submitted, "", None, None, (str(exc),))
        candidates = self._normalized_matches(original, canonical)
        if not candidates.empty:
            exact = candidates[candidates["__canonical_smiles"].eq(original)]
            return self._cached(submitted, exact if not exact.empty else candidates)
        if not self.allow_generation:
            return PfasLookup("uncached", submitted, canonical, None, None, (
                "This bundle supports reviewed PFAS only. Repackage it to enable live structure features.",
            ))
        row = generate_features(canonical)
        warnings = ("Structure was normalized to the curated protonation convention; the displayed SMILES was used.",) if canonical != original else ()
        return self._validated(PfasLookup("generated", submitted, canonical, row, None, warnings,
            feature_source="computed_at_runtime", descriptor_version=DESCRIPTOR_VERSION,
            rdkit_version=rdBase.rdkitVersion,
            identity_resolution={"source": "submitted_smiles", "original_canonical_smiles": original}))

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
