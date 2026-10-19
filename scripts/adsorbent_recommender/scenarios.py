"""Portable scenario snapshots and comparisons; uploaded files never execute code."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adsorbent_recommender.mixture_inputs import from_mg_l, normalize_unit, to_mg_l
from adsorbent_recommender.pfas_descriptors import DESCRIPTOR_VERSION, prepare_structure
from adsorbent_recommender.service import RESULT_SCHEMA_VERSION
from adsorbent_recommender.additive_inputs import AdditiveInputError, validate_additives
from adsorbent_recommender.input_validation import (
    InputValidationError, NONNEGATIVE_WATER_FIELDS, finite_number, validate_scenario_numbers,
)

SCENARIO_FORMAT = "adsorbent_recommender_scenario"
SCENARIO_VERSION = 3
MAX_SCENARIO_BYTES = 20_000_000
MAX_SAVED_SCENARIOS = 12
MAX_LIBRARY_BYTES = 60_000_000
SCENARIO_EXAMPLE_PATH = Path(__file__).parent / "examples" / "scenario_inputs_example.csv"
WATER_NUMBERS = ("pH", "organic_carbon_mg_l", "tds_mg_l", "ionic_strength_mol_l")
ION_KEYS = ("Na", "K", "Ca", "Mg", "Cl", "HCO3", "SO4", "phosphate")


class ScenarioError(ValueError):
    pass


def _portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _portable(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _text(value: Any, field: str, *, required: bool = False, limit: int = 10_000) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ScenarioError(f"{field} must be {'nonempty ' if required else ''}text of at most {limit} characters.")
    return value.strip()


def _number(value: Any, field: str, *, optional: bool = False, nonnegative: bool = False) -> float | None:
    try:
        return finite_number(value, field, optional=optional, nonnegative=nonnegative)
    except InputValidationError as exc:
        raise ScenarioError(str(exc)) from exc


def validate_inputs(inputs: Any) -> dict:
    if not isinstance(inputs, dict) or not isinstance(inputs.get("pfas"), list) or not 1 <= len(inputs["pfas"]) <= 100:
        raise ScenarioError("A scenario needs 1–100 PFAS entries.")
    rows = []
    for index, raw in enumerate(inputs["pfas"], 1):
        if not isinstance(raw, dict):
            raise ScenarioError(f"PFAS row {index} must be an object.")
        identity = _text(raw.get("identity"), f"PFAS row {index} identity", required=True)
        kind = _text(raw.get("identity_type"), "identity_type", required=True).casefold()
        if kind not in {"name", "cas", "abbreviation", "smiles"}:
            raise ScenarioError(f"PFAS row {index}: identity_type must be name, abbreviation, cas, or smiles.")
        mg_l = _number(raw.get("initial_concentration_mg_l"), f"PFAS row {index} concentration")
        if mg_l <= 0:
            raise ScenarioError(f"PFAS row {index} needs a concentration greater than zero.")
        unit = normalize_unit(raw.get("concentration_unit", "mg/L"))
        amount = _number(raw.get("concentration_value", from_mg_l(mg_l, unit)), "Submitted concentration")
        if not math.isclose(to_mg_l(amount, unit), mg_l, rel_tol=1e-12, abs_tol=0):
            raise ScenarioError(f"PFAS row {index}: submitted concentration and mg/L value disagree.")
        row = {"identity": identity, "identity_type": kind, "initial_concentration_mg_l": mg_l,
               "concentration_value": amount, "concentration_unit": unit}
        for key in ("label", "confirmed_smiles", "canonical_smiles", "structure_key", "abbreviation", "full_name", "cas"):
            if raw.get(key) is not None:
                row[key] = _text(raw[key], key)
        if "structure_confirmed" in raw:
            if not isinstance(raw["structure_confirmed"], bool):
                raise ScenarioError("structure_confirmed must be true or false.")
            row["structure_confirmed"] = raw["structure_confirmed"]
        if kind == "smiles":
            prepare_structure(identity)
        if row.get("canonical_smiles"):
            prepare_structure(row["canonical_smiles"])
        if row.get("confirmed_smiles"):
            prepare_structure(row["confirmed_smiles"])
            if raw.get("structure_confirmed") is not True:
                raise ScenarioError(f"PFAS row {index}: the supplied structure has not been confirmed.")
            row["structure_confirmed"] = True
        structures = [row[key] for key in ("canonical_smiles", "confirmed_smiles") if row.get(key)]
        if kind == "smiles":
            structures.append(identity)
        if len({prepare_structure(smiles)[1] for smiles in structures}) > 1:
            raise ScenarioError(f"PFAS row {index} contains conflicting structures.")
        for key in ("resolution", "import_source"):
            if key in raw:
                if not isinstance(raw[key], dict):
                    raise ScenarioError(f"{key} must be an object.")
                row[key] = _portable(raw[key])
        rows.append(row)
    raw_water = inputs.get("water", {})
    if not isinstance(raw_water, dict):
        raise ScenarioError("water must be an object.")
    water = {key: _number(raw_water.get(key), key, optional=True,
                        nonnegative=key in NONNEGATIVE_WATER_FIELDS) for key in WATER_NUMBERS}
    water["water_type"] = _text(raw_water.get("water_type") or "", "water_type")
    organic_present = raw_water.get("organic_matter_present", False)
    if not isinstance(organic_present, bool):
        raise ScenarioError("organic_matter_present must be true or false.")
    if any(raw_water.get(key) for key in ("organic_matter", "added_organic_matter", "organic_additives")):
        raise ScenarioError("Use organic_matter_present true/false instead of organic composition text.")
    water["organic_matter_present"] = organic_present
    if raw_water.get("inorganic_matter") or raw_water.get("added_inorganic_matter"):
        raise ScenarioError("Inorganic additives now use species, concentration, and unit fields. Use the current example CSV.")
    try:
        water["inorganic_additives"] = validate_additives(raw_water.get("inorganic_additives", []))
    except AdditiveInputError as exc:
        raise ScenarioError(str(exc)) from exc
    ions = raw_water.get("ions", {})
    if not isinstance(ions, dict) or any(key not in ION_KEYS or not isinstance(value, bool) for key, value in ions.items()):
        raise ScenarioError("Water ions must use supported ion names with true/false values.")
    water["ions"] = {key: ions.get(key, False) for key in ION_KEYS}
    exploratory = inputs.get("include_exploratory_adsorbents", False)
    if not isinstance(exploratory, bool):
        raise ScenarioError("include_exploratory_adsorbents must be true or false.")
    return {"pfas": rows, "water": water, "include_exploratory_adsorbents": exploratory,
            "temperature_c": _number(inputs.get("temperature_c"), "Temperature", optional=True)}


def scenario_requests(inputs: dict) -> list[dict]:
    """Mirror the service request, excluding editor-only identity metadata."""
    common = {"water": {key: value for key, value in inputs["water"].items() if value not in (None, "") or key == "ions"},
              "temperature_c": inputs["temperature_c"], "adsorbent": {"categories": [], "include_exploratory": inputs.get("include_exploratory_adsorbents", False)}}
    return [{**common, "pfas": {key: row[key] for key in
             ("identity", "identity_type", "label", "confirmed_smiles", "structure_confirmed", "resolution") if key in row},
             "initial_concentration_mg_l": row["initial_concentration_mg_l"],
             "concentration_input": {"value": row["concentration_value"], "unit": row["concentration_unit"]}}
            for row in inputs["pfas"]]


def _object(value: Any, field: str, *, nullable: bool = False) -> dict:
    if value is None and nullable:
        return {}
    if not isinstance(value, dict):
        raise ScenarioError(f"{field} must be an object{' or null' if nullable else ''}.")
    return value


def _text_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ScenarioError(f"{field} must be a list of text values.")


def _caveat_list(value: Any, field: str, *, required_text: bool = False) -> None:
    if not isinstance(value, list):
        raise ScenarioError(f"{field} must be a list of objects.")
    for index, item in enumerate(value):
        path = f"{field}[{index}]"
        _object(item, path)
        if (required_text or "text" in item) and not isinstance(item.get("text"), str):
            raise ScenarioError(f"{path}.text must be text.")
        if item.get("pfas") is not None:
            _text_list(item["pfas"], f"{path}.pfas")


def _lookup_structure(value: Any, field: str) -> None:
    lookup = _object(value, field)
    if "support" in lookup:
        _object(lookup["support"], f"{field}.support")
    for name in ("canonical_smiles", "cache_key", "label"):
        if lookup.get(name) is not None and not isinstance(lookup[name], str):
            raise ScenarioError(f"{field}.{name} must be text or null.")


def _score_structure(item: dict, path: str) -> None:
    """Validate nested fields read by result export and saved-run comparison."""
    _text_list(item.get("warnings", []), f"{path}.warnings")
    _caveat_list(item.get("caveats", []), f"{path}.caveats")
    numeric_fields = {
        "performance": ("mean_logKd", "ensemble_sd", "Kd_L_g", "dose_mg_L", "match_count", "study_count"),
        "prediction": ("mean_logKd", "ensemble_sd"),
        "mixture_performance": ("worst_case_logKd", "mean_logKd", "pfas_count", "database_supported_pfas"),
        "adsorbent_support": ("training_rows", "source_records", "independent_studies"),
        "concentration_input": ("value",),
    }
    for field, numbers in numeric_fields.items():
        if field not in item:
            continue
        mapping = _object(item[field], f"{path}.{field}", nullable=field != "mixture_performance")
        for number in numbers:
            if number in mapping:
                _number(mapping[number], f"{path}.{field}.{number}", optional=True)
    for field in ("pfas_support", "database_evidence"):
        if field in item:
            mapping = _object(item[field], f"{path}.{field}", nullable=True)
            if field == "database_evidence":
                for child in ("performance", "related_evidence"):
                    if child in mapping:
                        _object(mapping[child], f"{path}.{field}.{child}", nullable=child == "related_evidence")
                if "performance" in mapping:
                    _score_structure({"performance": mapping["performance"]}, f"{path}.database_evidence")
    if item.get("pfas_lookup") is not None:
        _lookup_structure(item["pfas_lookup"], f"{path}.pfas_lookup")
    if "initial_concentration_mg_l" in item:
        value = _number(item["initial_concentration_mg_l"], f"{path}.initial_concentration_mg_l")
        if value <= 0:
            raise ScenarioError(f"{path}.initial_concentration_mg_l must be greater than zero.")


def validate_result_structure(result: Any) -> None:
    """Check current and compatible historical results without changing them."""
    if not isinstance(result, dict) or result.get("status") != "recommendations_ready":
        raise ScenarioError("Saved results must be a completed screening run.")
    requests = result.get("scenarios")
    if not isinstance(requests, list) or not 1 <= len(requests) <= 100:
        raise ScenarioError("Saved results need 1–100 PFAS scenarios.")
    for index, request in enumerate(requests):
        path = f"result.scenarios[{index}]"
        _object(request, path)
        _object(request.get("pfas"), f"{path}.pfas")
        _object(request.get("adsorbent", {}), f"{path}.adsorbent")
        try:
            validate_scenario_numbers(request)
        except InputValidationError as exc:
            raise ScenarioError(f"{path}: {exc}") from exc
        _score_structure({"concentration_input": request.get("concentration_input")}, path)
    for field in ("bundle", "water_chemistry", "water_support", "screening_conditions", "mixture"):
        if field in result:
            _object(result[field], f"result.{field}", nullable=field == "bundle")
    conditions = result.get("screening_conditions", {})
    if "fixed_conditions" in conditions:
        fixed = _object(conditions["fixed_conditions"], "result.screening_conditions.fixed_conditions")
        for field, value in fixed.items():
            if field.startswith("Temperature_("):
                _number(value, f"result.screening_conditions.fixed_conditions.{field}")
    candidates = result.get("recommendations")
    if not isinstance(candidates, list) or not candidates or len(candidates) > 10_000:
        raise ScenarioError("Saved results need a bounded, nonempty candidate list.")
    keys, ranks = set(), set()
    for index, item in enumerate(candidates):
        path = f"result.recommendations[{index}]"
        _object(item, path)
        product = _object(item.get("product"), f"{path}.product")
        key = _text(product.get("Product_key"), f"{path}.product.Product_key", required=True)
        if key in keys:
            raise ScenarioError("Saved results contain duplicate product keys.")
        keys.add(key)
        _number(item.get("ranking_score"), f"{path}.ranking_score")
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 or rank in ranks:
            raise ScenarioError(f"{path}.rank must be a unique positive integer.")
        ranks.add(rank)
        _score_structure(item, path)
        pairs = item.get("pfas_results")
        if pairs is not None and not isinstance(pairs, list):
            raise ScenarioError(f"{path}.pfas_results must be a list of objects.")
        if (pairs or len(requests) > 1) and (pairs is None or len(pairs) != len(requests)):
            raise ScenarioError(f"{path}.pfas_results must contain one result for every PFAS.")
        for pair_index, pair in enumerate(pairs or []):
            pair_path = f"{path}.pfas_results[{pair_index}]"
            _object(pair, pair_path)
            _score_structure(pair, pair_path)
            request = requests[pair_index]
            for field in ("input_identity", "identity_type"):
                expected = request["pfas"].get("identity" if field == "input_identity" else field)
                if field in pair and pair[field] != expected:
                    raise ScenarioError(f"{pair_path}.{field} disagrees with the saved PFAS input.")
            if len(requests) > 1:
                score = pair.get("performance") or pair.get("prediction") or {}
                _number(score.get("mean_logKd"), f"{pair_path}.performance.mean_logKd")
    lookups = result.get("pfas_lookups", [])
    if not isinstance(lookups, list):
        raise ScenarioError("result.pfas_lookups must be a list of objects.")
    for index, lookup in enumerate(lookups):
        _lookup_structure(lookup, f"result.pfas_lookups[{index}]")
    _caveat_list(result.get("scenario_caveats", []), "result.scenario_caveats", required_text=True)


def _validate_result(result: Any, inputs: dict) -> dict:
    validate_result_structure(result)
    expected = scenario_requests(inputs)
    actual = result.get("scenarios")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ScenarioError("Saved results do not describe every PFAS in the saved inputs.")
    for request, saved in zip(expected, actual):
        # Compare scientific inputs, allowing harmless omissions of optional labels.
        if not isinstance(saved, dict) or any(saved.get(key) != request[key] for key in ("water", "temperature_c", "initial_concentration_mg_l")):
            raise ScenarioError("Saved results and scenario inputs disagree. Re-run screening before saving results.")
        if not isinstance(saved.get("pfas"), dict) or any(saved["pfas"].get(key) != request["pfas"].get(key) for key in
                ("identity", "identity_type", "confirmed_smiles", "structure_confirmed")):
            raise ScenarioError("Saved result PFAS identities disagree with the saved inputs.")
        if not isinstance(saved.get("adsorbent", {}), dict) or saved.get("adsorbent", {}).get("categories", []) != []:
            raise ScenarioError("This scenario file uses a screening scope the editor cannot restore.")
        saved_exploratory = saved.get("adsorbent", {}).get("include_exploratory", False)
        if not isinstance(saved_exploratory, bool) or saved_exploratory != inputs.get("include_exploratory_adsorbents", False):
            raise ScenarioError("Saved results and inputs disagree about exploratory adsorbents. Re-run screening.")
    return _portable(result)


def recommendation_bytes(result: dict) -> bytes:
    """Strict JSON: validated inputs/scores, with missing catalog data as null."""
    validate_result_structure(result)
    return json.dumps(_portable(result), ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def make_snapshot(name: str, inputs: dict, reference: dict, result: dict | None = None) -> dict:
    inputs = validate_inputs(inputs)
    snapshot = {"format": SCENARIO_FORMAT, "scenario_version": SCENARIO_VERSION,
                "scenario_id": uuid4().hex, "name": _text(name, "Scenario name", required=True, limit=80),
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bundle_reference": reference, "inputs": inputs, "result": None}
    if result is not None:
        if result.get("result_schema_version") != RESULT_SCHEMA_VERSION:
            raise ScenarioError("Run screening again before saving results in the current format.")
        snapshot["result"] = _validate_result(result, inputs)
    return snapshot


def snapshot_bytes(snapshot: dict) -> bytes:
    public = {key: value for key, value in snapshot.items() if not key.startswith("_")}
    payload = json.dumps(_portable(public), ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(payload) > MAX_SCENARIO_BYTES:
        raise ScenarioError("The scenario exceeds 20 MB. Save its inputs without results or use a smaller mixture.")
    return payload


def read_snapshot(payload: bytes, default_name: str = "Imported scenario") -> dict:
    if len(payload) > MAX_SCENARIO_BYTES:
        raise ScenarioError("Scenario files must be smaller than 20 MB.")
    try:
        doc = json.loads(payload.decode("utf-8-sig"),
                         parse_float=lambda value: finite_number(float(value), "JSON number"),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(doc, dict):
            raise ScenarioError("A scenario file must contain a JSON object.")
        if doc.get("format") != SCENARIO_FORMAT:
            if doc.get("result_schema_version") == RESULT_SCHEMA_VERSION and isinstance(doc.get("scenarios"), list):
                return _recommendation_snapshot(doc, default_name)
            raise ScenarioError("Choose a saved scenario JSON or a current complete recommendation JSON.")
        if doc.get("scenario_version") != SCENARIO_VERSION:
            raise ScenarioError("This scenario version is unsupported. Export it with a compatible app.")
        inputs = validate_inputs(doc.get("inputs"))
        reference = doc.get("bundle_reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("id"), (str, type(None))):
            raise ScenarioError("The scenario has invalid bundle provenance.")
        snapshot = make_snapshot(doc.get("name"), inputs, reference)
        snapshot["scenario_id"] = _text(doc.get("scenario_id"), "Scenario ID", required=True, limit=100)
        snapshot["saved_at"] = _text(doc.get("saved_at"), "Save time", required=True, limit=100)
        if doc.get("result") is not None:
            if not isinstance(doc["result"], dict):
                raise ScenarioError("Saved results must be an object or null.")
            if doc["result"].get("result_schema_version") == RESULT_SCHEMA_VERSION:
                snapshot["result"] = _validate_result(doc["result"], inputs)
            else:
                snapshot["_notices"] = ["The saved result format is unsupported; inputs can be loaded and screened again."]
        snapshot["_imported"] = True
        return snapshot
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
        if isinstance(exc, ScenarioError):
            raise
        raise ScenarioError(f"Invalid scenario JSON: {exc}") from exc


def _recommendation_snapshot(result: dict, name: str) -> dict:
    validate_result_structure(result)
    requests = result["scenarios"]
    if not requests or not all(isinstance(row, dict) for row in requests):
        raise ScenarioError("The recommendation export has no usable scenario inputs.")
    rows = []
    for request in requests:
        submitted = request.get("concentration_input") or {}
        rows.append({**request.get("pfas", {}), "initial_concentration_mg_l": request.get("initial_concentration_mg_l"),
                     "concentration_value": submitted.get("value", request.get("initial_concentration_mg_l")),
                     "concentration_unit": submitted.get("unit", "mg/L")})
    inputs = {"pfas": rows, "water": requests[0].get("water", {}), "temperature_c": requests[0].get("temperature_c"),
              "include_exploratory_adsorbents": requests[0].get("adsorbent", {}).get("include_exploratory", False)}
    bundle = result.get("bundle") or {}
    reference = {"id": None, "name": str(bundle.get("directory", "Unknown bundle")).replace("\\", "/").rstrip("/").split("/")[-1],
                 "model_family": bundle.get("model_family"), "target": bundle.get("target")}
    snapshot = make_snapshot(name[:80], inputs, reference, result)
    snapshot.update({"_imported": True, "_notices": ["This recommendation export has no bundle fingerprint. Re-screen to compare changes with verified bundle provenance."]})
    return snapshot


def add_snapshot(library: list[dict], snapshot: dict) -> tuple[list[dict], dict]:
    if len(library) >= MAX_SAVED_SCENARIOS:
        raise ScenarioError(f"Keep at most {MAX_SAVED_SCENARIOS} scenarios in this session. Download and remove one before adding another.")
    snapshot = dict(snapshot)
    if any(item["scenario_id"] == snapshot["scenario_id"] for item in library):
        snapshot["scenario_id"] = uuid4().hex
    names = {item["name"].casefold() for item in library}
    base = snapshot["name"]
    suffix = 2
    while snapshot["name"].casefold() in names:
        snapshot["name"] = f"{base[:72]} ({suffix})"
        suffix += 1
    size = len(snapshot_bytes(snapshot)) + sum(len(snapshot_bytes(item)) for item in library)
    if size > MAX_LIBRARY_BYTES:
        raise ScenarioError("Saved scenarios exceed the 60 MB session limit. Download and remove a scenario before adding another.")
    return [*library, snapshot], snapshot


def bundle_reference(directory: Path) -> dict:
    """Content hashes are portable across machines and distinguish repackaged data."""
    manifest = json.loads((directory / "recommender_bundle.json").read_text(encoding="utf-8"))
    names = {"recommender_bundle.json"}
    def filenames(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                filenames(child)
        elif isinstance(value, str):
            path = directory / value
            if not Path(value).is_absolute() and path.resolve().is_relative_to(directory.resolve()) and path.is_file():
                names.add(Path(value).as_posix())
    filenames(manifest)
    hashes = {}
    for name in sorted(names):
        digest = hashlib.sha256()
        with (directory / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    from rdkit import rdBase
    implementation = {}
    for name in ("service.py", "catalog.py", "evidence.py", "pfas_cache.py", "pfas_descriptors.py", "pfas_support.py", "material_support.py", "water_support.py", "additive_inputs.py", "input_validation.py"):
        source = Path(__file__).parent / name
        implementation[name] = hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    runtime = {"rdkit": rdBase.rdkitVersion}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "xgboost-cpu"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            pass
    complete = all(manifest.get(section, {}).get("snapshot_filename") in names for section in ("pfas_features", "database_evidence"))
    override = os.environ.get("ADSORBENT_PFAS_FEATURES_PATH")
    if override and Path(override).is_file():
        hashes["runtime_pfas_features_override"] = hashlib.sha256(Path(override).read_bytes()).hexdigest()
    # Older bundles can use external workbooks. Their inputs remain reusable,
    # but a bundle-only hash cannot establish that those data were unchanged.
    provenance = {"files": hashes, "implementation": implementation, "runtime": runtime,
                  "descriptor_version": DESCRIPTOR_VERSION, "rdkit_version": rdBase.rdkitVersion, "data_snapshots_complete": complete}
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return {"id": digest, "name": directory.name, "model_family": manifest.get("model_family"),
            "target": manifest.get("target"), **provenance}


def _pfas_signature(snapshot: dict) -> tuple[str, ...] | None:
    lookups = snapshot["result"].get("pfas_lookups", [])
    if len(lookups) != len(snapshot["inputs"]["pfas"]):
        return None
    try:
        return tuple(sorted(prepare_structure(row["canonical_smiles"])[1] for row in lookups))
    except (ValueError, KeyError, TypeError):
        return None


def _notes(candidate: dict) -> str:
    notes = []
    for item in candidate.get("caveats", []):
        if isinstance(item, dict) and item.get("text"):
            names = ", ".join(str(name) for name in item.get("pfas") or [])
            notes.append((f"{names}: " if names else "") + str(item["text"]))
    # Schema-5 caveats belong to the score actually used. A database-backed
    # score must not inherit caveats from the model computed alongside it.
    if "caveats" not in candidate:
        notes += [str(item) for item in candidate.get("warnings", [])]
    if candidate.get("mixture_performance", {}).get("limiting_pfas_unseen"):
        notes.insert(0, "An unseen PFAS determines this candidate's limiting score.")
    return " | ".join(dict.fromkeys(filter(None, notes)))


def compare_snapshots(snapshots: list[dict], reference_id: str) -> tuple[list[dict], list[str]]:
    if not 2 <= len(snapshots) <= 4 or any(not item.get("result") for item in snapshots):
        raise ScenarioError("Select two to four scenarios with saved screening results.")
    for snapshot in snapshots:
        validate_result_structure(snapshot["result"])
    reference = next((item for item in snapshots if item["scenario_id"] == reference_id), None)
    if reference is None:
        raise ScenarioError("Choose a reference scenario from this comparison.")
    ref_signature = _pfas_signature(reference)
    maps = [{item["product"]["Product_key"]: item for item in snapshot["result"]["recommendations"]} for snapshot in snapshots]
    ref_map = maps[snapshots.index(reference)]
    warnings, comparable = [], []
    for snapshot in snapshots:
        same_bundle = (bool(reference["bundle_reference"].get("id"))
                       and snapshot["bundle_reference"].get("id") == reference["bundle_reference"]["id"]
                       and reference["bundle_reference"].get("data_snapshots_complete", True)
                       and snapshot["bundle_reference"].get("data_snapshots_complete", True))
        same_pfas = ref_signature is not None and _pfas_signature(snapshot) == ref_signature
        comparable.append(same_bundle and same_pfas)
        if not same_bundle:
            warnings.append(f"{snapshot['name']}: bundle provenance differs or is unavailable; score changes against the reference are withheld.")
        if not same_pfas:
            warnings.append(f"{snapshot['name']}: the PFAS mixture differs or its structural identity is unavailable; limiting scores describe different mixtures, so score changes are withheld.")
    keys = set().union(*(mapping.keys() for mapping in maps))
    rows = []
    for key in sorted(keys, key=lambda key: (ref_map.get(key, {}).get("rank", math.inf), key)):
        exemplar = next(mapping[key] for mapping in maps if key in mapping)
        row = {"Product key": key, "Product": exemplar["product"].get("Name_Commercial") or key,
               "Class": exemplar["product"].get("adsorbent_category", "")}
        for snapshot, mapping, can_compare in zip(snapshots, maps, comparable):
            item = mapping.get(key)
            name = snapshot["name"]
            row[f"{name} · rank"] = item["rank"] if item else None
            row[f"{name} · logKd"] = item["ranking_score"] if item else None
            row[f"{name} · ΔlogKd"] = item["ranking_score"] - ref_map[key]["ranking_score"] if item and key in ref_map and can_compare else None
            row[f"{name} · basis"] = {"database_match": "Database", "model_prediction": "Model", "mixed_evidence": "Mixed evidence"}.get(item.get("performance_source"), "Unspecified") if item else "Not in saved ranking"
            row[f"{name} · limiting PFAS"] = item.get("mixture_performance", {}).get("limiting_pfas", "") if item else ""
            row[f"{name} · caveats"] = _notes(item) if item else ""
        rows.append(row)
    return rows, list(dict.fromkeys(warnings))


def comparison_csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    def safe(value: Any) -> Any:
        return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
    writer = csv.writer(stream)
    writer.writerow([safe(key) for key in rows[0]] if rows else [])
    for row in rows:
        writer.writerow(safe(value) for value in row.values())
    return stream.getvalue().encode("utf-8-sig")


SCENARIO_CSV_COLUMNS = ("scenario_name", "row_type", "pfas", "identity_type", "inorganic_species", "concentration", "unit", "parent_smiles", "label",
    "water_type", "ph", "temperature_c", "organic_carbon_mg_l", "organic_matter_present", "tds_mg_l", "ionic_strength_mol_l",
    *(f"ion_{key.lower()}" for key in ION_KEYS), "include_exploratory_adsorbents")


def scenario_inputs_csv(snapshot: dict) -> bytes:
    """One typed row per PFAS or inorganic additive, with shared settings repeated."""
    inputs = snapshot["inputs"]
    water = inputs["water"]
    common = {"scenario_name": snapshot["name"], "water_type": water["water_type"], "ph": water["pH"],
              "include_exploratory_adsorbents": str(inputs.get("include_exploratory_adsorbents", False)).lower(),
              "temperature_c": inputs["temperature_c"], **{key: water[key] for key in WATER_NUMBERS if key != "pH"},
              "organic_matter_present": str(water["organic_matter_present"]).lower(),
              **{f"ion_{key.lower()}": str(water["ions"][key]).lower() for key in ION_KEYS}}
    rows = []
    for pfas in inputs["pfas"]:
        row = {**common, "row_type": "pfas", "pfas": pfas["identity"], "identity_type": pfas["identity_type"],
               "concentration": pfas["concentration_value"], "unit": pfas["concentration_unit"],
               "parent_smiles": "" if pfas["identity_type"] == "smiles" else pfas.get("confirmed_smiles") or pfas.get("canonical_smiles", ""),
               "label": pfas.get("label", "")}
        rows.append({key: row.get(key) for key in SCENARIO_CSV_COLUMNS})
    for additive in water.get("inorganic_additives", []):
        row = {**common, "row_type": "inorganic_additive", "inorganic_species": additive["species"],
               "concentration": additive["concentration"], "unit": additive["unit"]}
        rows.append({key: row.get(key) for key in SCENARIO_CSV_COLUMNS})
    def text_cell(value: Any) -> Any:
        if isinstance(value, str) and (value.startswith("'") or value.lstrip().startswith(("=", "+", "-", "@"))):
            return "'" + value
        return value
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SCENARIO_CSV_COLUMNS)
    writer.writeheader()
    writer.writerows({key: text_cell(value) for key, value in row.items()} for row in rows)
    payload = stream.getvalue().encode("utf-8-sig")
    if len(payload) > 1_000_000:
        raise ScenarioError("The input CSV exceeds 1 MB. Use the full JSON snapshot under Advanced.")
    return payload


def read_scenario_csv(payload: bytes, default_name: str = "Imported scenario") -> tuple[str, dict]:
    if len(payload) > 1_000_000:
        raise ScenarioError("The input CSV exceeds 1 MB.")
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
        columns = [str(name).strip().casefold() for name in reader.fieldnames or []]
        if len(columns) != len(set(columns)):
            raise ScenarioError("CSV column names must be unique.")
        reader.fieldnames = columns
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ScenarioError(f"CSV line {reader.line_num} has an incorrect number of fields.")
            if any(str(value).strip() for value in row.values()):
                rows.append({**{key: str(value).strip() for key, value in row.items()}, "csv_row": reader.line_num})
            if len(rows) > 150:
                raise ScenarioError("Use at most 100 PFAS and 50 additive rows.")
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ScenarioError("Use a well-formed CSV UTF-8 file with the current example headers.") from exc
    if not rows:
        raise ScenarioError("The scenario CSV has no input rows.")
    if any("rank" in row or "product_key" in row for row in rows):
        raise ScenarioError("This is a results CSV. Use the scenario inputs CSV to restore mixture and water settings.")
    if {"row_type", "pfas", "inorganic_species", "concentration", "unit"} - set(columns):
        raise ScenarioError("Use the current example CSV with row_type, pfas, inorganic_species, concentration, and unit columns.")
    unknown = set(rows[0]) - set(SCENARIO_CSV_COLUMNS) - {"csv_row"}
    if unknown:
        raise ScenarioError("Unrecognized scenario CSV columns: " + ", ".join(sorted(unknown)) + ". Start with the example inputs CSV so settings are not omitted.")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and value.startswith("'") and (value[1:].startswith("'") or value[1:].lstrip().startswith(("=", "+", "-", "@"))):
                row[key] = value[1:]
    names = {row.get("scenario_name", "").strip() for row in rows if row.get("scenario_name", "").strip()}
    if len(names) > 1:
        raise ScenarioError("Use one scenario_name per file. Import different scenarios as separate CSV files.")
    name = next(iter(names), default_name)
    def optional_number(raw: str, field: str) -> float | None:
        if not raw.strip():
            return None
        try:
            value = float(raw)
        except ValueError as exc:
            raise ScenarioError(f"{field} must be numeric or blank.") from exc
        return _number(value, field)
    def ion_value(raw: str, field: str) -> bool:
        if raw.casefold() in {"", "false", "0", "no"}:
            return False
        if raw.casefold() in {"true", "1", "yes"}:
            return True
        raise ScenarioError(f"{field} must be true/false, yes/no, 1/0, or blank.")
    shared, pfas_rows, additives = None, [], []
    for row in rows:
        water = {key: optional_number(row.get("ph" if key == "pH" else key, ""), key) for key in WATER_NUMBERS}
        water["water_type"] = row.get("water_type", "")
        water["organic_matter_present"] = ion_value(row.get("organic_matter_present", ""), "organic_matter_present")
        water["ions"] = {key: ion_value(row.get(f"ion_{key.lower()}", ""), f"ion_{key.lower()}") for key in ION_KEYS}
        current = {"water": water, "temperature_c": optional_number(row.get("temperature_c", ""), "temperature_c"),
                   "include_exploratory_adsorbents": ion_value(row.get("include_exploratory_adsorbents", ""), "include_exploratory_adsorbents")}
        if shared is not None and current != shared:
            raise ScenarioError(f"CSV line {row['csv_row']}: water chemistry, temperature, and exploratory adsorbent settings must be identical for all rows in one scenario. Repeat shared settings on each row.")
        shared = current
        row_type = row.get("row_type", "").casefold()
        if row_type == "inorganic_additive":
            if any(row.get(key) for key in ("pfas", "identity_type", "parent_smiles", "label")):
                raise ScenarioError(f"CSV line {row['csv_row']}: an additive row cannot also contain PFAS identity fields.")
            additives.append({"species": row.get("inorganic_species"), "concentration": optional_number(row["concentration"], "additive concentration"), "unit": row["unit"]})
            continue
        if row_type != "pfas" or row.get("inorganic_species"):
            raise ScenarioError(f"CSV line {row['csv_row']}: use row_type pfas or inorganic_additive, with one identity per row.")
        kind = row.get("identity_type", "").casefold() or "name"
        amount = optional_number(row["concentration"], "concentration")
        unit = normalize_unit(row["unit"])
        pfas = {"identity": row["pfas"], "identity_type": kind, "label": row.get("label") or row["pfas"],
                "concentration_value": amount, "concentration_unit": unit, "initial_concentration_mg_l": to_mg_l(amount, unit)}
        parent = row.get("parent_smiles", "")
        if parent or kind == "smiles":
            pfas.update({"confirmed_smiles": parent or row["pfas"], "structure_confirmed": True,
                         "resolution": {"source": "scenario_csv", "query": row["pfas"], "identity_type": kind}})
        pfas_rows.append(pfas)
    shared["water"]["inorganic_additives"] = additives
    return _text(name, "Scenario name", required=True, limit=80), validate_inputs({"pfas": pfas_rows, **shared})


def results_csv(result: dict, name: str = "Screening result") -> bytes:
    """One row per candidate–PFAS pair, with the aggregate rank alongside it."""
    validate_result_structure(result)
    rows = []
    requests = result.get("scenarios", [])
    for candidate in result.get("recommendations", []):
        product = candidate["product"]
        mixture = candidate.get("mixture_performance", {})
        per_pfas = candidate.get("pfas_results") or [candidate]
        for index, item in enumerate(per_pfas):
            request = requests[index] if index < len(requests) else {}
            submitted = item.get("concentration_input") or request.get("concentration_input") or {}
            lookup = item.get("pfas_lookup") or (result.get("pfas_lookups") or [{}])[0]
            support = item.get("pfas_support") or lookup.get("support", {})
            performance = item.get("performance") or item.get("prediction") or {}
            evidence = item.get("database_evidence") or {}
            related = evidence.get("related_evidence") or {}
            material = item.get("adsorbent_support") or candidate.get("adsorbent_support") or {}
            rows.append({"scenario_name": name, "rank": candidate["rank"], "product_key": product["Product_key"],
                "effective_ionic_strength_mol_l": result.get("water_chemistry", {}).get("ionic_strength_mol_l"),
                "normalized_water_type": result.get("water_chemistry", {}).get("normalized_water_type"),
                "water_training_model_rows": result.get("water_support", {}).get("model_rows"),
                "water_training_source_records": result.get("water_support", {}).get("source_records"),
                "water_training_studies": result.get("water_support", {}).get("studies"),
                "water_crosswater_studies": result.get("water_support", {}).get("studies_with_other_water"),
                "water_model_support_warning": result.get("water_support", {}).get("warning", "") if item.get("performance_source") == "model_prediction" else "",
                "organic_matter_present": result.get("water_chemistry", {}).get("organic_matter_present"),
                "organic_carbon_mg_l": result.get("water_chemistry", {}).get("organic_carbon_mg_l"),
                "include_exploratory_adsorbents": request.get("adsorbent", {}).get("include_exploratory", False),
                "product": product.get("Name_Commercial") or product["Product_key"], "class": product.get("adsorbent_category", ""),
                "limiting_logkd": candidate["ranking_score"], "mean_logkd": mixture.get("mean_logKd"),
                "limiting_pfas": mixture.get("limiting_pfas", ""), "limiting_pfas_unseen": mixture.get("limiting_pfas_unseen"),
                "aggregate_basis": candidate.get("performance_source"), "pfas": item.get("pfas") or lookup.get("label") or request.get("pfas", {}).get("identity", ""),
                "canonical_smiles": lookup.get("canonical_smiles", ""), "concentration": submitted.get("value"),
                "unit": submitted.get("unit", "mg/L"), "concentration_mg_l": item.get("initial_concentration_mg_l", request.get("initial_concentration_mg_l")),
                "pfas_logkd": performance.get("mean_logKd", candidate["ranking_score"] if len(per_pfas) == 1 else None),
                "pfas_basis": item.get("performance_source"), "seen_in_training": support.get("seen_in_training"),
                "comparable_record_count": evidence.get("performance", {}).get("match_count", 0),
                "related_record_count": related.get("record_count", 0), "related_study_count": related.get("study_count", 0),
                "related_evidence_summary": related.get("summary", ""),
                "adsorbent_support_status": material.get("status", "unknown"),
                "adsorbent_training_rows": material.get("training_rows"), "adsorbent_source_records": material.get("source_records"),
                "adsorbent_independent_studies": material.get("independent_studies"),
                "adsorbent_exploratory_prediction": material.get("exploratory_prediction"),
                "feature_source": lookup.get("feature_source", ""), "candidate_caveats": _notes(candidate),
                "pfas_caveats": _notes(item), "scenario_caveats": " | ".join(caveat["text"] for caveat in result.get("scenario_caveats", []))})
    return comparison_csv(rows)
