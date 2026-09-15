"""Unit normalization and conversion helpers for the adsorption database pipeline."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Optional

import pandas as pd

from adsorbent_consolidation import (
    ADSORBENT_PROPERTY_COLUMN_MAP,
    NumericProperty,
    parse_numeric_values,
)
from normalization_rules import (
    _append_normalization_notes,
    _as_float_or_blank,
    _blank,
    _coerce_number,
    _debug_pfas_c0_row,
    _ensure_columns,
    get_pfasmw,
)

PARAM_SPECS = {
    "Langmuir_Qm": dict(
        value_col="Langmuir_Qm_value", unit_col="Langmuir_Qm_unit",
        target="mg/g",
        dim=["amount", "mass"],
        bases={"amount": "mg", "mass": "g"}
    ),
    "Langmuir_KL": dict(
        value_col="Langmuir_KL_value", unit_col="Langmuir_KL_unit",
        target="L/mg",
        dim=["volume", "amount"],
        bases={"volume": "L", "amount": "mg"}
    ),
    "Kd": dict(
        value_col="Kd_value", unit_col="Kd_unit",
        target="L/g",
        dim=["volume", "mass"],
        bases={"volume": "L", "mass": "g"}
    ),
    "Qe": dict(
        value_col="Qe_value", unit_col="Qe_unit",
        target="mg/g",
        dim=["amount", "mass"],
        bases={"amount": "mg", "mass": "g"}
    ),
    "PFO_k1": dict(
        value_col="PFO_k1_value", unit_col="PFO_k1_unit",
        target="1/h",
        dim=["unity", "time"],
        bases={"time": "h"}
    ),
    "PFO_Qe": dict(
        value_col="PFO_Qe_value", unit_col="PFO_Qe_unit",
        target="mg/g",
        dim=["amount", "mass"],
        bases={"amount": "mg", "mass": "g"}
    ),
    "PSO_k2": dict(
        value_col="PSO_k2_value", unit_col="PSO_k2_unit",
        target="g/(mg·h)",
        dim=["mass", ("amount", "time")],
        bases={"mass": "g", "amount": "mg", "time": "h"}
    ),
    "PSO_v0": dict(
        value_col="PSO_v0_value", unit_col="PSO_v0_unit",
        target="mg/(g·h)",
        dim=["amount", ("mass", "time")],
        bases={"amount": "mg", "mass": "g", "time": "h"}
    ),
    "PSO_Qe": dict(
        value_col="PSO_Qe_value", unit_col="PSO_Qe_unit",
        target="mg/g",
        dim=["amount", "mass"],
        bases={"amount": "mg", "mass": "g"}
    ),
    "PFAS_C0": dict(
        value_col="PFAS_C0_value", unit_col="PFAS_C0_unit",
        target="mg/L",
        dim=["amount", "volume"],
        bases={"amount": "mg", "volume": "L"}
    ),
    "Adsorbent_dosage": dict(
        value_col="Adsorbent_dosage_value", unit_col="Adsorbent_dosage_unit",
        target="mg/L",
        dim=["amount", "volume"],
        bases={"amount": "mg", "volume": "L"}
    ),
    "Nominal_grain_size": dict(
        value_col="Nominal_grain_size_value", unit_col="Nominal_grain_size_unit",
        target="mm",
        dim=["length"],
        bases={"length": "mm"}
    ),
    # IEC keeps split outputs (meq/g and meq/L) like before
    "ion_exchange_capacity": dict(
        value_col="ion_exchange_capacity_value", unit_col="ion_exchange_capacity_unit",
        target="meq/g",  # used only for naming; converter writes two columns
        dim=["amount", "mass"],
        bases={"amount": "meq", "mass": "g"}
    ),
    # KF is special; we keep the original converter and decision logic
    "Freundlich_KF": dict(
        value_col="Freundlich_KF_value", unit_col="Freundlich_KF_unit",
        target="(mg/g)/(mg/L)^(1/n)",
        dim=None,
        bases=None
    ),
}

FREUNDLICH_EXPONENT_GREATER_THAN_ONE_COLUMN = "Freundlich_exponent_greater_than_1"

# ---------- Precompiled regex patterns ----------
try:
    KF_PROD_RE = re.compile(
        r"^\(?(?P<A>[^)]+)\)?(?:·|\*)?\(?(?P<B>(?:(?:[lL]|m[Ll])/[^\)]+|[^/]+/(?:[lL]|m[Ll])))\)?"
        r"\^\(?(?P<exp>-?(?:(?:1/)?[A-Za-z\u0370-\u03ff]+|[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?))\)?$"
    )
except re.error:
    KF_PROD_RE = re.compile(r"^\(?(?P<A>[^)]+)\)?(?:·|\*)\(?(?P<B>[^)]+)\)?\^\(?(?P<exp>[^)]+)\)?$")

try:
    KF_DIV_RE = re.compile(
        r"^\(?(?P<A>[^)]+)\)?/\(?(?P<B>[^)\^]+)\)?\^\(?(?P<exp>-?(?:(?:1/)?[A-Za-z\u0370-\u03ff]+|[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?))\)?$"
    )
except re.error:
    KF_DIV_RE = re.compile(r"^\(?(?P<A>[^)]+)\)?/\(?(?P<B>[^)\^]+)\)?\^\(?(?P<exp>[^)]+)\)?$")

SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")

def _coerce_or_blank(v):
    n = _coerce_number(v)
    return n if n is not None else ""


PORE_VOLUME_TARGET_UNIT = "cm3/g"
PORE_VOLUME_VALUE_COLUMNS = {
    "pore_volume_total_value": f"pore_volume_total_value_{PORE_VOLUME_TARGET_UNIT}",
    "pore_volume_micro_value": f"pore_volume_micro_value_{PORE_VOLUME_TARGET_UNIT}",
    "pore_volume_meso_value": f"pore_volume_meso_value_{PORE_VOLUME_TARGET_UNIT}",
    "pore_volume_macro_value": f"pore_volume_macro_value_{PORE_VOLUME_TARGET_UNIT}",
    "pore_volume_micro_value_avg": f"pore_volume_micro_value_avg_{PORE_VOLUME_TARGET_UNIT}",
    "pore_volume_meso_value_avg": f"pore_volume_meso_value_avg_{PORE_VOLUME_TARGET_UNIT}",
    "pore_volume_macro_value_avg": f"pore_volume_macro_value_avg_{PORE_VOLUME_TARGET_UNIT}",
}


def _pore_volume_factor_to_cm3_per_g(unit_raw) -> float | None:
    if _blank(unit_raw):
        return 1.0

    unit = unicodedata.normalize("NFKC", str(unit_raw)).strip().lower()
    unit = unit.replace(" ", "").replace("_", "")
    unit = unit.replace("\u00b5", "u").replace("\u03bc", "u")
    unit = unit.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    unit = unit.replace("\u00b3", "3").replace("^3", "3")
    unit = unit.replace("cubiccentimeter", "cm3").replace("cubiccentimetre", "cm3")
    unit = unit.replace("milliliter", "ml").replace("millilitre", "ml")
    unit = unit.replace("cc", "cm3")

    if unit in {"cm3/g", "cm3g-1", "cm3g^-1", "ml/g", "mlg-1", "mlg^-1"}:
        return 1.0
    if unit in {"l/g", "lg-1", "lg^-1"}:
        return 1000.0
    if unit in {"cm3/kg", "cm3kg-1", "cm3kg^-1", "ml/kg", "mlkg-1", "mlkg^-1"}:
        return 0.001
    if unit in {"l/kg", "lkg-1", "lkg^-1"}:
        return 1.0
    return None


def _pore_volume_to_cm3_per_g(value, unit_raw):
    number = _coerce_number(value)
    if number is None:
        return ""
    factor = _pore_volume_factor_to_cm3_per_g(unit_raw)
    if factor is None:
        return ""
    return number * factor


def normalize_pore_volumes(df_in: pd.DataFrame) -> None:
    cols = [
        # pore volumes
        "pore_volume_micro_value",
        "pore_volume_meso_value",
        "pore_volume_macro_value",
        # porosities
        "porosity_micro_value",
        "porosity_meso_value",
        "porosity_macro_value",
    ]
    for col in cols:
        if col in df_in.columns and f"{col}_avg" not in df_in.columns:
            df_in[f"{col}_avg"] = df_in[col].apply(_coerce_or_blank)

    unit_series = (
        df_in["pore_volume_unit"]
        if "pore_volume_unit" in df_in.columns
        else pd.Series([""] * len(df_in), index=df_in.index)
    )
    for src_col, out_col in PORE_VOLUME_VALUE_COLUMNS.items():
        if src_col in df_in.columns:
            df_in[out_col] = [
                _pore_volume_to_cm3_per_g(value, unit)
                for value, unit in zip(df_in[src_col], unit_series)
            ]

def get_solution_volume_L(row):
    col = "Solution_volume_(mL)"
    try:
        return float(row[col]) / 1000.0 if col in row.index else None
    except Exception:
        return None


def normalize_unit(u: str) -> str:
    if not isinstance(u, str) or not u.strip():
        return ""
    s = unicodedata.normalize("NFKC", u).replace("\u00A0", " ").strip()
    # Normalize Greek mu and do *molar* substitutions BEFORE lowercasing
    s = s.replace("μ", "µ")
    # µM, uM  →  µmol/L  (molar concentration, not mass)
    s = re.sub(r"\b(?:µM|uM)\b", "µmol/L", s)
    # mM      →  mmol/L
    s = re.sub(r"\bmM\b", "mmol/L", s)
    # bare 'M' (e.g. 0.01 M) → mol/L
    s = re.sub(r"(?<![A-Za-z0-9])M(?![A-Za-z0-9])", "mol/L", s)

    # Now go to lowercase and normalize superscripts
    s = s.lower().translate(SUPERSCRIPT_MAP)
    # PPM → mg/L, PPB → µg/L, PPT → ng/L (volume variants treated similarly)
    s = re.sub(r"\bppm(v|/v)?\b", "mg/l", s)
    s = re.sub(r"\bppb(v|/v)?\b", "µg/l", s)
    s = re.sub(r"\bppt(v|/v)?\b", "ng/l", s)
    s = re.sub(r"\b(h|hr|hour|hours)\s*\^?\s*-?1\b", "1/h", s, flags=re.I)
    s = re.sub(r"\b(min|minute|minutes)\s*\^?\s*-?1\b", "1/min", s, flags=re.I)
    s = re.sub(r"\b(s|sec|second|seconds)\s*\^?\s*-?1\b", "1/s", s, flags=re.I)
    s = re.sub(r"\b(d|day|days)\s*\^?\s*-?1\b", "1/day", s, flags=re.I)
    s = re.sub(r"\bper\b", "/", s)
    s = re.sub(r"\b([µu]?mol|mmol|mol)\s*[/ ]\s*(g|l|ml)\b", r"\1/\2", s)
    s = re.sub(r"\b([µu]?mol|mmol|mol)\s*(g|l|ml)\s*[-^]?\s*1\b", r"\1/\2", s)
    s = re.sub(r"\b(mg|µg|ng|g)\s*(l|ml)\s*[-^]?\s*1\b", r"\1/\2", s)

    s = s.replace(" ", "")
    s = s.replace("*", "·").replace("⋅", "·").replace("×", "·")
    # A period between unit tokens is a multiplication separator in source
    # tables (for example, ``g/µg.h``); decimal values are unaffected.
    s = re.sub(r"(?<=[a-zµ])\.(?=[a-zµ])", "·", s)
    # Some tables use a hyphen before a time unit as multiplication, e.g.
    # ``g/mg-h`` for ``g/(mg·h)``.
    s = re.sub(
        r"(?<=[a-zµ])-(?=(?:s|sec|min|h|hr|day|d)(?:$|[·/)]))",
        "·",
        s,
    )
    s = re.sub(r"(?<=[a-zA-Zµ])x(?=[a-zA-Z])", "·", s)

    s = re.sub(r"\bumol\b", "µmol", s)
    s = re.sub(r"\bug\b", "µg", s)
    s = s.replace("m^2", "m2").replace("m²", "m2")

    s = s.replace("//", "/")
    s = re.sub(r"(^|[^a-z])l(?=/|$)", lambda m: m.group(1) + "L", s)
    s = re.sub(r"(^|[^a-z])ml(?=/|$)", lambda m: m.group(1) + "mL", s)
    s = s.replace("l-1", "/L").replace("ml-1", "/mL")
    s = s.replace("mg/hr", "mg/h").replace("g/hr", "g/h")
    s = s.replace("mg-hr", "mg·h").replace("mgmin", "mg·min").replace("gmin", "g·min")
    if s in {"µm", "um", "μm"}:
        return "µm"
    return s


# ---------------- Dimension engine ----------------

MASS_UNITS = {"ng": 1e-6, "µg": 1e-3, "ug": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}
VOL_UNITS = {"µl": 1e-6, "ul": 1e-6, "ml": 1e-3, "l": 1.0}
TIME_UNITS = {"s": 1 / 3600.0, "sec": 1 / 3600.0, "min": 1 / 60.0, "h": 1.0, "hr": 1.0, "day": 24.0, "d": 24.0}
LEN_UNITS = {"nm": 1e-6, "µm": 1e-3, "um": 1e-3, "μm": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0}
SURF_UNITS = {"m2": 1.0}

def amount_token_to_mg(token: str, mw: float):
    t = token
    if t in MASS_UNITS:
        return MASS_UNITS[t]
    if t in {"mol", "mmol", "µmol", "umol"}:
        if mw is None:
            return None
        if t == "mol":
            return mw * 1000.0
        if t == "mmol":
            return mw
        return mw / 1000.0
    if t in {"eq", "meq"}:
        if mw is None:
            return None
        if t == "eq":
            return mw * 1000.0
        return mw
    return None


def mass_token_to_g(token: str):
    if token in MASS_UNITS:
        return MASS_UNITS[token] / 1000.0
    return None


def volume_token_to_L(token: str):
    return VOL_UNITS.get(token, None)


def time_token_to_h(token: str):
    return TIME_UNITS.get(token, None)


def length_token_to_mm(token: str):
    return LEN_UNITS.get(token, None)


def surface_token_to_m2(token: str):
    return SURF_UNITS.get(token, None)


MESH_TO_MM = {
    4: 4.75, 5: 4.00, 6: 3.35, 7: 2.80, 8: 2.36, 10: 2.00, 12: 1.70,
    14: 1.40, 16: 1.18, 18: 1.00, 20: 0.850, 25: 0.710, 30: 0.600,
    35: 0.500, 40: 0.425, 45: 0.355, 50: 0.300, 60: 0.250, 70: 0.212,
    80: 0.180, 100: 0.150, 120: 0.125, 140: 0.106, 170: 0.090,
    200: 0.075, 230: 0.063, 270: 0.053, 325: 0.045, 400: 0.038,
}


def standardize_grain_size_values(value, unit) -> tuple[list[float], str]:
    """Convert one raw grain-size expression to one or more values in mm.

    Ranges retain both endpoints here. Consolidation uses those endpoints for
    min/max traceability, while the performance output can select its scalar
    representation separately.
    """
    if _blank(value):
        return [], ""

    raw_value = str(value)
    raw_unit = str(unit or "")
    if "&&" in raw_unit:
        value_parts = [part.strip() for part in raw_value.split("&&") if part.strip()]
        unit_parts = [part.strip() for part in raw_unit.split("&&") if part.strip()]
        if len(value_parts) != len(unit_parts):
            return [], "value/unit list length mismatch"
        converted: list[float] = []
        for value_part, unit_part in zip(value_parts, unit_parts):
            part_values, part_note = standardize_grain_size_values(
                value_part, unit_part
            )
            if part_note or not part_values:
                return [], part_note or "unrecognized value/unit in list"
            converted.extend(part_values)
        return converted, "multi-value list converted to mm"

    unit_norm = normalize_unit(raw_unit)
    if re.fullmatch(r"\s*(?:µ|μ|u)\s*[mM]\s*\.?\s*", raw_unit):
        unit_norm = "µm"
    values = parse_numeric_values(raw_value)
    if not values:
        return [], "value not parseable"

    if "mesh" in unit_norm:
        converted = []
        missing = []
        for value_number in values:
            mesh = int(round(value_number))
            mm = MESH_TO_MM.get(mesh)
            if mm is None:
                missing.append(mesh)
            else:
                converted.append(mm)
        if missing:
            return [], f"mesh size(s) not in lookup: {', '.join(map(str, missing))}"
        return converted, "mesh converted to mm using sieve-opening lookup"

    factors = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "µm": 0.001,
        "um": 0.001,
        "μm": 0.001,
        "micron": 0.001,
        "microns": 0.001,
        "nm": 1e-6,
        "cm": 10.0,
    }
    factor = factors.get(unit_norm)
    if factor is None:
        return [], f"unit {unit_norm or '<blank>'!r} not converted to mm"
    return [number * factor for number in values], ""


def _performance_grain_size_value(value, unit):
    values, note = standardize_grain_size_values(value, unit)
    if not values:
        return "", note
    if "&&" in str(value) or "&&" in str(unit):
        return "&&".join(f"{number:.12g}" for number in values), (
            note or "multi-value list kept; no averaging"
        )
    if len(values) == 1:
        return values[0], note
    return sum(values) / len(values), (
        (note + "; " if note else "") + "range/list averaged in mm"
    )


def _expand_inverse_exponents(num_tokens: list[str], den_tokens: list[str]):
    """Move explicit ``unit^-1`` factors across the fraction bar."""
    out_num: list[str] = []
    out_den: list[str] = []
    inverse_re = re.compile(r"^(?P<token>[A-Za-zµ]+)(?:\^?-1)$")

    def add(tokens: list[str], direct: list[str], inverse: list[str]):
        for raw_token in tokens:
            token = raw_token.strip()
            if not token:
                continue
            match = inverse_re.fullmatch(token)
            if match:
                inverse.append(match.group("token"))
            else:
                direct.append(token)

    add(num_tokens, out_num, out_den)
    add(den_tokens, out_den, out_num)

    # The one in ``mg·µg^-1·1/h`` is neutral; retain it for ``1/h``.
    if len(out_num) > 1:
        out_num = [token for token in out_num if token != "1"]
    if len(out_den) > 1:
        out_den = [token for token in out_den if token != "1"]
    return out_num, out_den


def parse_unit_tokens(u: str):
    if not u:
        return [], []
    s = u

    if s.startswith("(") and s.endswith(")") and s.count("(") == 1:
        s = s[1:-1]

    if s.count("/") >= 2 and "·" not in s:
        parts = s.split("/")
        s = parts[0] + "/(" + "·".join(parts[1:]) + ")"

    if s.startswith("1/"):
        num = ["1"]
        den = s[2:]
        if den.startswith("(") and den.endswith(")"):
            den = den[1:-1]
        den_tokens = den.split("·")
        return _expand_inverse_exponents(num, den_tokens)

    m = re.match(r"^(?P<A>[^/]+?)/\((?P<B>[^)]+)\)$", s)
    if m:
        A = m.group("A")
        B = m.group("B")
        return _expand_inverse_exponents(A.split("·"), B.split("·"))

    m = re.match(r"^(?P<A>[^/]+?)/(?P<B>[^/]+?)$", s)
    if m:
        return _expand_inverse_exponents(
            m.group("A").split("·"), m.group("B").split("·")
        )

    return _expand_inverse_exponents(s.split("·"), [])


def classify_token(token: str):
    t = token
    if t == "1":
        return "unity"
    if t in MASS_UNITS:
        return "mass"
    if t in {"mol", "mmol", "µmol", "umol", "eq", "meq"}:
        return "amount"
    if t in VOL_UNITS:
        return "volume"
    if t in TIME_UNITS:
        return "time"
    if t in LEN_UNITS:
        return "length"
    if t in SURF_UNITS:
        return "surface"
    if t in {"L", "mL", "µL"}:
        return "volume"
    return None


def token_to_base_factor(token: str, desired_dim: str, base: str, mw: float):
    if desired_dim == "unity":
        return 1.0 if token == "1" else None

    if desired_dim == "amount":
        if base != "mg":
            return None
        return amount_token_to_mg(token, mw)

    if desired_dim == "mass":
        if base != "g":
            return None
        return mass_token_to_g(token)

    if desired_dim == "volume":
        if base != "L":
            return None
        tok = token.replace("mL", "ml").replace("µL", "µl")
        return volume_token_to_L(tok)

    if desired_dim == "time":
        if base != "h":
            return None
        return time_token_to_h(token)

    if desired_dim == "length":
        if base != "mm":
            return None
        tok = token.replace("um", "µm")
        return length_token_to_mm(tok)

    if desired_dim == "surface":
        if base != "m2":
            return None
        return surface_token_to_m2(token)

    return None


def compute_factor_against_signature(u_norm: str, spec_dim, spec_bases, mw: float):
    num_tokens, den_tokens = parse_unit_tokens(u_norm)

    if isinstance(spec_dim[0], tuple):
        expected_num_dims = list(spec_dim[0])
    else:
        expected_num_dims = [spec_dim[0]]

    if len(spec_dim) > 1:
        if isinstance(spec_dim[1], tuple):
            expected_den_dims = list(spec_dim[1])
        else:
            expected_den_dims = [spec_dim[1]]
    else:
        expected_den_dims = []

    if len(num_tokens) != len(expected_num_dims) or len(den_tokens) != len(expected_den_dims):
        return None, f"Unit dimensionality mismatch for expected {spec_dim}"

    num_factor = 1.0
    for tok, dim in zip(num_tokens, expected_num_dims):
        cat = classify_token(tok)
        if dim == "amount" and cat in {"amount", "mass"}:
            f = token_to_base_factor(tok, "amount", spec_bases["amount"], mw)
        else:
            if cat != dim and not (dim == "unity" and tok == "1"):
                return None, f"Token '{tok}' not {dim}"
            f = token_to_base_factor(tok, dim, spec_bases.get(dim, ""), mw)

        if f is None:
            if dim == "amount" and tok in {"mol", "mmol", "µmol", "umol", "eq", "meq"} and (mw is None):
                return None, ("PFAS molecular weight is missing; cannot convert "
                              f"'{tok}' → mg for amount base")
            return None, f"Cannot convert token '{tok}' to {dim} base"
        num_factor *= f

    den_factor = 1.0
    for tok, dim in zip(den_tokens, expected_den_dims):
        cat = classify_token(tok)
        if dim == "amount" and cat in {"amount", "mass"}:
            f = token_to_base_factor(tok, "amount", spec_bases["amount"], mw)
        else:
            if cat != dim and not (dim == "unity" and tok == "1"):
                return None, f"Token '{tok}' not {dim}"
            f = token_to_base_factor(tok, dim, spec_bases.get(dim, ""), mw)

        if f is None:
            if dim == "amount" and tok in {"mol", "mmol", "µmol", "umol", "eq", "meq"} and (mw is None):
                return None, ("PFAS molecular weight is missing; cannot convert "
                              f"'{tok}' → mg for amount base")
            return None, f"Cannot convert token '{tok}' to {dim} base"
        den_factor *= f

    return (num_factor / den_factor), ""


def _dimensionless_mass_ratio_per_time_factor(u_norm: str):
    """Return a 1/h conversion factor for a mass-ratio rate coefficient.

    A unit such as ``g/µg/h`` is dimensionally a reciprocal time because the
    two mass units cancel.  It occurs in reported PFO parameters and should
    convert to ``1/h`` rather than be sent to review as a dimensional mismatch.
    """
    num_tokens, den_tokens = parse_unit_tokens(u_norm)
    if len(num_tokens) != 1 or len(den_tokens) != 2:
        return None

    numerator_factor = mass_token_to_g(num_tokens[0])
    if numerator_factor is None:
        return None

    denominator_mass = None
    denominator_time = None
    for token in den_tokens:
        if denominator_mass is None:
            denominator_mass = mass_token_to_g(token)
            if denominator_mass is not None:
                continue
        if denominator_time is None:
            denominator_time = time_token_to_h(token)
            if denominator_time is not None:
                continue
        return None

    if denominator_mass is None or denominator_time is None:
        return None
    return numerator_factor / denominator_mass / denominator_time


# ---------------- Freundlich helpers ----------------

FREUNDLICH_SYMBOL_RE = re.compile(
    r"^(?P<inverse>1/)?(?P<symbol>[a-z\u0370-\u03ff]+)$",
    flags=re.I,
)


def _freundlich_equation_form_from_token(raw) -> Optional[str]:
    """Map a symbolic Ce exponent to the canonical ``n``/``1/n`` form.

    Literature frequently substitutes a different symbol for ``n`` (for
    example ``m`` or ``β``).  The symbol itself does not change the equation:
    a leading ``1/`` means the reported parameter must be inverted to obtain
    the exponent applied to Ce; otherwise the reported parameter is the Ce
    exponent directly.
    """
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass

    token = unicodedata.normalize("NFKC", str(raw)).strip().lower()
    token = token.replace(" ", "").strip("()[]{}")
    if token in {"1n", "oneovern"}:
        token = "1/n"
    match = FREUNDLICH_SYMBOL_RE.fullmatch(token)
    if not match:
        return None
    return "1/n" if match.group("inverse") else "n"


def _freundlich_kf_unit_equation_form(unit) -> tuple[Optional[str], Optional[str]]:
    """Return the canonical form and literal exponent token from a KF unit."""
    if not isinstance(unit, str) or not unit.strip():
        return None, None
    match = re.search(r"\^\s*\(?\s*(?P<token>[^)\s]+)\s*\)?\s*$", unit)
    if not match:
        return None, None
    token = match.group("token")
    return _freundlich_equation_form_from_token(token), token


def _parse_exponent_value(raw: str):
    if not isinstance(raw, str) or not raw.strip():
        return None, None
    s = raw.strip().lower().replace(" ", "")
    m = re.match(r"^(?P<token>(?:1/)?[a-z\u0370-\u03ff]+)_(?P<value>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$", s)
    if not m:
        return None, None
    form = _freundlich_equation_form_from_token(m.group("token"))
    if form is None:
        return None, None
    return form, float(m.group("value"))


def _get_eq_form_symbol(row):
    eq_text = row.get("Freundlich_equation_form", "")
    return _freundlich_equation_form_from_token(eq_text)


def _resolve_freundlich_exponent_value(row) -> tuple[Optional[float], Optional[str]]:
    """Return the numeric exponent actually applied to Ce in the fit equation."""
    exponent_sources = ("Freundlich_n_or_1/n_verbatim",)
    reported_form, val, source_col = None, None, None
    for col in exponent_sources:
        reported_form, val = _parse_exponent_value(row.get(col, None))
        if reported_form is not None and val is not None:
            source_col = col
            break
    if val is None:
        return None, None

    equation_form = _get_eq_form_symbol(row)
    unit_form, unit_token = _freundlich_kf_unit_equation_form(
        row.get("Freundlich_KF_unit", "")
    )
    form = equation_form or unit_form or reported_form
    # ``1/n_0.4`` reports the applied exponent itself, whereas ``n_2.5``
    # reports its reciprocal parameter when the equation uses ``1/n`` (or an
    # equivalent spelling such as ``1/m``).  Preserve that distinction even
    # when the equation form is inferred from the KF unit.
    if reported_form == "1/n":
        source = source_col
        if equation_form:
            source = f"{source_col} (equation form={equation_form})"
        elif unit_form:
            source = f"{source_col} (KF-unit exponent={unit_token})"
        return val, source
    if form == "1/n":
        if val == 0:
            return None, None
        source = source_col
        if equation_form:
            source = f"{source_col} (equation form={equation_form})"
        elif unit_form:
            source = f"{source_col} (KF-unit exponent={unit_token}; inverted to 1/n)"
        elif reported_form == "n":
            source = f"{source_col} (inverted n→1/n)"
        return 1.0 / val, source
    if form == "n":
        source = source_col
        if equation_form:
            source = f"{source_col} (equation form={equation_form})"
        elif unit_form:
            source = f"{source_col} (KF-unit exponent={unit_token})"
        return val, source
    return None, None


def _resolve_freundlich_form_review(
    df: pd.DataFrame,
    idx,
    row,
    unit: str,
    one_over_n: Optional[float],
) -> None:
    """Replace a stale blank-form review flag when the KF unit resolves it."""
    form, unit_token = _freundlich_kf_unit_equation_form(unit)
    if form is None or one_over_n is None or one_over_n <= 0:
        return
    if _get_eq_form_symbol(row) is not None:
        return

    if "Freundlich_equation_form" not in df.columns:
        df["Freundlich_equation_form"] = ""
    df.at[idx, "Freundlich_equation_form"] = form

    trace = (
        "Freundlich_equation_form inferred from "
        f"Freundlich_KF_unit exponent {unit_token!r} as {form}"
    )
    if "normalization_trace" not in df.columns:
        df["normalization_trace"] = ""
    existing_trace = str(df.at[idx, "normalization_trace"] or "").strip("; ")
    if trace not in existing_trace:
        df.at[idx, "normalization_trace"] = "; ".join(
            part for part in (existing_trace, trace) if part
        )

    review_reason = str(row.get("review_reason", "") or "")
    if (
        "Freundlich_equation_form left blank" in review_reason
        and "no readable KF-unit evidence" in review_reason
    ):
        df.at[idx, "review_reason"] = (
            "Freundlich_equation_form filled by post-processing safety guard "
            "from Freundlich_KF_unit: "
            f"KF-unit evidence ({form}={one_over_n:.12g})"
        )


def annotate_freundlich_exponent_flags(df: pd.DataFrame) -> None:
    """Flag superlinear Freundlich fits without treating them as invalid."""
    flags: list[bool] = []
    for _idx, row in df.iterrows():
        one_over_n, _source = _resolve_freundlich_exponent_value(row)
        if one_over_n is None:
            one_over_n = _coerce_number(row.get("Freundlich_exponent_value"))
        flags.append(bool(one_over_n is not None and one_over_n > 1.0))
    df[FREUNDLICH_EXPONENT_GREATER_THAN_ONE_COLUMN] = pd.Series(
        flags,
        index=df.index,
        dtype="bool",
    )


def _kf_unit_token_key(token: str) -> str:
    text = normalize_unit(token or "")
    text = text.strip().replace(" ", "").replace("_", "").lower()
    text = text.replace("\u00c2", "")
    text = text.replace("\u03bc", "u").replace("\u00b5", "u").replace("\u013e", "u")
    if text.startswith("lg/"):
        text = "ug/" + text[3:]
    return text


def _kf_amount_token_requires_mw(token: str) -> bool:
    key = _kf_unit_token_key(token)
    return re.match(r"^(mol|mmol|umol|eq|meq|ueq)/", key) is not None


def _kf_is_mass_loading_token(token: str) -> bool:
    key = _kf_unit_token_key(token)
    return re.match(r"^(ug|ng|mg|g|mol|mmol|umol)/(g|kg|mg)$", key) is not None


def _kf_is_concentration_token(token: str) -> bool:
    key = _kf_unit_token_key(token)
    return re.match(r"^(ug|ng|mg|g|mol|mmol|umol)/(l|ml)$", key) is not None


def _kf_is_equivalent_ratio_token(token: str) -> bool:
    key = _kf_unit_token_key(token)
    return re.match(r"^(eq|meq|ueq)/(eq|meq|ueq)$", key) is not None


def _kf_is_equivalent_concentration_token(token: str) -> bool:
    key = _kf_unit_token_key(token)
    return re.match(r"^(eq|meq|ueq)/(l|ml)$", key) is not None


def _kf_is_distribution_like_unit(unit_norm: str) -> bool:
    key = _kf_unit_token_key(unit_norm)
    return re.fullmatch(
        r"\(?(?:l|ml)/(?:g|kg|mg)\)?(?:\^\(?[-+a-z0-9/.]+\)?)?",
        key,
    ) is not None


def _mass_per_mass_to_mg_per_g(token: str, mw: float):
    token = _kf_unit_token_key(token)
    m = re.match(r"^(ug|ng|mg|g|mol|mmol|umol)/(g|kg|mg)$", token)
    if not m:
        return None
    num, den = m.group(1), m.group(2)
    if num == "mg":
        mg_num = 1.0
    elif num == "g":
        mg_num = 1000.0
    elif num == "ug":
        mg_num = 1e-3
    elif num == "ng":
        mg_num = 1e-6
    elif num in ("mol", "mmol", "umol"):
        if mw is None:
            return None
        if num == "mol":
            mg_num = mw * 1000.0
        elif num == "mmol":
            mg_num = mw
        else:
            mg_num = mw / 1000.0
    else:
        return None
    if den == "g":
        per_g = 1.0
    elif den == "kg":
        per_g = 1 / 1000.0
    else:
        per_g = 1000.0
    return mg_num * per_g


def _conc_to_mg_per_L(token: str, mw: float):
    token = _kf_unit_token_key(token)
    m = re.match(r"^(ug|ng|mg|g|mol|mmol|umol)/(l|ml)$", token)
    if not m:
        return None
    num, den = m.group(1), m.group(2)
    if num == "mg":
        mg_num = 1.0
    elif num == "g":
        mg_num = 1000.0
    elif num == "ug":
        mg_num = 1e-3
    elif num == "ng":
        mg_num = 1e-6
    elif num in ("mol", "mmol", "umol"):
        if mw is None:
            return None
        if num == "mol":
            mg_num = mw * 1000.0
        elif num == "mmol":
            mg_num = mw
        else:
            mg_num = mw / 1000.0
    else:
        return None
    den_norm = den.lower()
    per_L = 1.0 if den_norm == "l" else 1000.0
    return mg_num * per_L

def _fixed_volume_conc_to_mg_per_L(value_raw, unit_norm: str):
    """
    Convert units such as g/100mL, mg/400mL, or mg/45_mL to mg/L.
    The generic dimension parser treats the numeric denominator as part of
    the volume token, so handle this common extraction shape before fallback.
    """
    num_val = _coerce_number(value_raw)
    if num_val is None or not unit_norm:
        return None

    u = normalize_unit(unit_norm).replace("_", "").lower()
    u = u.replace("\u03bc", "\u00b5").replace("ug", "\u00b5g").replace("ul", "\u00b5l")
    m = re.match(
        r"^(?P<mass>ng|\u00b5g|mg|g|kg)/(?P<vol_num>\d+(?:\.\d+)?|)(?P<vol>\u00b5l|ml|l)$",
        u,
    )
    if not m:
        return None

    mass_factor = {"ng": 1e-6, "\u00b5g": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}.get(m.group("mass"))
    vol_factor = {"\u00b5l": 1e-6, "ml": 1e-3, "l": 1.0}.get(m.group("vol"))
    if mass_factor is None or vol_factor is None:
        return None

    vol_num = float(m.group("vol_num") or 1.0)
    if vol_num <= 0:
        return None
    return num_val * mass_factor / (vol_num * vol_factor)

def convert_freundlich_kf(v, u_norm: str, mw: float, one_over_n: float, target="(mg/g)/(mg/L)^(1/n)"):
    num = _coerce_number(v)
    if num is None:
        return "", "Freundlich_KF unrecognized numeric format"
    if not u_norm:
        return "", "Freundlich_KF unit missing"
    if one_over_n is None:
        return "", "Freundlich exponent missing: cannot convert KF"

    if _kf_is_distribution_like_unit(u_norm):
        return "", "Freundlich_KF unit is distribution-coefficient-like and lacks separate uptake/concentration terms; conversion skipped"

    try:
        m_prod = KF_PROD_RE.match(u_norm)
    except re.error:
        m_prod = None

    if m_prod:
        A, B, exp_str = m_prod.group("A"), m_prod.group("B"), m_prod.group("exp")
        neg = isinstance(exp_str, str) and exp_str.startswith("-")
        if neg:
            exp_str = exp_str[1:]
        try:
            m_vol_first = re.match(r"^(?P<vol>[lL]|m[Ll])/(?P<amt>.+)$", B)
        except re.error:
            m_vol_first = None
        if m_vol_first:
            vol = m_vol_first.group("vol")
            amt = m_vol_first.group("amt")
            u_norm = f"({A})/({amt}/{vol})^({exp_str})"
        else:
            u_norm = f"({A})/({B})^({exp_str})"

    if _kf_is_distribution_like_unit(u_norm):
        return "", "Freundlich_KF unit is distribution-coefficient-like and lacks separate uptake/concentration terms; conversion skipped"

    try:
        m = KF_DIV_RE.match(u_norm)
    except re.error:
        m = None
    if not m:
        return "", f"Unrecognized Freundlich_KF unit: {u_norm}"
    A, B, exp_str = m.group("A"), m.group("B"), m.group("exp")

    if isinstance(exp_str, str) and exp_str.startswith("-"):
        exp_str = exp_str[1:]

    fA = _mass_per_mass_to_mg_per_g(A, mw)
    fB = _conc_to_mg_per_L(B, mw)
    if fA is None or fB is None:
        if mw is None and (_kf_amount_token_requires_mw(A) or _kf_amount_token_requires_mw(B)):
            return "", "PFAS molecular weight is missing; cannot convert amount-based KF units"
        if _kf_is_equivalent_ratio_token(A) or _kf_is_equivalent_concentration_token(B):
            return "", "KF unit uses equivalent-based capacity/concentration terms that cannot be converted to mg/g without an adsorbent-mass basis"
        if _kf_is_concentration_token(A) and _kf_is_mass_loading_token(B):
            return "", "KF unit appears reversed (concentration term reported before uptake term); conversion skipped"
        if _kf_is_concentration_token(A) and _kf_is_concentration_token(B):
            return "", "KF unit lacks uptake-per-adsorbent-mass term; conversion skipped"
        return "", f"KF unit parse failed: A='{A}', B='{B}'"

    try:
        factor = fA / (fB ** float(one_over_n))
    except Exception:
        return "", "Invalid Freundlich exponent value"

    return num * factor, ""


# ---------------- Special legacy pieces (IEC, KL edge cases) ----------------

def _iec_unit_key(u: str) -> str:
    text = str(u or "").strip().lower()
    text = text.replace("\u03bc", "\u00b5").replace("\u013e", "\u00b5")
    text = text.replace("umol", "\u00b5mol").replace("ueq", "\u00b5eq")
    return text


def _fixed_mass_iec_to_meq_per_g(value_raw, unit_norm: str) -> float | None:
    """Convert IEC units with an explicit mass denominator to meq/g.

    This mirrors the fixed-volume concentration handling used for expressions
    such as mg/100mL.  The existing IEC convention treats mmol as meq when no
    ion valence is available, so the same convention is retained here.
    """
    value = _as_float_or_blank(value_raw)
    if value is None or not unit_norm:
        return None

    match = re.fullmatch(
        r"(?P<amount>meq|eq|equiv|mmol|\u00b5mol)/"
        r"(?P<denominator>\d+(?:\.\d+)?)(?P<mass>g|kg|mg)",
        _iec_unit_key(unit_norm),
    )
    if not match:
        return None

    amount_factor_meq = {
        "meq": 1.0,
        "eq": 1000.0,
        "equiv": 1000.0,
        "mmol": 1.0,
        "\u00b5mol": 0.001,
    }[match.group("amount")]
    mass_factor_g = {"g": 1.0, "kg": 1000.0, "mg": 0.001}[match.group("mass")]
    denominator_g = float(match.group("denominator")) * mass_factor_g
    if denominator_g <= 0:
        return None
    return value * amount_factor_meq / denominator_g


def _iec_value_unit_pairs(value_raw, unit_norm: str) -> tuple[list[tuple[object, str]], str]:
    """Return aligned IEC value/unit pairs from scalar or ``&&`` list cells."""
    raw_value = str(value_raw or "")
    raw_unit = str(unit_norm or "")
    if "&&" not in raw_value and "&&" not in raw_unit:
        return [(value_raw, unit_norm)], ""

    values = [part.strip() for part in raw_value.split("&&")]
    units = [part.strip() for part in raw_unit.split("&&")]
    if len(values) != len(units) or any(not value or not unit for value, unit in zip(values, units)):
        return [], "IEC value/unit list length mismatch or blank pair"
    return list(zip(values, units)), ""


def _iec_output_value(values: list[float]):
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "&&".join(f"{value:.12g}" for value in values)


def convert_iec(v, u, mw, target="meq/g"):
    num = _as_float_or_blank(v)
    if num is None or not u:
        return "", ""
    ul = _iec_unit_key(u)
    if re.search(r"/(l|ml)$", ul):
        return "", ""
    fixed_mass_value = _fixed_mass_iec_to_meq_per_g(v, ul)
    if fixed_mass_value is not None:
        return fixed_mass_value, f"converted {u} to meq/g using explicit mass denominator"
    if ul == "meq/g":
        return num, ""
    if ul == "eq/kg":
        return num, ""
    if ul == "mmol/g":
        return num, ""
    if ul == "\u00b5mol/mg":
        return num, ""
    if ul == "\u00b5mol/m2":
        return "", "Surface-area basis; not converted"
    if u == "µmol/mg":
        return num, ""
    if u == "µmol/m2":
        return "", "Surface-area basis; not converted"
    return "", f"Unrecognized IEC unit (normalized): {u}"


def convert_iec_solution(v, u, mw, target="meq/L"):
    num = _as_float_or_blank(v)
    if num is None or not u:
        return "", ""
    ul = _iec_unit_key(u)

    if ul == "meq/l":
        return num, ""
    if ul in {"eq/l", "equiv/l"}:
        return num * 1000.0, ""
    if ul == "mmol/l":
        return num, ""
    if ul == "\u00b5mol/l":
        return num / 1000.0, ""
    if "µmol/l" in ul or "umol/l" in ul:
        return num / 1000.0, ""
    if ul == "mol/l":
        return num * 1000.0, ""

    if ul == "meq/ml":
        return num * 1000.0, ""
    if ul in {"eq/ml", "equiv/ml"}:
        return num * 1_000_000.0, ""
    if ul == "mmol/ml":
        return num * 1000.0, ""

    if "/m2" in ul:
        return "", "Surface-area basis; not converted"

    return "", f"Unrecognized IEC solution unit (normalized): {u}"


def _iec_unit_basis(u: str) -> str:
    ul = _iec_unit_key(u)
    if re.search(r"/(?:l|ml)$", ul):
        return "solution"
    if re.search(r"/(?:g|kg|mg)$", ul):
        return "mass"
    if "/m2" in ul:
        return "surface"
    return "unknown"


def kl_first_power_only(u):
    if "^" in u:
        return False, f"Langmuir_KL has exponent in unit [{u}]; conversion skipped"
    if u.startswith("(") and u.endswith(")") and u.count("(") == 1:
        return True, u[1:-1]
    return True, u


def _ssa_num_g(row):
    col = "ssa_(m2/g)"
    if col not in row.index:
        return None
    val = row[col]
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    if not isinstance(val, str) or not val.strip():
        return None

    s = val.strip()
    if "," not in s and ";" not in s:
        return _coerce_number(s)

    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    nums = []
    for p in parts:
        v = _coerce_number(p)
        if v is not None:
            nums.append(v)
    if not nums:
        return None
    return sum(nums) / len(nums)


def _apply_ssa_surface_to_mass(value_raw, unit_norm, ssa):
    if not unit_norm or "/m2" not in unit_norm:
        return None, None
    v = _coerce_number(value_raw)
    if v is None or ssa is None:
        return None, None
    return v * ssa, unit_norm.replace("/m2", "/g")


def materialize_ssa_avg(df: pd.DataFrame) -> None:
    src = "ssa_(m2/g)"
    out = "ssa_(m2/g)_avg"
    if src in df.columns and out not in df.columns:
        df[out] = df.apply(_ssa_num_g, axis=1).fillna("")


def _adsorbent_ssa_for_conversion(row: pd.Series) -> float | None:
    values = parse_numeric_values(row.get("ssa_(m2/g)"))
    if not values:
        return None
    return sum(values) / len(values)


def _standardize_adsorbent_numeric_values(
    row: pd.Series,
    prop: NumericProperty,
) -> tuple[list[float], str]:
    """Apply this module's shared unit conversions to one adsorbent property."""
    values = parse_numeric_values(row.get(prop.source_col))
    if not values:
        return [], ""
    if prop.converter == "identity":
        return values, ""

    raw_unit = row.get(prop.unit_col, "") if prop.unit_col else ""
    unit = normalize_unit(raw_unit) if not _blank(raw_unit) else ""

    if prop.converter == "grain_size":
        converted, note = standardize_grain_size_values(
            row.get(prop.source_col), raw_unit
        )
        prefix = f"{prop.source_col}: "
        return converted, prefix + note if note else ""

    if prop.converter == "pore_volume":
        factor = _pore_volume_factor_to_cm3_per_g(raw_unit)
        if factor is None:
            return [], (
                f"{prop.source_col}: unit {unit or '<blank>'!r} "
                "not converted to cm3/g"
            )
        return [value * factor for value in values], ""

    if not unit and prop.specific_exchange_capacity:
        unit = "meq/g"

    if prop.converter == "exchange_mass":
        basis = _iec_unit_basis(unit)
        if basis == "solution":
            return [], ""
        if basis == "surface":
            ssa = _adsorbent_ssa_for_conversion(row)
            if ssa is None:
                return [], (
                    f"{prop.source_col}: surface-area basis requires "
                    "ssa_(m2/g)"
                )
            unit_key = _iec_unit_key(unit)
            if unit_key in {"µmol/m2", "umol/m2"}:
                return [value * ssa / 1000.0 for value in values], (
                    f"{prop.source_col}: converted µmol/m2 to meq/g using "
                    f"ssa_(m2/g)={ssa:.12g}"
                )
            if unit_key == "meq/m2":
                return [value * ssa for value in values], (
                    f"{prop.source_col}: converted meq/m2 to meq/g using "
                    f"ssa_(m2/g)={ssa:.12g}"
                )
            return [], (
                f"{prop.source_col}: surface-area unit {unit!r} "
                "not converted to meq/g"
            )

        converted: list[float] = []
        notes: list[str] = []
        for value in values:
            result, note = convert_iec(value, unit, None, "meq/g")
            if result != "":
                converted.append(float(result))
            if note:
                notes.append(note)
        if converted:
            return converted, "; ".join(dict.fromkeys(notes))
        if unit:
            return [], (
                f"{prop.source_col}: unit {unit!r} is not mass-based meq/g"
            )
        return [], f"{prop.source_col}: blank exchange-capacity unit"

    if prop.converter == "exchange_volume":
        basis = _iec_unit_basis(unit)
        if basis in {"mass", "surface"} or (
            not unit and prop.specific_exchange_capacity
        ):
            return [], ""
        converted = []
        notes = []
        for value in values:
            result, note = convert_iec_solution(value, unit, None, "meq/L")
            if result != "":
                converted.append(float(result))
            if note:
                notes.append(note)
        if converted:
            return converted, "; ".join(dict.fromkeys(notes))
        if unit:
            return [], f"{prop.source_col}: unit {unit!r} not converted to meq/L"
        return [], f"{prop.source_col}: blank exchange-capacity unit"

    raise ValueError(f"Unknown adsorbent numeric converter: {prop.converter}")



# ---------------- Load, convert, save ----------------

def _numeric_present(v) -> bool:
    return _coerce_number(v) is not None


def _extract_v0_amount_mass_tokens(u_raw: str):
    u = normalize_unit(u_raw)
    if not u:
        return None, None
    num, den = parse_unit_tokens(u)
    amt_tok = next((t for t in num if classify_token(t) in {"amount", "mass"}), None)
    mass_tok = next((t for t in den if classify_token(t) == "mass"), None)
    return amt_tok, mass_tok


def _extract_k2_amount_mass_tokens(u_raw: str):
    u = normalize_unit(u_raw)
    if not u:
        return None, None
    num, den = parse_unit_tokens(u)
    mass_tok = next((t for t in num if classify_token(t) == "mass"), None)
    amt_tok = next((t for t in den if classify_token(t) in {"amount", "mass"}), None)
    return amt_tok, mass_tok


def _has_val(v) -> bool:
    return not _blank(v)


def _has_unit(u) -> bool:
    if not isinstance(u, str):
        return False
    return normalize_unit(u) != ""

def _invert_kl_to_ce_unit(kl_unit_raw: str) -> str:
    u = normalize_unit(kl_unit_raw)
    if not u:
        return ""
    if u.startswith("(") and u.endswith(")") and u.count("(") == 1:
        u = u[1:-1]
    m = re.match(r"^(?:L|mL)/(.+)$", u)
    if not m:
        return ""
    amt = m.group(1)
    vol = "L" if u.startswith("L/") else "mL"
    return f"{amt}/{vol}"


def infer_kf_unit_from_langmuir(df_in: pd.DataFrame) -> None:
    kf_v, kf_u = "Freundlich_KF_value", "Freundlich_KF_unit"
    qm_u = "Langmuir_Qm_unit"
    kl_u = "Langmuir_KL_unit"

    for col in (kf_u,):
        if col in df_in.columns:
            df_in[col] = df_in[col].astype("object")
    for dst_col in ADSORBENT_PROPERTY_COLUMN_MAP.values():
        if dst_col in df_in.columns:
            df_in[dst_col] = df_in[dst_col].astype("object")

    _ensure_columns(df_in, ["normalization_trace"], "")

    for idx, row in df_in.iterrows():
        if not _numeric_present(row.get(kf_v)):
            continue
        if _has_unit(row.get(kf_u)):
            continue
        qm_unit_norm = normalize_unit(row.get(qm_u, ""))
        kl_unit_norm = normalize_unit(row.get(kl_u, ""))
        if not qm_unit_norm or not kl_unit_norm:
            continue

        qe_unit = qm_unit_norm
        ce_unit = _invert_kl_to_ce_unit(kl_unit_norm)
        if not ce_unit:
            continue
        form = _get_eq_form_symbol(row)
        if not form:
            continue
        derived = f"({qe_unit})/({ce_unit})^({form})"

        df_in.at[idx, kf_u] = derived
        df_in.at[idx, "normalization_trace"] = (
            str(df_in.at[idx, "normalization_trace"]).rstrip("; ")
            + f"; inferred Freundlich_KF_unit={derived} from Langmuir_Qm_unit={qm_unit_norm}, Langmuir_KL_unit={kl_unit_norm}"
        ).strip("; ")


def infer_pso_units(df_in: pd.DataFrame) -> None:
    k2_v, k2_u = "PSO_k2_value", "PSO_k2_unit"
    v0_v, v0_u = "PSO_v0_value", "PSO_v0_unit"
    qe_v, qe_u = "PSO_Qe_value", "PSO_Qe_unit"

    for col in (k2_u, v0_u, qe_u):
        if col in df_in.columns:
            df_in[col] = df_in[col].astype("object")

    _ensure_columns(df_in, ["normalization_trace"], "")

    for idx, row in df_in.iterrows():
        has_k2 = _numeric_present(row.get(k2_v)) and _has_unit(row.get(k2_u))
        has_v0 = _numeric_present(row.get(v0_v)) and _has_unit(row.get(v0_u))
        has_qe = _numeric_present(row.get(qe_v)) and _has_unit(row.get(qe_u))

        if _numeric_present(row.get(qe_v)) and not _has_unit(row.get(qe_u)) and has_k2 and has_v0:
            v0_amt, v0_mass = _extract_v0_amount_mass_tokens(row.get(v0_u, ""))
            derived_qe = None
            if v0_amt and v0_mass:
                derived_qe = f"{v0_amt}/{v0_mass}"
                source = f"from {v0_u}={normalize_unit(row.get(v0_u, ''))}"
            else:
                k2_amt, k2_mass = _extract_k2_amount_mass_tokens(row.get(k2_u, ""))
                if k2_amt and k2_mass:
                    derived_qe = f"{k2_amt}/{k2_mass}"
                    source = f"from {k2_u}={normalize_unit(row.get(k2_u, ''))}"
            if not derived_qe:
                derived_qe = "mg/g"
                source = "fallback to mg/g (tokens not parsed)"
            df_in.at[idx, qe_u] = derived_qe
            df_in.at[idx, "normalization_trace"] = (
                str(df_in.at[idx, "normalization_trace"]).rstrip("; ")
                + f"; inferred PSO_Qe_unit={derived_qe} via v0=k2·Qe^2 ({source})"
            ).strip("; ")
        elif _has_val(row.get(k2_v)) and not _has_unit(row.get(k2_u)) and has_v0 and has_qe:
            df_in[k2_u] = df_in[k2_u].astype(str)
            df_in.at[idx, k2_u] = "g/(mg·h)"
            df_in.at[idx, "normalization_trace"] = (
                str(df_in.at[idx, "normalization_trace"]).rstrip("; ")
                + "; inferred PSO_k2_unit=g/(mg·h) via v0=k2·Qe^2"
            ).strip("; ")

        elif _has_val(row.get(v0_v)) and not _has_unit(row.get(v0_u)) and has_k2 and has_qe:
            df_in[v0_u] = df_in[v0_u].astype(str)
            df_in.at[idx, v0_u] = "mg/(g·h)"
            df_in.at[idx, "normalization_trace"] = (
                str(df_in.at[idx, "normalization_trace"]).rstrip("; ")
                + "; inferred PSO_v0_unit=mg/(g·h) via v0=k2·Qe^2"
            ).strip("; ")


def standardize_unit_parameters(df: pd.DataFrame) -> None:
    """Apply parameter unit conversions in the historical iteration order."""
    annotate_freundlich_exponent_flags(df)
    for pname, spec in PARAM_SPECS.items():
        val_col = spec["value_col"]
        unit_col = spec["unit_col"]
        target_unit = spec["target"]

        if val_col not in df.columns or unit_col not in df.columns:
            continue

        unit_norm_series = df[unit_col].apply(lambda x: normalize_unit(x) if isinstance(x, str) else "")
        out_vals, notes = [], []

        for idx, row in df.iterrows():
            raw_val = row[val_col]
            if _blank(raw_val):
                out_vals.append("")
                notes.append("")
                continue

            needs_mw = pname in (
                "Langmuir_Qm", "Qe", "PFO_Qe", "PSO_Qe", "PFAS_C0",
                "Langmuir_KL", "Kd", "PSO_k2", "PSO_v0", "Freundlich_KF"
            )
            mw = get_pfasmw(row) if needs_mw else None
            u_norm = unit_norm_series.iloc[idx]
            raw_unit = row.get(unit_col, "")

            # Initial PFAS_C0 logging hook
            # (now routed through _debug_pfas_c0_row so that study-level
            # filtering via DEBUG_PFAS_C0_STUDIES applies uniformly)
            if pname == "PFAS_C0":
                _debug_pfas_c0_row(
                    idx,
                    row,
                    raw_val,
                    row.get(unit_col, ""),
                    u_norm,
                    mw,
                    stage="initial",
                    extra=""
                )

            if pname == "Freundlich_KF":
                ssa = _ssa_num_g(row)
                if u_norm and "/m2" in u_norm and ssa is not None:
                    new_v, new_u = _apply_ssa_surface_to_mass(raw_val, u_norm, ssa)
                    if new_v is not None and new_u is not None:
                        raw_val = new_v
                        u_norm = new_u
                        df.at[idx, "normalization_trace"] = (
                            str(df.at[idx, "normalization_trace"]).rstrip("; ")
                            + "; SSA applied to KF: converted A '*/m2' → '*/g' using ssa_(m2/g)"
                        ).strip("; ")
                one_over_n, src = _resolve_freundlich_exponent_value(row)
                _resolve_freundlich_form_review(
                    df, idx, row, u_norm, one_over_n
                )
                exp_out_col = "Freundlich_exponent_value"
                if exp_out_col not in df.columns:
                    df[exp_out_col] = ""
                df[exp_out_col] = df[exp_out_col].astype("object")
                if one_over_n is not None and one_over_n <= 0:
                    df.at[idx, exp_out_col] = one_over_n
                    out_vals.append("")
                    notes.append(
                        f"Freundlich_KF [{u_norm}→{target_unit}] exponent={one_over_n:g} must be > 0; conversion skipped"
                    )
                    continue

                v_new, note = convert_freundlich_kf(raw_val, u_norm, mw, one_over_n, target_unit)
                if one_over_n is not None:
                    df.at[idx, exp_out_col] = one_over_n
                out_vals.append(v_new if v_new != "" else "")
                if note:
                    notes.append(f"Freundlich_KF [{u_norm}→{target_unit}] {note}")
                else:
                    src_info = f" using exponent={one_over_n:g}" if one_over_n is not None else " (exponent missing)"
                    src_hint = f" from {src}" if src else ""
                    notes.append(f"Freundlich_KF [{u_norm}→{target_unit}]{src_info}{src_hint}".strip())
                continue

            if pname == "ion_exchange_capacity":
                pairs, pair_error = _iec_value_unit_pairs(raw_val, u_norm)
                mass_values: list[float] = []
                solution_values: list[float] = []
                row_notes: list[str] = []

                if pair_error:
                    row_notes.append(pair_error)
                for pair_value, pair_unit in pairs:
                    basis = _iec_unit_basis(pair_unit)
                    if basis == "solution":
                        v_soln, note_soln = convert_iec_solution(
                            pair_value, pair_unit, mw, "meq/L"
                        )
                        if v_soln != "":
                            solution_values.append(float(v_soln))
                        if note_soln:
                            row_notes.append(note_soln)
                    elif basis in {"mass", "surface"}:
                        v_mass, note_mass = convert_iec(
                            pair_value, pair_unit, mw, "meq/g"
                        )
                        if v_mass != "":
                            mass_values.append(float(v_mass))
                        if note_mass:
                            row_notes.append(note_mass)
                    else:
                        v_mass, note_mass = convert_iec(
                            pair_value, pair_unit, mw, "meq/g"
                        )
                        if v_mass != "":
                            mass_values.append(float(v_mass))
                        if note_mass:
                            row_notes.append(note_mass)
                        elif v_mass == "":
                            v_soln, note_soln = convert_iec_solution(
                                pair_value, pair_unit, mw, "meq/L"
                            )
                            if v_soln != "":
                                solution_values.append(float(v_soln))
                            if note_soln:
                                row_notes.append(note_soln)
                if f"{val_col}_meq/g" not in df.columns:
                    df[f"{val_col}_meq/g"] = ""
                if f"{val_col}_meq/L" not in df.columns:
                    df[f"{val_col}_meq/L"] = ""
                for out_col in (f"{val_col}_meq/g", f"{val_col}_meq/L"):
                    df[out_col] = df[out_col].astype("object")
                if mass_values:
                    df.at[idx, f"{val_col}_meq/g"] = _iec_output_value(mass_values)
                if solution_values:
                    df.at[idx, f"{val_col}_meq/L"] = _iec_output_value(solution_values)
                row_note = "; ".join(dict.fromkeys(row_notes))
                out_vals.append("")
                notes.append(f"ion_exchange_capacity [{u_norm}] {row_note}" if row_note else "")
                continue

            if pname in ("PFAS_C0", "Adsorbent_dosage"):
                vol_L = get_solution_volume_L(row)

                # (1) Mass-only C0/dosage (e.g., "mg" of PFAS, separate solution volume)
                if u_norm in {"ng", "µg", "ug", "mg", "g", "kg"}:
                    if pname == "PFAS_C0":
                        _debug_pfas_c0_row(
                            idx, row, raw_val, raw_unit, u_norm, mw,
                            stage="mass-only-branch",
                            extra=f"vol_L={vol_L!r}"
                        )
                    if not vol_L:
                        out_vals.append("")
                        notes.append(f"{pname} [{u_norm}] Solution volume missing; mass-only value not converted")
                        if pname == "PFAS_C0":
                            _debug_pfas_c0_row(
                                idx, row, raw_val, raw_unit, u_norm, mw,
                                stage="mass-only-no-volume",
                                extra="Solution_volume_(mL) missing or non-numeric"
                            )
                        continue
                    num = _coerce_number(raw_val)
                    if num is None:
                        out_vals.append("")
                        notes.append("")
                        continue
                    tok = "µg" if u_norm == "ug" else u_norm
                    mg_factor = {"ng": 1e-6, "µg": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}.get(tok)
                    if mg_factor is None:
                        out_vals.append("")
                        notes.append(f"{pname} [{u_norm}] Unrecognized bare-mass token")
                        if pname == "PFAS_C0":
                            _debug_pfas_c0_row(
                                idx, row, raw_val, raw_unit, u_norm, mw,
                                stage="mass-only-unrecognized",
                                extra=f"tok={tok!r}"
                            )
                        continue
                    new_val = num * mg_factor / vol_L
                    out_vals.append(new_val)

                    notes.append("")
                    if pname == "PFAS_C0":
                        _debug_pfas_c0_row(
                            idx, row, raw_val, raw_unit, u_norm, mw,
                            stage="mass-only-converted",
                            extra=f"num={num}, mg_factor={mg_factor}, vol_L={vol_L}, new_val={new_val}"
                        )

                    continue

                fixed_volume_conc = _fixed_volume_conc_to_mg_per_L(raw_val, u_norm)
                if fixed_volume_conc is not None:
                    out_vals.append(fixed_volume_conc)
                    notes.append("")
                    continue

                # (2) Concentration-style C0/dosage (e.g., µM, mM, mol/L, mg/L, µg/L, ppb)
                # Use the dedicated concentration → mg/L helper so MW is **always** respected.
                num = _coerce_number(raw_val)
                if num is not None:
                    conc_factor = _conc_to_mg_per_L(u_norm, mw)
                    if pname == "PFAS_C0":
                        _debug_pfas_c0_row(
                            idx, row, raw_val, raw_unit, u_norm, mw,
                            stage="conc-style",
                            extra=f"conc_factor={conc_factor!r}"
                        )
                    # _conc_to_mg_per_L returns "mg per L" for 1 unit of the given concentration.
                    # Examples:
                    #   µmol/L  → (MW / 1000) mg/L
                    #   mol/L   → (MW * 1000) mg/L
                    #   mg/L    → 1 mg/L
                    #   µg/L    → 1e-3 mg/L
                    if conc_factor is not None:
                        new_val = num * conc_factor
                        out_vals.append(new_val)
                        notes.append("")
                        if pname == "PFAS_C0":
                            suspicious = ""
                            try:
                                # If new_val ~= raw_val but unit is clearly not mg/L, flag it.
                                if abs(float(new_val) - float(num)) < 1e-12 and u_norm not in {"mg/L", "mg/l"}:
                                    suspicious = " (WARNING: new_val ≈ raw_val but u_norm is not mg/L)"
                            except Exception:
                                pass
                            _debug_pfas_c0_row(
                                idx, row, raw_val, raw_unit, u_norm, mw,
                                stage="conc-style-converted",
                                extra=f"num={num}, conc_factor={conc_factor}, new_val={new_val}{suspicious}"
                            )
                        continue
                    else:
                        if pname == "PFAS_C0":
                            _debug_pfas_c0_row(
                                idx, row, raw_val, raw_unit, u_norm, mw,
                                stage="conc-style-no-factor",
                                extra="conc_factor is None – falling back to generic engine"
                            )

                # For PFAS_C0/Adsorbent_dosage, attempt to convert concentration units explicitly.
                # Many records use molar units like µM or mmol/L; ensure these are converted using PFAS MW.
                if pname == "PFAS_C0":
                    num = _coerce_number(raw_val)
                    if num is not None:
                        mw_val = get_pfasmw(row)
                        # Attempt custom conversion when a PFAS molecular weight is available.
                        # First handle normalized units like 'µmol/L', 'mmol/L', 'mol/L', 'nmol/L'.
                        # Also catch abbreviated forms without '/L' such as 'µM', 'mM', 'M' that may slip through normalization.
                        u_low = u_norm.lower() if isinstance(u_norm, str) else ""
                        conc_converted = None
                        # Map numerator unit to mg conversion factor using MW when applicable
                        if mw_val is not None:
                            # Recognize pure molarity forms (µm, um, m, etc) without denominator
                            if u_low in {"µm", "um", "μm"}:  # micro metre, not concentration
                                conc_converted = None
                            else:
                                # Determine numerator and denominator tokens
                                # Accept forms with '/'
                                if "/" in u_low:
                                    num_tok, den_tok = u_low.split("/", 1)
                                else:
                                    # If no explicit denominator, infer that M-like tokens imply per litre
                                    num_tok, den_tok = u_low, "l"
                                # Normalize synonyms for micro and milli
                                num_tok = num_tok.replace("μ", "µ")
                                if num_tok == "ug":
                                    num_tok = "µg"
                                # Determine mg factor from the numerator token
                                mg_f = None
                                # Mass based units
                                if num_tok in {"ng", "µg", "mg", "g", "kg"}:
                                    mg_f = {"ng": 1e-6, "µg": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}.get(num_tok)
                                # Amount based units (use MW)
                                elif num_tok in {"mol", "mmol", "µmol", "umol", "nmol"}:
                                    if num_tok == "mol":
                                        mg_f = mw_val * 1000.0
                                    elif num_tok == "mmol":
                                        mg_f = mw_val
                                    elif num_tok in {"µmol", "umol"}:
                                        mg_f = mw_val / 1000.0
                                    elif num_tok == "nmol":
                                        mg_f = mw_val / 1e6
                                # If the unit is simply 'µm' or 'mm' etc, skip custom conversion
                                # Determine denominator volume factor
                                if mg_f is not None:
                                    den_tok = den_tok.replace("µl", "µl").replace("ul", "µl")
                                    vol_f = {"l": 1.0, "ml": 1e-3, "µl": 1e-6}.get(den_tok, None)
                                    if vol_f is not None and vol_f > 0:
                                        conc_converted = num * mg_f / vol_f
                        # If custom conversion succeeded, use it and skip the generic engine
                        if conc_converted is not None:
                            out_vals.append(conc_converted)
                            notes.append("")
                            if pname == "PFAS_C0":
                                _debug_pfas_c0_row(
                                    idx, row, raw_val, raw_unit, u_norm, mw,
                                    stage="custom-converted",
                                    extra=f"num={num}, mg_factor={mg_f if 'mg_f' in locals() else None}, "
                                          f"vol_factor={vol_f if 'vol_f' in locals() else None}, "
                                          f"conc_converted={conc_converted}"
                                )
                            continue
                        else:
                            if pname == "PFAS_C0":
                                _debug_pfas_c0_row(
                                    idx, row, raw_val, raw_unit, u_norm, mw,
                                    stage="custom-no-conversion",
                                    extra="custom PFAS_C0 concentration handler could not convert; "
                                          "falling back to generic dimension engine"
                                )

            if pname == "Langmuir_KL":
                raw_kl_unit = str(raw_unit).strip() if raw_unit is not None else ""
                raw_kl_unit = raw_kl_unit.replace("\u00b5", "u").replace("\u03bc", "u")
                if re.fullmatch(r"(?:M\s*\^?\s*-?1|1\s*/\s*M)", raw_kl_unit):
                    u_norm = "L/mol"
                ok, fixed = kl_first_power_only(u_norm)
                if not ok:
                    out_vals.append("")
                    notes.append(f"{pname} [{u_norm}] Langmuir_KL has exponent; conversion skipped")
                    continue
                u_norm = fixed

            if pname == "Nominal_grain_size":
                converted_grain_size, grain_size_note = _performance_grain_size_value(
                    raw_val,
                    row.get(unit_col, "") if unit_col in row.index else "",
                )
                out_vals.append(converted_grain_size)
                notes.append(
                    f"{pname} {grain_size_note}" if grain_size_note else ""
                )
                continue

            if pname in ("Langmuir_Qm", "Qe", "PFO_Qe", "PSO_Qe"):
                if u_norm and "/m2" in u_norm:
                    ssa = _ssa_num_g(row)
                    if ssa is not None:
                        new_v, new_u = _apply_ssa_surface_to_mass(raw_val, u_norm, ssa)
                        if new_v is not None and new_u is not None:
                            raw_val = new_v
                            u_norm = new_u
                            unit_norm_series.iloc[idx] = u_norm
                            df.at[idx, "normalization_trace"] = (
                                str(df.at[idx, "normalization_trace"]).rstrip("; ")
                                + f"; SSA applied to {pname}: converted '*/m2' → '*/g' using ssa_(m2/g)"
                            ).strip("; ")

            if not u_norm:
                out_vals.append("")
                notes.append(f"{pname} [] Unit not available")
                continue

            spec_dim = spec["dim"]
            bases = spec["bases"]

            u_fixed = (
                u_norm.replace("mL", "ml").replace("L", "l")
                     .replace("µL", "µl")
                     .replace("ug", "µg").replace("umol", "µmol")
            )

            if u_fixed.count("/") >= 2 and "·" not in u_fixed:
                parts = u_fixed.split("/")
                u_fixed = parts[0] + "/(" + "·".join(parts[1:]) + ")"

            u_fixed = u_fixed.replace("hr", "h").replace("day", "d")

            factor = None
            why = ""
            if pname == "PFO_k1":
                factor = _dimensionless_mass_ratio_per_time_factor(u_fixed)
            if factor is None:
                factor, why = compute_factor_against_signature(
                    u_fixed, spec_dim, bases, mw if needs_mw else None
                )
            num_val = _coerce_number(raw_val)
            if pname == "PFAS_C0":
                _debug_pfas_c0_row(
                    idx, row, raw_val, raw_unit, u_fixed, mw,
                    stage="dimension-engine",
                    extra=(
                        f"spec_dim={spec_dim}, bases={bases}, "
                        f"factor={factor!r}, reason={why!r}, "
                        f"num_val={num_val!r}"
                    )
                )

            if num_val is None:
                out_vals.append("")
                notes.append(f"{pname} [{u_norm}] Non-numeric value")
                continue
            if factor is None:
                out_vals.append("")
                notes.append(f"{pname} [{u_norm}] {why or 'Unrecognized unit'}")
                continue

            out_vals.append(num_val * factor)
            notes.append("")

        if pname != "ion_exchange_capacity":
            if len(out_vals) != len(df):
                out_vals = (out_vals + [""] * len(df))[:len(df)]
                notes = (notes + [""] * len(df))[:len(df)]
            df[f"{val_col}_{target_unit}"] = out_vals

        _append_normalization_notes(df, notes)
