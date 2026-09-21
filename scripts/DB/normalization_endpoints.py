"""Endpoint derivation, final Kd selection, and output-sheet helpers."""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict

import pandas as pd

from adsorbent_consolidation import (
    FUNCTIONAL_GROUP_INDICATOR_COLUMNS,
    PORE_CLASS_INDICATOR_COLUMNS,
)
from normalization_rules import (
    _append_normalization_notes,
    _coerce_number,
    _ensure_columns,
    _reported_value_text,
    ISOTHERM_EXPANDED_INHERITED_KD_FLAG_COLUMN,
    ZETA_POTENTIAL_NEAR_PH_7_COLUMN,
    zeta_potential_near_ph_7,
)

def _fmt_ce_token(x: float) -> str:
    return f"{x:g}"


def compute_kd_benchmarks(df_in: pd.DataFrame,
                          ce_stars_mgL=(0.0001, 0.001, 0.01, 0.1, 1.0)) -> None:
    qm_col = "Langmuir_Qm_value_mg/g"
    kl_col = "Langmuir_KL_value_L/mg"
    kf_col = "Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)"
    on_col = "Freundlich_exponent_value"

    _ensure_columns(df_in, [qm_col, kl_col, kf_col, on_col], "")

    lin_col = "Kd_Langmuir_linear_L/g"
    _ensure_columns(df_in, [lin_col], "")

    for ce in ce_stars_mgL:
        tag = _fmt_ce_token(ce)
        lm_col = f"Kd_Langmuir_at_{tag}_mgL_L/g"
        fr_col = f"Kd_Freundlich_at_{tag}_mgL_L/g"
        _ensure_columns(df_in, [lm_col, fr_col], "")

    errs = [""] * len(df_in)
    for i, row in df_in.iterrows():
        qmax = _coerce_number(row.get(qm_col))
        kl = _coerce_number(row.get(kl_col))
        kf = _coerce_number(row.get(kf_col))
        one_over_n = _coerce_number(row.get(on_col))

        if (qmax is not None) and (kl is not None):
            df_in.at[i, lin_col] = qmax * kl

        invalid_exponent = one_over_n is not None and one_over_n <= 0

        for ce in ce_stars_mgL:
            tag = _fmt_ce_token(ce)
            lm_col = f"Kd_Langmuir_at_{tag}_mgL_L/g"
            fr_col = f"Kd_Freundlich_at_{tag}_mgL_L/g"

            if (qmax is not None) and (kl is not None) and (ce > 0):
                # Langmuir q(Ce) = Qm*KL*Ce/(1 + KL*Ce), so Kd(Ce) = q(Ce)/Ce.
                df_in.at[i, lm_col] = (qmax * kl) / (1.0 + kl * ce)
            if (kf is not None) and (one_over_n is not None) and (ce > 0):
                if invalid_exponent:
                    pass
                else:
                    df_in.at[i, fr_col] = kf * (ce ** (one_over_n - 1.0))

        if invalid_exponent and (kf is not None):
            errs[i] = (errs[i] + "; " if errs[i] else "") + f"Freundlich Kd benchmarks skipped: exponent={one_over_n:g} must be > 0"

    if any(errs):
        _append_normalization_notes(df_in, errs)


# ---------------- Removal% → Ce, Qe, Kd ----------------

def compute_kd_from_removal(df_in: pd.DataFrame) -> None:
    """
    Kd (L/g) from Removal_rate using standardized columns.
    """
    _ensure_columns(
        df_in,
        ["Removal_rate_avg_fraction", "Ce_from_removal_mg/L",
         "Qe_from_removal_mg/g", "Kd_from_removal_L/g",
         "normalization_trace", "normalization_issues"],
        "",
    )

    notes = []
    for i, row in df_in.iterrows():
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        rem_raw = row.get("Removal_rate")
        rem_txt = _reported_value_text(rem_raw)
        if not rem_txt or (_coerce_number(rem_raw) is None and "%" not in rem_txt):
            notes.append("")
            continue
        rem_val = _coerce_number(rem_raw)

        dosage = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))

        if dosage is None:
            notes.append("Removal→Kd missing inputs: Adsorbent_dosage_value_mg/L")
            continue

        if dosage <= 0:
            notes.append("Removal→Kd invalid dosage")
            continue

        if isinstance(rem_txt, str) and "%" in rem_txt:
            f = max(0.0, min((rem_val or 0.0) / 100.0, 1.0))
        elif rem_val is not None and rem_val <= 1.0:
            f = max(0.0, min(rem_val, 1.0))
        else:
            f = max(0.0, min((rem_val or 0.0) / 100.0, 1.0))

        if f >= 1.0:
            notes.append("Removal→Kd Ce ≤ 0 after applying removal")
            continue

        # A non-positive C0 is incompatible with a reported removal result.
        # Preserve the reported value for audit, but do not derive any endpoint
        # metric from an internally impossible adsorption condition.
        if c0 is not None and c0 <= 0:
            df_in.at[i, "Kd_from_removal_L/g"] = ""
            df_in.at[i, "Ce_from_removal_mg/L"] = ""
            df_in.at[i, "Qe_from_removal_mg/g"] = ""
            notes.append(
                "Removal endpoint invalid because PFAS C0 is zero: Kd, Ce, and Qe not calculated"
            )
            continue

        # Kd from removal and dosage does not require a *reported* C0 because
        # C0 cancels: Kd = Qe/Ce = 1000*f/[D*(1-f)]. If C0 is unavailable,
        # retain the removal-derived Kd but leave Ce and Qe unmaterialized.
        Kd = 1000.0 * f / (dosage * (1.0 - f))
        df_in.at[i, "Removal_rate_avg_fraction"] = f
        df_in.at[i, "Kd_from_removal_L/g"] = Kd

        if c0 is not None:
            Ce = c0 * (1.0 - f)
            if Ce <= 0:
                notes.append("Removal→Kd Ce ≤ 0 after applying removal")
                continue
            Qe = 1000.0 * (c0 - Ce) / dosage
            if Qe < 0:
                notes.append("Removal→Kd negative Qe")
                continue
            df_in.at[i, "Ce_from_removal_mg/L"] = Ce
            df_in.at[i, "Qe_from_removal_mg/g"] = Qe
            notes.append(f"Removal_rate interpreted as {f*100:.4g}%")
        else:
            notes.append(f"Removal_rate interpreted as {f*100:.4g}%; Kd calculated without C0")

    _append_normalization_notes(df_in, notes)


# ---------------- Qe → Ce, Kd ----------------

def compute_kd_from_qe(df_in: pd.DataFrame) -> None:
    """
    Compute Kd from directly reported equilibrium Qe using:
        Ce = C0 - Qe * D/1000
        Kd = Qe / Ce

    Kinetic fitted capacities such as PFO_Qe and PSO_Qe are retained as
    standardized kinetic-model parameters, but are intentionally not converted
    to Ce/Kd for the equilibrium ML target table.
    """
    _ensure_columns(
        df_in,
        ["Ce_from_Qe_mg/L", "Kd_from_Qe_L/g",
         "normalization_trace", "normalization_issues"],
        "",
    )

    sources = [
        ("Qe_value_mg/g", "Ce_from_Qe_mg/L", "Kd_from_Qe_L/g", "Qe"),
    ]

    errs = [""] * len(df_in)

    for i, row in df_in.iterrows():
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        dosage = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))

        for src_col, ce_col, kd_col, lbl in sources:
            q = _coerce_number(row.get(src_col))
            if q is None:
                continue
            if (c0 is None) or (dosage is None):
                continue
            if c0 <= 0:
                errs[i] = (errs[i] + "; " if errs[i] else "") + f"{lbl}→Kd invalid C0"
                continue
            if dosage <= 0:
                errs[i] = (errs[i] + "; " if errs[i] else "") + f"{lbl}→Kd invalid dosage"
                continue
            if q < 0:
                errs[i] = (errs[i] + "; " if errs[i] else "") + f"{lbl}→Kd invalid Qe"
                continue

            Ce = c0 - q * (dosage / 1000.0)
            if Ce is None or Ce <= 0:
                errs[i] = (errs[i] + "; " if errs[i] else "") + f"{lbl}→Kd Ce ≤ 0 given uptake, C0 and dosage"
                continue

            Kd = q / Ce

            df_in.at[i, ce_col] = Ce
            df_in.at[i, kd_col] = Kd

    _append_normalization_notes(df_in, errs)

def compute_ce_from_kd(df_in: pd.DataFrame) -> None:
    col_out = "Ce_from_Kd_mg/L"
    _ensure_columns(df_in, [col_out, "normalization_issues"], "")

    errs = [""] * len(df_in)
    for i, row in df_in.iterrows():
        if _is_isotherm_expanded_inherited_kd(row):
            df_in.at[i, col_out] = ""
            continue
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        kd = _coerce_number(row.get("Kd_value_L/g"))
        dose = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))
        if c0 is None or kd is None or dose is None:
            continue
        invalids = []
        if c0 <= 0:
            invalids.append("Ce_from_Kd invalid C0")
        if dose <= 0:
            invalids.append("Ce_from_Kd invalid dosage")
        if kd < 0:
            invalids.append("Ce_from_Kd invalid Kd")
        if invalids:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "; ".join(invalids)
            continue
        denom = 1.0 + kd * (dose / 1000.0)
        if denom <= 0:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "Ce_from_Kd non-positive denominator"
            continue
        Ce = c0 / denom
        if Ce <= 0:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "Ce_from_Kd ≤ 0"
            continue
        df_in.at[i, col_out] = Ce

    if any(errs):
        _append_normalization_notes(df_in, errs)


def _bisect_root(fun, lo, hi, max_iter=60, tol=1e-12):
    f_lo = fun(lo)
    f_hi = fun(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fun(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def compute_ce_from_freundlich(df_in: pd.DataFrame) -> None:
    col_out = "Ce_from_Freundlich_mg/L"
    _ensure_columns(df_in, [col_out, "normalization_issues"], "")


    errs = [""] * len(df_in)
    for i, row in df_in.iterrows():
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        kf = _coerce_number(row.get("Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)"))
        one_over_n = _coerce_number(row.get("Freundlich_exponent_value"))
        dose = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))
        if c0 is None or kf is None or one_over_n is None or dose is None:
            continue
        invalids = []
        if c0 <= 0:
            invalids.append("Ce_from_Freundlich invalid C0")
        if dose <= 0:
            invalids.append("Ce_from_Freundlich invalid dosage")
        if kf < 0:
            invalids.append("Ce_from_Freundlich invalid KF")
        if one_over_n <= 0:
            invalids.append("Ce_from_Freundlich invalid exponent")
        if invalids:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "; ".join(invalids)
            continue
        A = (dose * kf) / 1000.0

        def f(x):
            return x + A * (x ** one_over_n) - c0

        lo, hi = 0.0, max(c0 if c0 is not None else 0.0, 1.0)
        it = 0
        while f(lo) * f(hi) > 0 and it < 30:
            hi *= 10.0
            it += 1
            if hi > 1e12:
                break
        Ce = _bisect_root(f, lo, hi)
        if Ce is None or Ce <= 0:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "Ce_from_Freundlich root-finding failed"
            continue
        df_in.at[i, col_out] = Ce

    if any(errs):
        _append_normalization_notes(df_in, errs)


def compute_ce_from_langmuir(df_in: pd.DataFrame) -> None:
    col_out = "Ce_from_Langmuir_mg/L"
    _ensure_columns(df_in, [col_out, "normalization_issues"], "")

    errs = [""] * len(df_in)
    for i, row in df_in.iterrows():
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        qmax = _coerce_number(row.get("Langmuir_Qm_value_mg/g"))
        kl = _coerce_number(row.get("Langmuir_KL_value_L/mg"))
        dose = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))
        if c0 is None or qmax is None or kl is None or dose is None:
            continue
        invalids = []
        if c0 <= 0:
            invalids.append("Ce_from_Langmuir invalid C0")
        if dose <= 0:
            invalids.append("Ce_from_Langmuir invalid dosage")
        if qmax < 0:
            invalids.append("Ce_from_Langmuir invalid Qm")
        if kl < 0:
            invalids.append("Ce_from_Langmuir invalid KL")
        if invalids:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "; ".join(invalids)
            continue
        a = 1000.0 / dose
        A = a * kl
        B = (a - a * kl * c0 + qmax * kl)
        C = -a * c0
        disc = B * B - 4 * A * C
        if A == 0 or disc < 0:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "Ce_from_Langmuir no real solution"
            continue
        sqrt_disc = math.sqrt(disc)
        r1 = (-B + sqrt_disc) / (2 * A)
        r2 = (-B - sqrt_disc) / (2 * A)
        candidates = [x for x in (r1, r2) if x is not None and x > 0]
        if c0 is not None:
            candidates = [x for x in candidates if x <= c0 + 1e-9]
        Ce = min(candidates) if candidates else None
        if Ce is None or Ce <= 0:
            errs[i] = (errs[i] + "; " if errs[i] else "") + "Ce_from_Langmuir no physical root"
            continue
        df_in.at[i, col_out] = Ce

    if any(errs):
        _append_normalization_notes(df_in, errs)


def select_final_ce(df_in: pd.DataFrame) -> None:
    _ensure_columns(df_in, ["Ce_final_mg/L", "Ce_final_source"], "")

    sources = [
        ("Ce_from_removal_mg/L", "Removal"),
        ("Ce_from_Qe_mg/L", "Qe"),
        ("Ce_from_Kd_mg/L", "Kd"),
        ("Ce_from_Langmuir_mg/L", "Langmuir"),
        ("Ce_from_Freundlich_mg/L", "Freundlich"),
    ]
    for i, row in df_in.iterrows():
        val, src = _pick_first_number([(row.get(c, ""), s) for c, s in sources])
        df_in.at[i, "Ce_final_mg/L"] = val
        df_in.at[i, "Ce_final_source"] = src


FINAL_CE_BY_KD_SOURCE = {
    "Removal": ("Ce_from_removal_mg/L", "Removal"),
    "Qe": ("Ce_from_Qe_mg/L", "Qe"),
    "Kd": ("Ce_from_Kd_mg/L", "Kd"),
    "Langmuir": ("Ce_from_Langmuir_mg/L", "Langmuir"),
    "Freundlich": ("Ce_from_Freundlich_mg/L", "Freundlich"),
}


def align_final_ce_to_kd_source(df_in: pd.DataFrame) -> None:
    """Make final Ce use the same endpoint family chosen for final Kd.

    The early call to :func:`select_final_ce` remains necessary to derive
    isotherm-based Kd candidates. Once ``Kd_final_source`` is chosen, however,
    the final Ce and every downstream calculation based on it must use that
    source's counterpart instead of the generic Ce priority order.
    """
    _ensure_columns(df_in, ["Ce_final_mg/L", "Ce_final_source", "normalization_issues"], "")

    errors = [""] * len(df_in)
    for i, row in df_in.iterrows():
        kd_source = str(row.get("Kd_final_source", "")).strip()
        ce_spec = FINAL_CE_BY_KD_SOURCE.get(kd_source)
        if ce_spec is None:
            continue

        ce_column, ce_source = ce_spec
        ce_value = _coerce_number(row.get(ce_column))
        if ce_value is None:
            # Do not fall back to a different endpoint family: that would make
            # Ce_final and apparent removal inconsistent with Kd_final.
            df_in.at[i, "Ce_final_mg/L"] = ""
            df_in.at[i, "Ce_final_source"] = ""
            errors[i] = f"Ce_final unavailable for Kd_final_source {kd_source}"
            continue

        df_in.at[i, "Ce_final_mg/L"] = ce_value
        df_in.at[i, "Ce_final_source"] = ce_source

    if any(errors):
        df_in["normalization_issues"] = (
            df_in["normalization_issues"].fillna("").astype(str).str.rstrip()
            + ["; " + message if message else "" for message in errors]
        ).str.strip("; ").str.strip()


def _pick_first_number(vals_with_src):
    for v, src in vals_with_src:
        num = _coerce_number(v)
        if num is not None:
            return num, src
    return "", ""

def compute_kd_from_isotherm_ce(df_in: pd.DataFrame) -> None:
    """
    Compute Kd from isotherm-derived Ce using mass balance:
        Qe = 1000 · (C0 − Ce) / D
        Kd = Qe / Ce

    where:
      - C0 = PFAS_C0_value_mg/L
      - D  = Adsorbent_dosage_value_mg/L
    """
    _ensure_columns(
        df_in,
        ["Kd_from_Langmuir_L/g", "Kd_from_Freundlich_L/g", "normalization_issues"],
        "",
    )

    errs = [""] * len(df_in)
    for i, row in df_in.iterrows():
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        dose = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))
        if c0 is None or dose is None or dose <= 0:
            continue

        # Langmuir-based Kd
        ce_lm = _coerce_number(row.get("Ce_from_Langmuir_mg/L"))
        if ce_lm is not None and ce_lm > 0:
            qmax = _coerce_number(row.get("Langmuir_Qm_value_mg/g"))
            kl = _coerce_number(row.get("Langmuir_KL_value_L/mg"))
            if qmax is not None and kl is not None and qmax >= 0 and kl >= 0:
                df_in.at[i, "Kd_from_Langmuir_L/g"] = (qmax * kl) / (1.0 + kl * ce_lm)
            else:
                q_lm = 1000.0 * (c0 - ce_lm) / dose
                if q_lm < 0:
                    errs[i] = (errs[i] + "; " if errs[i] else "") + "Kd_from_Langmuir negative Qe from Ce"
                else:
                    df_in.at[i, "Kd_from_Langmuir_L/g"] = q_lm / ce_lm

        # Freundlich-based Kd
        ce_fr = _coerce_number(row.get("Ce_from_Freundlich_mg/L"))
        if ce_fr is not None and ce_fr > 0:
            kf = _coerce_number(row.get("Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)"))
            one_over_n = _coerce_number(row.get("Freundlich_exponent_value"))
            if kf is not None and one_over_n is not None and kf >= 0 and one_over_n > 0:
                df_in.at[i, "Kd_from_Freundlich_L/g"] = kf * (ce_fr ** (one_over_n - 1.0))
            else:
                q_fr = 1000.0 * (c0 - ce_fr) / dose
                if q_fr < 0:
                    errs[i] = (errs[i] + "; " if errs[i] else "") + "Kd_from_Freundlich negative Qe from Ce"
                else:
                    df_in.at[i, "Kd_from_Freundlich_L/g"] = q_fr / ce_fr

    if any(errs):
        _append_normalization_notes(df_in, errs)


# ---------------- Final Kd selector ----------------

ISOTHERM_R2_MINIMUM = 0.90
# Reported R2 values are often rounded and rarely carry enough information to
# distinguish fits that differ by only a few hundredths. In that interval,
# prefer the endpoint with fewer conversion-risk signals.
ISOTHERM_R2_TIE_BAND = 0.02
UNIVERSAL_ANALYTICAL_CONCENTRATION_FLOOR_MG_L = 1e-5
LOW_CE_DETECTION_LIMIT_MG_L = UNIVERSAL_ANALYTICAL_CONCENTRATION_FLOOR_MG_L
SMALL_CONCENTRATION_DIFFERENCE_LIMIT_MG_L = UNIVERSAL_ANALYTICAL_CONCENTRATION_FLOOR_MG_L
LOW_REMOVAL_RATE_FRACTION = 0.01
# This is a physical-plausibility screen for a final endpoint, independent of
# whether Kd itself was reported or calculated.  It intentionally uses the
# same one-percent tail as the conversion-sensitivity calculations below.
HIGH_APPARENT_REMOVAL_FRACTION = 0.99
LOGKD_CONVERSION_REMOVAL_PERTURBATION = 0.01
LOGKD_CONVERSION_MAX_DELTA_LOG10_KD = 0.35

ENDPOINT_CANDIDATE_SPECS = (
    ("Kd", "Kd_value_L/g", "Ce_from_Kd_mg/L", None),
    ("Removal", "Kd_from_removal_L/g", "Ce_from_removal_mg/L", None),
    ("Qe", "Kd_from_Qe_L/g", "Ce_from_Qe_mg/L", None),
    ("Langmuir", "Kd_from_Langmuir_L/g", "Ce_from_Langmuir_mg/L", "Langmuir_R2"),
    ("Freundlich", "Kd_from_Freundlich_L/g", "Ce_from_Freundlich_mg/L", "Freundlich_R2"),
)
ENDPOINT_SOURCE_PRIORITY = {
    "Kd": 0,
    "Removal": 1,
    "Qe": 2,
    "Langmuir": 3,
    "Freundlich": 4,
}


def _is_isotherm_expanded_inherited_kd(row) -> bool:
    """Return whether a reported Kd was copied onto generated isotherm rows.

    Such a Kd can be retained for audit but cannot safely be paired with any
    generated C0/dosage condition, so it is not an endpoint candidate.
    """
    value = row.get(ISOTHERM_EXPANDED_INHERITED_KD_FLAG_COLUMN, False)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _eligible_isotherm_kd_candidates(row) -> list[tuple[float, str, float]]:
    """Return curve-derived Kd candidates whose matching fit R2 passes the gate."""
    candidates = []
    for kd_col, source, r2_col in (
        ("Kd_from_Langmuir_L/g", "Langmuir", "Langmuir_R2"),
        ("Kd_from_Freundlich_L/g", "Freundlich", "Freundlich_R2"),
    ):
        kd = _coerce_number(row.get(kd_col))
        r2 = _coerce_number(row.get(r2_col))
        if (
            kd is not None
            and r2 is not None
            and math.isfinite(r2)
            and r2 >= ISOTHERM_R2_MINIMUM
        ):
            candidates.append((kd, source, r2))
    return candidates


def _endpoint_candidate_risk_reasons(ce: float, c0: float) -> list[str]:
    """Return non-independent endpoint quality risks once per root cause."""
    removal = 1.0 - (ce / c0)
    if ce <= LOW_CE_DETECTION_LIMIT_MG_L:
        return ["Ce <= 10 ng/L"]
    if removal >= 1.0 - LOGKD_CONVERSION_REMOVAL_PERTURBATION:
        return ["apparent removal >= 99%"]
    if removal <= LOGKD_CONVERSION_REMOVAL_PERTURBATION:
        return ["apparent removal <= 1%"]
    return []


def _complete_endpoint_candidates(row) -> list[dict[str, object]]:
    """Build comparable, mass-balance-complete Kd/Ce/removal candidates.

    Kd, Ce, and apparent removal are mathematically coupled, so a candidate is
    assessed as one endpoint bundle rather than as three independent votes.
    """
    c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
    if c0 is None or not math.isfinite(c0) or c0 <= 0:
        return []

    candidates: list[dict[str, object]] = []
    for source, kd_column, ce_column, r2_column in ENDPOINT_CANDIDATE_SPECS:
        if source == "Kd" and _is_isotherm_expanded_inherited_kd(row):
            continue
        kd = _coerce_number(row.get(kd_column))
        ce = _coerce_number(row.get(ce_column))
        if (
            kd is None
            or ce is None
            or not math.isfinite(kd)
            or not math.isfinite(ce)
            or kd <= 0
            or not 0.0 < ce < c0
        ):
            continue

        r2 = None
        if r2_column is not None:
            r2 = _coerce_number(row.get(r2_column))
            if r2 is None or not math.isfinite(r2) or r2 < ISOTHERM_R2_MINIMUM:
                continue

        risk_reasons = _endpoint_candidate_risk_reasons(ce, c0)
        candidates.append(
            {
                "source": source,
                "kd": kd,
                "ce": ce,
                "r2": r2,
                "risk_reasons": risk_reasons,
            }
        )
    return candidates


def _select_complete_endpoint_candidate(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    """Select the lowest-risk endpoint, preserving direct data when tied."""
    if not candidates:
        return None

    lowest_risk_count = min(len(candidate["risk_reasons"]) for candidate in candidates)
    lowest_risk = [
        candidate
        for candidate in candidates
        if len(candidate["risk_reasons"]) == lowest_risk_count
    ]

    direct = [candidate for candidate in lowest_risk if candidate["r2"] is None]
    if direct:
        return min(
            direct,
            key=lambda candidate: ENDPOINT_SOURCE_PRIORITY[str(candidate["source"])],
        )

    best_r2 = max(float(candidate["r2"]) for candidate in lowest_risk)
    tied = [
        candidate
        for candidate in lowest_risk
        if best_r2 - float(candidate["r2"]) <= ISOTHERM_R2_TIE_BAND
    ]
    return min(
        tied,
        key=lambda candidate: ENDPOINT_SOURCE_PRIORITY[str(candidate["source"])],
    )


def _endpoint_selection_note(
    selected: dict[str, object],
    candidates: list[dict[str, object]],
) -> str:
    """Return a compact audit note only when a non-trivial choice was made."""
    alternatives = [
        candidate
        for candidate in candidates
        if candidate["source"] != selected["source"]
    ]
    if not alternatives:
        return ""

    selected_source = str(selected["source"])
    risky = [
        f"{candidate['source']} ({', '.join(candidate['risk_reasons'])})"
        for candidate in alternatives
        if candidate["risk_reasons"]
    ]
    if risky:
        return (
            f"Endpoint selection: {selected_source} selected; "
            f"not selected: {'; '.join(risky)}"
        )

    model_alternatives = [
        candidate
        for candidate in alternatives
        if candidate["r2"] is not None and selected["r2"] is not None
    ]
    if model_alternatives:
        competitor = max(model_alternatives, key=lambda candidate: float(candidate["r2"]))
        delta = abs(float(selected["r2"]) - float(competitor["r2"]))
        if delta <= ISOTHERM_R2_TIE_BAND:
            return (
                f"Endpoint selection: {selected_source} selected over {competitor['source']} "
                f"(R2 difference {delta:.3f} within {ISOTHERM_R2_TIE_BAND:.3f} tie band)"
            )
    return ""


def _isotherm_r2_blocker_reason(row, kd_col: str, source: str, r2_col: str) -> str:
    """Explain why an otherwise available curve-derived Kd failed the fit gate."""
    if _coerce_number(row.get(kd_col)) is None:
        return ""
    r2 = _coerce_number(row.get(r2_col))
    if r2 is None or not math.isfinite(r2):
        return f"{source} reported fit R2 missing"
    if r2 < ISOTHERM_R2_MINIMUM:
        return f"{source} reported fit R2 below {ISOTHERM_R2_MINIMUM:.2f}"
    return ""

def _diagnose_kd_blockers(row) -> list[str]:
    reasons = []
    c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
    dose = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))
    qe_sources = [
        ("Qe_value_mg/g", "Ce_from_Qe_mg/L", "Kd_from_Qe_L/g", "Qe"),
    ]
    for val_col, ce_col, kd_col, label in qe_sources:
        q = _coerce_number(row.get(val_col))
        kd = _coerce_number(row.get(kd_col))
        if q is None:
            continue
        if kd is not None:
            continue
        missing = []
        if c0 is None:
            missing.append("PFAS_C0_value_mg/L")
        if dose is None:
            missing.append("Adsorbent_dosage_value_mg/L")
        if missing:
            reasons.append(f"{label} missing " + ", ".join(missing))
        else:
            if c0 <= 0:
                reasons.append(f"{label} invalid C0")
            elif dose <= 0:
                reasons.append(f"{label} invalid dosage")
            elif q < 0:
                reasons.append(f"{label} invalid Qe")
            else:
                ce = _coerce_number(row.get(ce_col))
                if ce is not None and ce <= 0:
                    reasons.append(f"{label} Ce ≤ 0")
                else:
                    reasons.append(f"{label} could not compute")

    rem_raw = row.get("Removal_rate")
    rem_txt = _reported_value_text(rem_raw)
    kd_rem = _coerce_number(row.get("Kd_from_removal_L/g"))
    if rem_txt != "" and kd_rem is None:
        if c0 is not None and c0 <= 0:
            # The removal normalizer has already recorded the source-data
            # conflict; avoid adding a vague duplicate Kd-final failure.
            pass
        elif dose is None:
            reasons.append("Removal missing Adsorbent_dosage_value_mg/L")
        elif dose <= 0:
            reasons.append("Removal invalid dosage")
        else:
            ce = _coerce_number(row.get("Ce_from_removal_mg/L"))
            if ce is not None and ce <= 0:
                reasons.append("Removal Ce ≤ 0")
            else:
                reasons.append("Removal could not compute (check removal value or dosage)")
    # Langmuir/Freundlich: just diagnose missing parameters, no assumed-Ce Kd
    qmax = _coerce_number(row.get("Langmuir_Qm_value_mg/g"))
    kl = _coerce_number(row.get("Langmuir_KL_value_L/mg"))
    if (qmax is not None or kl is not None):
        miss = []
        if qmax is None:
            miss.append("Langmuir_Qm_value_mg/g")
        if kl is None:
            miss.append("Langmuir_KL_value_L/mg")
        if miss:
            reasons.append("Langmuir missing " + ", ".join(miss))
        else:
            missing = []
            if c0 is None:
                missing.append("PFAS_C0_value_mg/L")
            if dose is None:
                missing.append("Adsorbent_dosage_value_mg/L")
            if missing:
                reasons.append("Langmuir missing " + ", ".join(missing))
            else:
                invalids = []
                if c0 <= 0:
                    invalids.append("C0")
                if dose <= 0:
                    invalids.append("dosage")
                if qmax < 0:
                    invalids.append("Qm")
                if kl < 0:
                    invalids.append("KL")
                if invalids:
                    reasons.append("Langmuir invalid " + ", ".join(invalids))

    kf = _coerce_number(row.get("Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)"))
    one_over_n = _coerce_number(row.get("Freundlich_exponent_value"))
    if (kf is not None or one_over_n is not None):
        miss = []
        if kf is None:
            raw_kf_reported = (
                _coerce_number(row.get("Freundlich_KF_value")) is not None
                or bool(_reported_value_text(row.get("Freundlich_KF_value")))
            )
            if one_over_n is not None and one_over_n <= 0:
                reasons.append("Freundlich exponent must be > 0")
            elif raw_kf_reported:
                reasons.append("Freundlich KF could not standardize")
            else:
                miss.append("Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)")
        if one_over_n is None:
            miss.append("Freundlich_exponent_value")
        if miss:
            reasons.append("Freundlich missing " + ", ".join(miss))
        elif kf is not None and one_over_n is not None:
            missing = []
            if c0 is None:
                missing.append("PFAS_C0_value_mg/L")
            if dose is None:
                missing.append("Adsorbent_dosage_value_mg/L")
            if missing:
                reasons.append("Freundlich missing " + ", ".join(missing))
            else:
                invalids = []
                if c0 <= 0:
                    invalids.append("C0")
                if dose <= 0:
                    invalids.append("dosage")
                if kf < 0:
                    invalids.append("KF")
                if one_over_n <= 0:
                    invalids.append("exponent")
                if invalids:
                    reasons.append("Freundlich invalid " + ", ".join(invalids))

    for kd_col, source, r2_col in (
        ("Kd_from_Langmuir_L/g", "Langmuir", "Langmuir_R2"),
        ("Kd_from_Freundlich_L/g", "Freundlich", "Freundlich_R2"),
    ):
        reason = _isotherm_r2_blocker_reason(row, kd_col, source, r2_col)
        if reason:
            reasons.append(reason)

    return reasons


def select_final_kd(df_in: pd.DataFrame) -> None:
    _ensure_columns(df_in, ["Kd_final_L/g", "Kd_final_source", "normalization_trace", "normalization_issues"], "")

    direct_k_cols = [
        ("Kd_value_L/g", "Kd"),
        ("Kd_from_removal_L/g", "Removal"),
        ("Kd_from_Qe_L/g", "Qe"),
    ]

    selection_notes = [""] * len(df_in)
    for i, row in df_in.iterrows():
        inherited_kd = _is_isotherm_expanded_inherited_kd(row)
        note_parts = []
        if inherited_kd:
            note_parts.append(
                "Reported Kd retained but excluded from final selection after isotherm expansion"
            )

        candidates = _complete_endpoint_candidates(row)
        selected = _select_complete_endpoint_candidate(candidates)
        if selected is not None:
            df_in.at[i, "Kd_final_L/g"] = selected["kd"]
            df_in.at[i, "Kd_final_source"] = selected["source"]
            endpoint_note = _endpoint_selection_note(selected, candidates)
            if endpoint_note:
                note_parts.append(endpoint_note)
            selection_notes[i] = "; ".join(note_parts)
            continue

        # Preserve the former priority-based behavior for incomplete endpoint
        # bundles. An inherited Kd is excluded because no generated C0/dosage
        # row can be identified as its reported calculation condition.
        ordered_vals = [
            (row.get(c, ""), src)
            for c, src in direct_k_cols
            if src != "Kd" or not inherited_kd
        ]
        val, src = _pick_first_number(ordered_vals)
        if not src:
            isotherm_candidates = _eligible_isotherm_kd_candidates(row)
            if isotherm_candidates:
                val, src, _ = max(
                    isotherm_candidates,
                    key=lambda candidate: (
                        candidate[2],
                        candidate[1] == "Langmuir",
                    ),
                )
        df_in.at[i, "Kd_final_L/g"] = val
        df_in.at[i, "Kd_final_source"] = src
        if src:
            note_parts.append(
                f"Endpoint selection fallback: {src} selected without a complete C0/Ce bundle"
            )
        selection_notes[i] = "; ".join(note_parts)

    _append_normalization_notes(df_in, selection_notes)

    if "normalization_issues" in df_in.columns:
        if "Test_mode" in df_in.columns:
            is_kinetic = (
                df_in["Test_mode"].astype(str).str.strip().str.lower().str.startswith("kinet")
            )
        else:
            is_kinetic = pd.Series(False, index=df_in.index)

        need_err = df_in["Kd_final_L/g"].apply(lambda x: (x == "") or pd.isna(x)) & (~is_kinetic)
        msgs = []
        for i, row in df_in.iterrows():
            if not need_err.iloc[i]:
                msgs.append("")
                continue
            reasons = _diagnose_kd_blockers(row)
            if reasons:
                msgs.append("Kd_final not determined: " + "; ".join(reasons))
            else:
                msgs.append("")
        df_in["normalization_issues"] = (
            df_in["normalization_issues"].fillna("").astype(str).str.rstrip()
            + ["; " + m if m else "" for m in msgs]
        ).str.strip("; ").str.strip()


# Conversion-instability annotations for derived log10(Kd).
# These labels identify numerical sensitivity in the conversion, not an
# adsorption mechanism or a physical impossibility.
MASS_BALANCE_KD_SOURCES = {"Removal", "Qe"}


def _mass_balance_conversion_instability_flags(apparent_removal: float) -> tuple[bool, bool]:
    """Return (spike, dip) flags for Kd proportional to R / (1 - R)."""

    h = LOGKD_CONVERSION_REMOVAL_PERTURBATION
    limit = LOGKD_CONVERSION_MAX_DELTA_LOG10_KD
    spike = False
    dip = False

    if apparent_removal + h >= 1.0:
        spike = True
    elif 0.0 < apparent_removal < 1.0 - h:
        baseline = math.log10(apparent_removal / (1.0 - apparent_removal))
        shifted = math.log10((apparent_removal + h) / (1.0 - apparent_removal - h))
        spike = (shifted - baseline) > limit

    if apparent_removal - h <= 0.0:
        dip = True
    elif h < apparent_removal < 1.0:
        baseline = math.log10(apparent_removal / (1.0 - apparent_removal))
        shifted = math.log10((apparent_removal - h) / (1.0 - apparent_removal + h))
        dip = (baseline - shifted) > limit

    return spike, dip


def _freundlich_conversion_instability_flags(
    apparent_removal: float,
    exponent: float | None,
) -> tuple[bool, bool]:
    """Return (spike, dip) flags for Kd = KF * Ce**(exponent - 1).

    ``exponent`` is the applied power on Ce (``Freundlich_exponent_value``),
    whether the source reported it as 1/n or as n. Raising removal by h
    changes log10(Kd) by (1 - exponent) * log10[(1 - R) / (1 - R - h)]: a
    spike when exponent < 1 and a dip (Kd collapsing towards zero) when
    exponent > 1. Both directions use the same magnitude threshold.
    """

    if exponent is None or not math.isfinite(exponent) or exponent <= 0.0 or exponent == 1.0:
        return False, False

    h = LOGKD_CONVERSION_REMOVAL_PERTURBATION
    if apparent_removal + h >= 1.0:
        unstable = True
    elif not 0.0 <= apparent_removal < 1.0 - h:
        unstable = False
    else:
        delta_log10_kd = (1.0 - exponent) * math.log10(
            (1.0 - apparent_removal) / (1.0 - apparent_removal - h)
        )
        unstable = abs(delta_log10_kd) > LOGKD_CONVERSION_MAX_DELTA_LOG10_KD

    if exponent < 1.0:
        return unstable, False
    return False, unstable


def compute_apparent_removal_and_logkd_conversion_instability_flags(df_in: pd.DataFrame) -> None:
    """Materialize apparent removal and separate log10(Kd) spike/dip labels.

    apparent_removal_fraction is calculated as ``1 - Ce / C0`` when both
    values are available. For a directly reported final Kd with a known dosage,
    the same fraction can instead be inferred as ``Kd * D / (1000 + Kd * D)``.
    That diagnostic does not change Kd or imply that it was converted. It uses
    Removal_rate_avg_fraction only when final Kd is also removal-derived (or no
    final Kd source exists), preventing a different endpoint family from being
    mixed into a source-aligned result. Values are intentionally not clipped to
    [0, 1], so mass-balance problems remain visible. The low-Ce flag applies
    only when the final selected endpoint supplies a finite Ce; ``NA`` means
    the detection-limit check was not available. When both C0 and Ce are valid,
    their absolute difference is materialized and compared with the same
    universal analytical concentration floor. The low- and high-apparent-
    removal flags use symmetric inclusive tails (<=1% and >=99%) for every
    valid apparent-removal value. These physical data-quality flags remain
    separate from conversion-instability flags, which are nullable and are
    evaluated only for Kd values calculated from another endpoint family.
    """
    df_in.drop(
        columns=[
            "high_logKd_conversion_spike_flag",
            "high_logKd_conversion_spike_reason",
        ],
        inplace=True,
        errors="ignore",
    )
    _ensure_columns(
        df_in,
        [
            "apparent_removal_fraction",
            "apparent_removal_source",
            "logKd_conversion_spike_flag",
            "logKd_conversion_dip_flag",
            "low_Ce_detection_limit_flag",
            "concentration_difference_C0_minus_Ce_mg/L",
            "small_concentration_difference_flag",
            "low_removal_rate_flag",
            "high_apparent_removal_flag",
        ],
        "",
    )
    df_in["logKd_conversion_spike_flag"] = pd.Series(pd.NA, index=df_in.index, dtype="boolean")
    df_in["logKd_conversion_dip_flag"] = pd.Series(pd.NA, index=df_in.index, dtype="boolean")
    df_in["low_Ce_detection_limit_flag"] = pd.Series(pd.NA, index=df_in.index, dtype="boolean")
    df_in["concentration_difference_C0_minus_Ce_mg/L"] = pd.Series(
        float("nan"),
        index=df_in.index,
        dtype="float64",
    )
    df_in["small_concentration_difference_flag"] = pd.Series(
        pd.NA,
        index=df_in.index,
        dtype="boolean",
    )
    df_in["low_removal_rate_flag"] = pd.Series(pd.NA, index=df_in.index, dtype="boolean")
    df_in["high_apparent_removal_flag"] = pd.Series(pd.NA, index=df_in.index, dtype="boolean")

    for i, row in df_in.iterrows():
        c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
        ce = _coerce_number(row.get("Ce_final_mg/L"))
        removal_reported = _coerce_number(row.get("Removal_rate_avg_fraction"))
        kd_source = str(row.get("Kd_final_source", "")).strip()

        if ce is not None and math.isfinite(ce):
            df_in.at[i, "low_Ce_detection_limit_flag"] = (
                0.0 <= ce <= LOW_CE_DETECTION_LIMIT_MG_L
            )

        if (
            c0 is not None
            and ce is not None
            and math.isfinite(c0)
            and math.isfinite(ce)
            and c0 > 0.0
            and 0.0 <= ce <= c0
        ):
            concentration_difference = c0 - ce
            df_in.at[i, "concentration_difference_C0_minus_Ce_mg/L"] = (
                concentration_difference
            )
            df_in.at[i, "small_concentration_difference_flag"] = (
                concentration_difference <= SMALL_CONCENTRATION_DIFFERENCE_LIMIT_MG_L
            )

        apparent = None
        apparent_source = ""
        if c0 is not None and c0 > 0 and ce is not None:
            apparent = 1.0 - (ce / c0)
            apparent_source = "Ce_final_mg/L and PFAS_C0_value_mg/L"
        elif kd_source == "Kd":
            kd = _coerce_number(row.get("Kd_final_L/g"))
            dose = _coerce_number(row.get("Adsorbent_dosage_value_mg/L"))
            if (
                kd is not None
                and dose is not None
                and math.isfinite(kd)
                and math.isfinite(dose)
                and kd >= 0.0
                and dose > 0.0
            ):
                kd_dose = kd * (dose / 1000.0)
                if math.isfinite(kd_dose) and kd_dose >= 0.0:
                    apparent = kd_dose / (1.0 + kd_dose)
                    apparent_source = (
                        "Kd_final_L/g and Adsorbent_dosage_value_mg/L "
                        "(direct-Kd diagnostic)"
                    )
        elif removal_reported is not None and kd_source in {"", "Removal"}:
            apparent = removal_reported
            apparent_source = "Removal_rate_avg_fraction"

        if apparent is None or not math.isfinite(float(apparent)):
            continue

        df_in.at[i, "apparent_removal_fraction"] = apparent
        df_in.at[i, "apparent_removal_source"] = apparent_source
        df_in.at[i, "low_removal_rate_flag"] = (
            apparent <= LOW_REMOVAL_RATE_FRACTION
        )
        df_in.at[i, "high_apparent_removal_flag"] = (
            apparent >= HIGH_APPARENT_REMOVAL_FRACTION
        )
        if kd_source in MASS_BALANCE_KD_SOURCES:
            spike, dip = _mass_balance_conversion_instability_flags(apparent)
            df_in.at[i, "logKd_conversion_spike_flag"] = spike
            df_in.at[i, "logKd_conversion_dip_flag"] = dip
        elif kd_source == "Freundlich":
            exponent = _coerce_number(row.get("Freundlich_exponent_value"))
            spike, dip = _freundlich_conversion_instability_flags(apparent, exponent)
            df_in.at[i, "logKd_conversion_spike_flag"] = spike
            df_in.at[i, "logKd_conversion_dip_flag"] = dip
        elif kd_source == "Langmuir":
            df_in.at[i, "logKd_conversion_spike_flag"] = False
            df_in.at[i, "logKd_conversion_dip_flag"] = False


def mark_not_applicable_adsorbent_fields(df_in: pd.DataFrame) -> None:
    """
    Mark adsorbent-property fields as 'not applicable' when they do not
    conceptually apply, so they can be distinguished from truly missing data.

    Rules:
      1) When adsorbent_category ≠ 'Activated carbon', AC_activation_method
         does not apply.
       2) When adsorbent_category ≠ 'Ion exchange resin', all ion exchange
          capacity fields do not apply:
            - ion_exchange_capacity_value
            - ion_exchange_capacity_unit
            - ion_exchange_capacity_value_meq/g
            - ion_exchange_capacity_value_meq/L

    We only enforce 'not applicable' when adsorbent_category is non-blank
    and explicitly different from the relevant category, so that rows with
    unknown / missing categories are left untouched.
    """
    _ensure_columns(df_in, ["adsorbent_category"], default="")

    cats = df_in["adsorbent_category"].astype(str).str.strip().str.lower()

    # (1) AC_activation_method for non–activated-carbon adsorbents
    if "AC_activation_method" in df_in.columns:
        mask_non_ac = (cats != "") & (cats != "activated carbon")
        # For all adsorbents that are explicitly not activated carbon,
        # AC_activation_method does not conceptually apply.
        df_in.loc[mask_non_ac, "AC_activation_method"] = "not applicable"

    # (2) ion-exchange capacity fields for non–ion-exchange-resin adsorbents
    ie_cols = [
        "ion_exchange_capacity_value",
        "ion_exchange_capacity_unit",
        "ion_exchange_capacity_value_meq/g",
        "ion_exchange_capacity_value_meq/L",
    ]
    _ensure_columns(df_in, ie_cols, default="")

    mask_non_ier = (cats != "") & (cats != "ion exchange resin")
    for col in ie_cols:
        if col in df_in.columns:
            df_in.loc[mask_non_ier, col] = "not applicable"

def _add_log10_kd_columns(df_in: pd.DataFrame) -> None:
    kd_cols = [c for c in df_in.columns if ("Kd" in c) and c.endswith("_L/g")]
    for c in kd_cols:
        log_c = c.replace("_L/g", "_log10(L/g)")
        if log_c not in df_in.columns:
            df_in[log_c] = ""
        df_in[log_c] = df_in[c].apply(
            lambda v: (math.log10(float(v)) if (v not in ("", None) and pd.notna(v)
                                               and _coerce_number(v) is not None and float(v) > 0)
                       else "")
        )

def load_column_order(csv_path: str) -> list[str]:
    """
    Load a desired column order from a simple CSV with header 'column'.

    - Ignores blank rows.
    - If the file is missing, returns [] (caller can fall back to current behavior).
    - Tries several encodings so older 'ANSI'/cp1252 CSVs won't fail.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] Column order file not found: {csv_path}")
        return []

    encodings_to_try = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_err = None

    for enc in encodings_to_try:
        try:
            df_order = pd.read_csv(csv_path, encoding=enc)
            if "column" not in df_order.columns:
                print(
                    f"[WARN] Column order file {csv_path} missing 'column' header; "
                    f"columns present: {list(df_order.columns)}"
                )
                return []
            cols = [
                str(c).strip()
                for c in df_order["column"].tolist()
                if str(c).strip()
            ]
            return cols
        except UnicodeDecodeError as e:
            # Try the next encoding
            last_err = e
            continue
        except Exception as e:
            # Other errors (e.g. parser issues) – report and abort
            print(f"[WARN] Failed to read column order from {csv_path} with encoding {enc}: {e}")
            return []

    # Only reached if *all* encodings hit UnicodeDecodeError
    print(
        f"[WARN] Failed to read column order from {csv_path} with encodings "
        f"{encodings_to_try}: {last_err}"
    )
    return []

# ---------------- Final save ----------------

STANDARDIZED_FULL_SHEET_NAME = "Standardized_Full"
EQUILIBRIUM_DATA_SHEET_NAME = "Equilibrium_Data"
REFERENCE_DOSE_SEARCH_SHEET_NAME = "Dose_25mgL_Search"
NON_EQUILIBRIUM_DATA_SHEET_NAME = "Non_Equilibrium_Data"
EQUILIBRIUM_STATUS_REVIEW_SHEET_NAME = "Equilibrium_Status_Review"
REVIEW_REQUIRED_SHEET_NAME = "Review_Required"

RECORD_STATUS_COLUMN = "record_status"
STATUS_REASON_COLUMN = "status_reason"
NORMALIZATION_TRACE_COLUMN = "normalization_trace"
REVIEW_OUTCOME_COLUMN = "review_outcome"

# These legacy status-message fields remain available while normalization is
# running. ``materialize_record_status`` folds them into the concise output
# schema only after all normalization is complete.
LEGACY_MESSAGE_COLUMNS = [
    "normalization_issues",
    "require_review",
    "review_reason",
    "warning_message",
    "human_review_decision",
    "human_review_unreliable_flag",
]

MODEL_INCLUDE_COL = "model_include"
MODEL_EXCLUDE_REASON_COL = "model_exclude_reason"
MODEL_FAMILY_COL = "model_family_for_audit"

EQUILIBRIUM_ALLOWED_KD_FINAL_SOURCES = {"Kd", "Removal", "Qe", "Langmuir", "Freundlich"}
EQUILIBRIUM_ENDPOINT_KD_FINAL_SOURCES = {"Kd", "Qe", "Langmuir", "Freundlich"}
EQUILIBRIUM_DETERMINATION_COLUMN = "Equilibrium_determination"

EQUILIBRIUM_DETERMINATIONS = {
    "equilibrium_endpoint_reported",
    "removal_explicit_equilibrium",
    "removal_single_timepoint_assumed",
    "removal_last_timepoint_assumed",
}

NON_EQUILIBRIUM_DETERMINATIONS = {
    "kinetic_model_reported",
    "removal_explicit_non_equilibrium",
    "removal_intermediate_timepoint",
}

UNKNOWN_EQUILIBRIUM_DETERMINATIONS = {
    "removal_equilibrium_unclear",
    "endpoint_unclear",
}

EQUILIBRIUM_ENDPOINT_INDICATOR_COLUMNS = [
    "Kd_value",
    "Kd_value_L/g",
    "Qe_value",
    "Qe_value_mg/g",
    "Langmuir_Qm_value",
    "Langmuir_Qm_value_mg/g",
    "Langmuir_KL_value",
    "Langmuir_KL_value_L/mg",
    "Freundlich_KF_value",
    "Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)",
    "Freundlich_n_or_1/n_verbatim",
    "Freundlich_exponent_value",
]

PERFORMANCE_INDICATOR_COLUMNS = (
    EQUILIBRIUM_ENDPOINT_INDICATOR_COLUMNS
    + ["Removal_rate", "Removal_rate_avg_fraction"]
)

REMOVAL_TIME_GROUP_COLUMNS = [
    "extraction_dataset",
    "study_no",
    "DOI",
    "PFAS_name",
    "adsorbent_id",
    "Adsorbent_id",
    "Test_mode",
    "Water_type",
    "pH",
    "PFAS_C0_value_mg/L",
    "PFAS_C0_value",
    "Adsorbent_dosage_value_mg/L",
    "Adsorbent_dosage_value",
    "Solution_volume_(mL)",
    "Mixing_speed_(rpm)",
    "Inorganic_matter",
    "inorganic_matter_present",
    "contains_Na",
    "contains_K",
    "contains_Ca",
    "contains_Mg",
    "contains_Cl",
    "contains_HCO3",
    "contains_SO4",
    "contains_phosphate",
    "Organic_matter",
    "organic_matter_present",
    "organic_matter_class",
    "Ionic_strength_(mol/L)",
    "organic_carbon_mg/L",
    "TDS_(mg/L)",
]

KINETIC_INDICATOR_COLUMNS = [
    "PFO_k1_value",
    "PFO_k1_value_1/h",
    "PFO_Qe_value",
    "PFO_Qe_value_mg/g",
    "PSO_k2_value",
    "PSO_k2_value_g/(mg·h)",
    "PSO_v0_value",
    "PSO_v0_value_mg/(g·h)",
    "PSO_Qe_value",
    "PSO_Qe_value_mg/g",
]

REVIEW_SIGNAL_COLUMNS = [
    "require_review",
    "review_reason",
    "warning_message",
    "normalization_issues",
]


def _existing_unique(columns, available_columns):
    """Return requested columns that exist, preserving order and removing duplicates."""
    available = set(available_columns)
    out = []
    seen = set()
    for col in columns:
        if col in available and col not in seen:
            out.append(col)
            seen.add(col)
    return out

def _column_family_key(col: str) -> str:
    """Return a stable grouping key for sibling columns."""
    text = str(col or "").strip()
    if not text:
        return text

    match = re.match(r"^(?P<base>.+?)_(?:raw|context)(?:_\([^)]*\))?$", text)
    if match:
        return match.group("base")

    for pattern in (
        r"_unit$",
        r"_value(?:_.+)?$",
    ):
        reduced = re.sub(pattern, "", text)
        if reduced != text:
            return reduced

    reduced = re.sub(r"_\([^)]*\)$", "", text)
    if reduced != text:
        return reduced

    return text


def order_columns_by_family(
    available_columns: list[str],
    preferred_order: list[str] | None = None,
    pinned: list[str] | None = None,
) -> list[str]:
    """Order columns by coarse anchors, with automatic sibling grouping.

    Exact CSV anchors use the inferred family key, so pH pulls pH_raw and
    pH_context. CSV anchors ending in ``*`` are prefix families, so a row like
    ``porosity_*`` pulls all columns whose names start with ``porosity_``.
    """
    available = list(dict.fromkeys(available_columns))
    available_set = set(available)
    placed: set[str] = set()
    ordered: list[str] = []

    def emit_family(anchor: str) -> None:
        anchor_text = str(anchor or "").strip()
        if not anchor_text:
            return

        if anchor_text.endswith("*"):
            prefix = anchor_text[:-1]
            siblings = [
                col for col in available
                if col not in placed and str(col).startswith(prefix)
            ]
            family_order = sorted(siblings, key=lambda c: (str(c).casefold(), str(c)))
        else:
            if anchor_text not in available_set:
                return
            family = _column_family_key(anchor_text)
            siblings = [
                col for col in available
                if col not in placed and _column_family_key(col) == family
            ]
            if anchor_text in siblings:
                siblings.remove(anchor_text)
                family_order = [anchor_text] + sorted(
                    siblings,
                    key=lambda c: (str(c).casefold(), str(c)),
                )
            else:
                family_order = sorted(siblings, key=lambda c: (str(c).casefold(), str(c)))

        for col in family_order:
            if col not in placed:
                ordered.append(col)
                placed.add(col)

    for anchor in (pinned or []):
        emit_family(anchor)

    for anchor in (preferred_order or []):
        emit_family(anchor)

    tail_families: list[str] = []
    for col in available:
        if col in placed:
            continue
        family = _column_family_key(col)
        if family not in tail_families:
            tail_families.append(family)

    for family in tail_families:
        siblings = [
            col for col in available
            if col not in placed and _column_family_key(col) == family
        ]
        for col in sorted(siblings, key=lambda c: (str(c).casefold(), str(c))):
            ordered.append(col)
            placed.add(col)

    return ordered


def reorder_columns_by_family(
    df_in: pd.DataFrame,
    preferred_order: list[str] | None = None,
    pinned: list[str] | None = None,
) -> pd.DataFrame:
    """Return df_in with family-grouped column ordering applied."""
    if df_in.empty and not len(df_in.columns):
        return df_in.copy()
    ordered = order_columns_by_family(list(df_in.columns), preferred_order, pinned)
    return df_in.loc[:, ordered].copy()


def _model_family_for_category(category) -> str:
    text = str(category or "").strip()
    if text == "Activated carbon":
        return "AC"
    if text in {"Ion exchange resin", "Nonionic resin"}:
        return "Resin"
    if text in {"Cyclodextrin polymer", "Cyclodextrin polymers"}:
        return "CDP"
    return "Other"


def _model_include_is_true(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "false", "0", "no", "n"}:
        return False
    return True

def build_model_ready_equilibrium_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Keep equilibrium rows that are ready for downstream modeling.

    Standardized_Full remains the complete audit trail with raw reported
    parameters and conversion intermediates. Only model-blocking rows are
    excluded here; non-blocking warning/review signals are kept for modeling
    and surfaced in Equilibrium_Warning_Review.
    """
    equilibrium = build_equilibrium_data(df_in)
    if equilibrium.empty:
        available = list(equilibrium.columns) + [MODEL_FAMILY_COL]
        columns = _existing_unique(MODEL_READY_EQUILIBRIUM_COLUMNS, available)
        return pd.DataFrame(columns=columns)
    out = equilibrium.copy()
    if MODEL_INCLUDE_COL not in out.columns:
        out.insert(0, MODEL_INCLUDE_COL, True)
    if MODEL_EXCLUDE_REASON_COL not in out.columns:
        insert_at = 1 if MODEL_INCLUDE_COL in out.columns else 0
        out.insert(insert_at, MODEL_EXCLUDE_REASON_COL, "")
    categories = out.get("adsorbent_category", pd.Series("", index=out.index))
    out[MODEL_FAMILY_COL] = categories.map(_model_family_for_category)

    blocking_labels = out.apply(_model_blocking_reason_labels, axis=1)
    blocking_mask = blocking_labels.apply(bool)
    if blocking_mask.any():
        out.loc[blocking_mask, MODEL_INCLUDE_COL] = False
        out.loc[blocking_mask, MODEL_EXCLUDE_REASON_COL] = blocking_labels.loc[
            blocking_mask
        ].apply(lambda labels: "; ".join(labels))
    ready = out.loc[out[MODEL_INCLUDE_COL].apply(_model_include_is_true)].copy()
    return project_model_ready_columns(ready)


def _review_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "false", "0", "no"}:
        return ""
    return text


BENIGN_REVIEW_REASON_PATTERNS = (
    re.compile(r"^Freundlich_equation_form corrected from\b", flags=re.I),
    re.compile(r"^Freundlich_KF_unit inferred from consistent units\b", flags=re.I),
    re.compile(r"^Freundlich_KF_unit filled from unit rescue\b", flags=re.I),
)

SUPPORTED_FREUNDLICH_FORM_SAFETY_FILL_PATTERN = re.compile(
    r"^Freundlich_equation_form filled by post-processing safety guard "
    r"from\b.*:\s*KF-unit evidence\s*\((?:n|1/n)="
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\)\s*$",
    flags=re.I | re.S,
)

BENIGN_WARNING_PATTERNS = (
    re.compile(r"^Duplicate (?:performance_id|adsorbent_id) within study\b", flags=re.I),
    re.compile(r"^coalesced fields$", flags=re.I),
    re.compile(r"^\w+_unit filled from same extraction output\b", flags=re.I),
    re.compile(r"^\w+_unit filled from study-level extracted records\b", flags=re.I),
    re.compile(r"^Freundlich_KF_unit filled from unit rescue\b", flags=re.I),
)


def _split_review_fragments(text: str):
    return [part.strip() for part in re.split(r";\s*", text or "") if part.strip()]


def _is_benign_review_fragment(col: str, text: str) -> bool:
    patterns = ()
    if col == "review_reason":
        patterns = BENIGN_REVIEW_REASON_PATTERNS
    elif col == "warning_message":
        patterns = BENIGN_WARNING_PATTERNS
    return any(pattern.search(text) for pattern in patterns)


def _is_supported_freundlich_form_safety_fill(col: str, text: str) -> bool:
    """Return true only for a guard fill supported by unambiguous KF-unit evidence."""
    return col == "review_reason" and bool(
        SUPPORTED_FREUNDLICH_FORM_SAFETY_FILL_PATTERN.search(text or "")
    )


def _filtered_review_signal_text(col: str, value) -> str:
    text = _review_value(value)
    if not text:
        return ""
    if col not in {"review_reason", "warning_message"}:
        return text
    if _is_supported_freundlich_form_safety_fill(col, text):
        return ""
    parts = [
        part for part in _split_review_fragments(text)
        if not _is_benign_review_fragment(col, part)
    ]
    return "; ".join(parts)


def _row_has_substantive_review_signal(row, signal_columns=REVIEW_SIGNAL_COLUMNS) -> bool:
    for col in signal_columns:
        if col == "require_review" or col not in row.index:
            continue
        if _filtered_review_signal_text(col, row.get(col)):
            return True
    return False


def _row_has_only_benign_trace_review(row) -> bool:
    saw_benign_trace = False
    for col in ("review_reason", "warning_message"):
        if col not in row.index:
            continue
        raw = _review_value(row.get(col))
        if not raw:
            continue
        if _is_supported_freundlich_form_safety_fill(col, raw):
            saw_benign_trace = True
            continue
        filtered = _filtered_review_signal_text(col, raw)
        if filtered:
            return False
        if any(_is_benign_review_fragment(col, part) for part in _split_review_fragments(raw)):
            saw_benign_trace = True
    return saw_benign_trace


def _review_signal_text(row, col: str, signal_columns=REVIEW_SIGNAL_COLUMNS) -> str:
    if col == "require_review":
        raw = _review_value(row.get(col))
        if not raw:
            return ""
        if _row_has_substantive_review_signal(row, signal_columns):
            return raw
        if _row_has_only_benign_trace_review(row):
            return ""
        return raw
    return _filtered_review_signal_text(col, row.get(col))


def _review_message_parts(row, signal_columns=REVIEW_SIGNAL_COLUMNS) -> list[str]:
    parts = []
    for col in signal_columns:
        if col not in row.index:
            continue
        text = _review_signal_text(row, col, signal_columns)
        if text:
            parts.append(f"{col}: {text}")
    return parts


def _review_message(row, signal_columns=REVIEW_SIGNAL_COLUMNS) -> str:
    return "; ".join(_review_message_parts(row, signal_columns))


def _append_unique_text(parts: list[str], value: str) -> None:
    text = re.sub(r"\s+", " ", str(value or "")).strip("; ")
    if text and text not in parts:
        parts.append(text)


def _short_status_reason(text: str) -> str:
    """Render source-level diagnostics in a short, consistent workbook form."""
    replacements = {
        "PFAS_C0_value_mg/L": "PFAS C0",
        "Adsorbent_dosage_value_mg/L": "adsorbent dosage",
        "Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)": "Freundlich KF",
        "Freundlich_exponent_value": "Freundlich exponent",
        "Langmuir_Qm_value_mg/g": "Langmuir Qm",
        "Langmuir_KL_value_L/mg": "Langmuir KL",
    }
    out = re.sub(r"\s+", " ", str(text or "")).strip("; ")
    for source, label in replacements.items():
        out = out.replace(source, label)
    return out[:240]


def _classify_normalization_fragment(fragment: str) -> tuple[str, str]:
    """Classify one legacy normalization message without inventing missing data."""
    text = _short_status_reason(fragment)
    lower = text.casefold()

    unrecognized = re.match(
        r"Ionic strength skipped unrecognized inorganic species:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if unrecognized:
        return "review_required", "Unsupported ionic-strength species: " + unrecognized.group(1)

    charge_imbalance = re.match(
        r"Ionic strength skipped charge-imbalanced inorganic ions:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if charge_imbalance:
        return "not_computable", "Ionic strength not computable: charge-imbalanced ions (" + charge_imbalance.group(1) + ")"

    ambiguous = re.match(
        r"Ionic strength skipped ambiguous inorganic concentration:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if ambiguous:
        return "not_computable", "Ionic strength not computable: concentration is ambiguous (" + ambiguous.group(1) + ")"

    missing_molar = re.match(
        r"Ionic strength missing molar concentration for inorganic species:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if missing_molar:
        return "not_computable", "Ionic strength not computable: molar concentration missing (" + missing_molar.group(1) + ")"

    if "ionic_strength unit not recognized" in lower or "ionic strength unit not recognized" in lower:
        return "review_required", "Unsupported ionic-strength unit: " + text.split(":", 1)[-1].strip()
    if "ionic_strength unit missing" in lower or "ionic strength unit missing" in lower:
        return "not_computable", "Ionic strength not computable: unit missing"

    if text.startswith("Removal→Kd missing inputs:"):
        return "not_computable", "Kd not computable: missing " + text.split(":", 1)[1].strip()
    if text == "Removal endpoint invalid because PFAS C0 is zero: Kd, Ce, and Qe not calculated":
        return (
            "not_computable",
            "Removal reported with PFAS C0 = 0: Kd, Ce, and Qe not calculated",
        )
    if re.match(
        r"^(?:Qe|Removal|Langmuir|Freundlich(?:_KF)?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "not_computable", "Kd not computable: " + text
    if text.startswith("Ce_final unavailable for Kd_final_source "):
        return "not_computable", "Final Ce not computable for selected Kd source " + text.rsplit(" ", 1)[-1]
    if text.startswith("Kd_final not determined:"):
        detail = text.split(":", 1)[1].strip()
        if any(token in detail.casefold() for token in ("unrecognized", "could not standardize", "unit dimensionality mismatch")):
            return "review_required", "Review Kd conversion: " + detail
        return "not_computable", "Kd not computable: " + detail

    if any(
        token in lower
        for token in (
            "unrecognized",
            "not recognized",
            "unsupported",
            "unit dimensionality mismatch",
            "invalid combined fraction",
            "not in lookup",
            "inconsistent range/list",
            "non-scalar or invalid",
        )
    ):
        return "review_required", "Review normalization: " + text

    # Unknown normalization failures stay in the review queue rather than
    # silently being presented as an expected inability to calculate.
    return "review_required", "Review normalization: " + text


def _classify_external_review_fragment(column: str, fragment: str) -> tuple[str, str]:
    """Classify merge/extraction diagnostics that are not normalization steps."""
    text = _short_status_reason(fragment)
    if "No matching adsorbent property record for adsorbent_id=" in text:
        return "review_required", text.replace(
            "No matching adsorbent property record for adsorbent_id=",
            "Adsorbent properties not matched: ",
        )
    if text == "Performance row lacks adsorbent_id":
        return "review_required", "Adsorbent ID missing"
    if text.startswith("Ingest error:"):
        return "review_required", "Review extraction: " + text.removeprefix("Ingest error:").strip()
    if text.startswith("Freundlich_equation_form"):
        return "review_required", "Freundlich equation form requires review"
    return "review_required", "Review source data: " + text


def _truthy(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def materialize_record_status(df_in: pd.DataFrame) -> None:
    """Collapse legacy message channels into a concise record status schema.

    ``not_computable`` records report an absent/insufficient input without
    treating the absence as an error.  ``review_required`` is reserved for a
    parser, source-data, or normalization issue that a person can assess or
    fix.  Informational conversion history remains in ``normalization_trace``.
    """
    for column in LEGACY_MESSAGE_COLUMNS:
        if column not in df_in.columns:
            df_in[column] = ""

    statuses: list[str] = []
    reasons: list[str] = []
    traces: list[str] = []
    outcomes: list[str] = []

    for _, row in df_in.iterrows():
        trace_parts: list[str] = []
        not_computable: list[str] = []
        review_required: list[str] = []

        # Preserve trace provenance from extraction, merge, and normalization.
        _append_unique_text(trace_parts, _review_value(row.get(NORMALIZATION_TRACE_COLUMN)))

        for fragment in _split_review_fragments(_review_value(row.get("normalization_issues"))):
            category, summary = _classify_normalization_fragment(fragment)
            target = review_required if category == "review_required" else not_computable
            _append_unique_text(target, summary)

        for column in ("review_reason", "warning_message"):
            raw_message = _review_value(row.get(column))
            if _is_supported_freundlich_form_safety_fill(column, raw_message):
                _append_unique_text(trace_parts, raw_message)
                continue
            for fragment in _split_review_fragments(raw_message):
                if _is_benign_review_fragment(column, fragment):
                    _append_unique_text(trace_parts, fragment)
                    continue
                _category, summary = _classify_external_review_fragment(column, fragment)
                _append_unique_text(review_required, summary)

        review_requested = _truthy(row.get("require_review"))
        if review_requested and not review_required and not _row_has_only_benign_trace_review(row):
            _append_unique_text(review_required, "Manual review requested")

        if review_required:
            statuses.append("review_required")
            reasons.append("; ".join(review_required + not_computable))
        elif not_computable:
            statuses.append("not_computable")
            reasons.append("; ".join(not_computable))
        else:
            statuses.append("clean")
            reasons.append("")

        decision = _review_value(row.get("human_review_decision")).casefold()
        if decision in {"reliable", "unreliable"}:
            outcomes.append(decision)
        elif _truthy(row.get("human_review_unreliable_flag")):
            outcomes.append("unreliable")
        else:
            outcomes.append("")
        traces.append("; ".join(trace_parts))

    df_in[RECORD_STATUS_COLUMN] = statuses
    df_in[STATUS_REASON_COLUMN] = reasons
    df_in[NORMALIZATION_TRACE_COLUMN] = traces
    df_in[REVIEW_OUTCOME_COLUMN] = outcomes
    df_in.drop(columns=[column for column in LEGACY_MESSAGE_COLUMNS if column in df_in.columns], inplace=True)


def _add_reason_label(labels: list[str], label: str) -> None:
    if label and label not in labels:
        labels.append(label)


def _safe_reason_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.upper() or "FIELD"


def _normalized_review_reason_labels(message: str) -> list[str]:
    text = str(message or "")
    labels: list[str] = []

    # Kd/logKd derivation failures. These become blocking only if no valid
    # final Kd/logKd exists for the row.
    if re.search(r"(Removal→Kd missing inputs:|Removal missing ).*PFAS_C0_value_mg/L", text):
        _add_reason_label(labels, "Removal→Kd_missing C0")
    if re.search(r"(Removal→Kd missing inputs:|Removal missing ).*Adsorbent_dosage_value_mg/L", text):
        _add_reason_label(labels, "Removal→Kd_missing dosage")
    if "Removal→Kd invalid C0" in text or "Removal invalid C0" in text:
        _add_reason_label(labels, "Removal→Kd_invalid C0")
    if (
        "Removal→Kd invalid adsorbent dosage" in text
        or "Removal→Kd invalid dosage" in text
        or "Removal invalid dosage" in text
    ):
        _add_reason_label(labels, "Removal→Kd_invalid dosage")
    if "Removal→Kd Ce ≤ 0" in text or "Removal Ce ≤ 0" in text:
        _add_reason_label(labels, "Removal→Kd_Ce<=0")
    if "Removal→Kd negative Qe" in text:
        _add_reason_label(labels, "Removal→Kd_negative Qe")
    if "Removal could not compute" in text:
        _add_reason_label(labels, "Removal→Kd_could not compute")

    if re.search(r"(Qe missing|Qe→Kd missing inputs:).*PFAS_C0_value_mg/L", text):
        _add_reason_label(labels, "Qe→Kd_missing C0")
    if re.search(r"(Qe missing|Qe→Kd missing inputs:).*Adsorbent_dosage_value_mg/L", text):
        _add_reason_label(labels, "Qe→Kd_missing dosage")
    if "Qe→Kd Ce ≤ 0" in text or "Qe Ce ≤ 0" in text:
        _add_reason_label(labels, "Qe→Kd_Ce<=0")
    if "Qe→Kd invalid C0" in text or "Qe invalid C0" in text:
        _add_reason_label(labels, "Qe→Kd_invalid C0")
    if "Qe→Kd invalid dosage" in text or "Qe invalid dosage" in text:
        _add_reason_label(labels, "Qe→Kd_invalid dosage")
    if "Qe→Kd invalid Qe" in text or "Qe invalid Qe" in text:
        _add_reason_label(labels, "Qe→Kd_invalid Qe")
    if "Qe could not compute" in text:
        _add_reason_label(labels, "Qe→Kd_could not compute")

    if re.search(r"Langmuir missing .*Langmuir_Qm_value_mg/g", text):
        _add_reason_label(labels, "Langmuir→Kd_missing Qm")
    if re.search(r"Langmuir missing .*Langmuir_KL_value_L/mg", text):
        _add_reason_label(labels, "Langmuir→Kd_missing KL")
    if re.search(r"Langmuir missing .*PFAS_C0_value_mg/L", text):
        _add_reason_label(labels, "Langmuir→Kd_missing C0")
    if re.search(r"Langmuir missing .*Adsorbent_dosage_value_mg/L", text):
        _add_reason_label(labels, "Langmuir→Kd_missing dosage")
    if "Ce_from_Langmuir invalid C0" in text or re.search(r"Langmuir invalid .*C0", text):
        _add_reason_label(labels, "Langmuir→Kd_invalid C0")
    if "Ce_from_Langmuir invalid dosage" in text or re.search(r"Langmuir invalid .*dosage", text):
        _add_reason_label(labels, "Langmuir→Kd_invalid dosage")
    if "Ce_from_Langmuir invalid Qm" in text or re.search(r"Langmuir invalid .*Qm", text):
        _add_reason_label(labels, "Langmuir→Kd_invalid Qm")
    if "Ce_from_Langmuir invalid KL" in text or re.search(r"Langmuir invalid .*KL", text):
        _add_reason_label(labels, "Langmuir→Kd_invalid KL")
    if "Ce_from_Langmuir invalid inputs" in text:
        _add_reason_label(labels, "Langmuir→Kd_invalid inputs")
    if "Ce_from_Langmuir no real solution" in text or "Ce_from_Langmuir no physical root" in text:
        _add_reason_label(labels, "Langmuir→Kd_no physical root")
    if "Kd_from_Langmuir negative Qe from Ce" in text:
        _add_reason_label(labels, "Langmuir→Kd_negative Qe")
    if "Langmuir reported fit R2 missing" in text:
        _add_reason_label(labels, "Langmuir→Kd_R2 missing")
    if f"Langmuir reported fit R2 below {ISOTHERM_R2_MINIMUM:.2f}" in text:
        _add_reason_label(labels, "Langmuir→Kd_R2 below threshold")

    freundlich_kf_conversion_failed = any(
        marker in text
        for marker in (
            "Freundlich_KF [",
            "Freundlich_KF unit missing",
            "KF unit appears reversed",
            "KF unit lacks uptake-per-adsorbent-mass term",
            "amount-based KF units",
            "Unrecognized Freundlich_KF unit",
        )
    )
    if re.search(r"Freundlich missing .*Freundlich_KF_value_", text) and not freundlich_kf_conversion_failed:
        _add_reason_label(labels, "Freundlich→Kd_missing KF")
    if re.search(r"Freundlich missing .*Freundlich_exponent_value", text):
        _add_reason_label(labels, "Freundlich→Kd_missing exponent")
    if re.search(r"Freundlich missing .*PFAS_C0_value_mg/L", text):
        _add_reason_label(labels, "Freundlich→Kd_missing C0")
    if re.search(r"Freundlich missing .*Adsorbent_dosage_value_mg/L", text):
        _add_reason_label(labels, "Freundlich→Kd_missing dosage")
    if "Freundlich 1/n must be > 0" in text:
        _add_reason_label(labels, "Freundlich→Kd_invalid 1/n")
    if "Freundlich KF could not standardize" in text:
        _add_reason_label(labels, "Freundlich→Kd_KF standardization failed")
    if "Ce_from_Freundlich invalid C0" in text or re.search(r"Freundlich invalid .*C0", text):
        _add_reason_label(labels, "Freundlich→Kd_invalid C0")
    if "Ce_from_Freundlich invalid dosage" in text or re.search(r"Freundlich invalid .*dosage", text):
        _add_reason_label(labels, "Freundlich→Kd_invalid dosage")
    if "Ce_from_Freundlich invalid KF" in text or re.search(r"Freundlich invalid .*KF", text):
        _add_reason_label(labels, "Freundlich→Kd_invalid KF")
    if "Ce_from_Freundlich invalid 1/n" in text or re.search(r"Freundlich invalid .*1/n", text):
        _add_reason_label(labels, "Freundlich→Kd_invalid 1/n")
    if "Ce_from_Freundlich invalid inputs" in text:
        _add_reason_label(labels, "Freundlich→Kd_invalid inputs")
    if "Ce_from_Freundlich root-finding failed" in text:
        _add_reason_label(labels, "Freundlich→Kd_root failed")
    if "Kd_from_Freundlich negative Qe from Ce" in text:
        _add_reason_label(labels, "Freundlich→Kd_negative Qe")
    if "Freundlich reported fit R2 missing" in text:
        _add_reason_label(labels, "Freundlich→Kd_R2 missing")
    if f"Freundlich reported fit R2 below {ISOTHERM_R2_MINIMUM:.2f}" in text:
        _add_reason_label(labels, "Freundlich→Kd_R2 below threshold")

    if re.search(r"Freundlich_KF \[[^\]]*\] 1/n=[^;]+ must be > 0", text):
        _add_reason_label(labels, "Freundlich→Kd_invalid 1/n")
    if re.search(r"Freundlich_KF \[[^\]]*\] 1/n missing: cannot convert KF", text):
        _add_reason_label(labels, "Freundlich→Kd_missing 1/n")
    if "Freundlich_KF unit missing" in text:
        _add_reason_label(labels, "Freundlich→Kd_missing KF unit")
    if "Freundlich_KF unit is distribution-coefficient-like" in text:
        _add_reason_label(labels, "Freundlich→Kd_distribution-like KF unit")
    if "KF unit appears reversed" in text:
        _add_reason_label(labels, "Freundlich→Kd_reversed KF unit")
    if "KF unit lacks uptake-per-adsorbent-mass term" in text:
        _add_reason_label(labels, "Freundlich→Kd_missing uptake basis")
    if "amount-based KF units" in text:
        _add_reason_label(labels, "Freundlich→Kd_missing PFAS MW")
    if "Unrecognized Freundlich_KF unit" in text:
        _add_reason_label(labels, "Freundlich→Kd_unrecognized KF unit")

    if "Ce_from_Kd ≤ 0" in text:
        _add_reason_label(labels, "Kd→Ce_Ce<=0")
    if "Ce_from_Kd invalid C0" in text:
        _add_reason_label(labels, "Kd→Ce_invalid C0")
    if "Ce_from_Kd invalid dosage" in text:
        _add_reason_label(labels, "Kd→Ce_invalid dosage")
    if "Ce_from_Kd invalid Kd" in text:
        _add_reason_label(labels, "Kd→Ce_invalid Kd")
    if "Ce_from_Kd invalid dose/Kd" in text:
        _add_reason_label(labels, "Kd→Ce_invalid dose/Kd")
    if "Ce_from_Kd non-positive denominator" in text:
        _add_reason_label(labels, "Kd→Ce_non-positive denominator")

    # Parameter-level warnings that should not block a model-ready row.
    if "Ionic strength skipped charge-imbalanced inorganic ions:" in text:
        _add_reason_label(labels, "IONIC_STRENGTH_CHARGE_IMBALANCE")
    if "Ionic strength skipped unrecognized inorganic species:" in text:
        _add_reason_label(labels, "IONIC_STRENGTH_UNRECOGNIZED_SPECIES")
    if "Ionic strength skipped ambiguous inorganic concentration:" in text:
        _add_reason_label(labels, "IONIC_STRENGTH_AMBIGUOUS_CONCENTRATION")
    if "Ionic strength missing molar concentration for inorganic species:" in text:
        _add_reason_label(labels, "IONIC_STRENGTH_MISSING_MOLAR_CONC")
    if "Ionic_strength unit missing" in text:
        _add_reason_label(labels, "IONIC_STRENGTH_UNIT_MISSING")

    for param in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*) \[[^\]]+\] Unit dimensionality mismatch", text):
        _add_reason_label(labels, f"{_safe_reason_token(param)}_UNIT_DIMENSION_MISMATCH")
    for param in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*) \[[^\]]+\] PFAS molecular weight is missing", text):
        _add_reason_label(labels, f"{_safe_reason_token(param)}_MISSING_PFAS_MW")
    for param in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*) \[[^\]]+\] Surface-area basis; not converted", text):
        _add_reason_label(labels, f"{_safe_reason_token(param)}_SURFACE_AREA_BASIS")

    if (
        "Nominal_grain_size mesh size(s) not in lookup:" in text
        or re.search(
            r"Nominal_grain_size \[mesh\] Mesh "
            r"(?:endpoints|[-0-9.eE+]+) not in lookup",
            text,
        )
    ):
        _add_reason_label(labels, "NOMINAL_GRAIN_SIZE_MESH_NOT_IN_LOOKUP")
    if re.search(r"Adsorbent_dosage \[[^\]]+\] Solution volume missing", text):
        _add_reason_label(labels, "Adsorbent_dosage→mg/L_missing volume")
    if "No matching adsorbent property record" in text:
        _add_reason_label(labels, "ADSORBENT_PROPERTIES_NO_MATCH")
    if "Performance row lacks adsorbent_id" in text:
        _add_reason_label(labels, "ADSORBENT_ID_MISSING")
    if "ambiguous isotherm design: both PFAS_C0 and adsorbent dosage are multi-valued" in text:
        _add_reason_label(labels, "ISOTHERM_DESIGN_AMBIGUOUS_C0_AND_DOSAGE")

    if not labels and text:
        _add_reason_label(labels, "MANUAL_REVIEW_UNCLASSIFIED")
    return labels


def _has_model_text(value) -> bool:
    text = _review_value(value)
    if not text:
        return False
    return text.strip().lower() not in {
        "na",
        "n/a",
        "nan",
        "none",
        "not reported",
        "not applicable",
        "unclassified",
    }


def _model_blocking_reason_labels(row) -> list[str]:
    labels: list[str] = []
    if not _has_model_text(row.get("PFAS_name")):
        _add_reason_label(labels, "PFAS_missing name")
    if not _has_model_text(row.get("adsorbent_id")):
        _add_reason_label(labels, "Adsorbent_missing id")
    # A removal percentage is physically inconsistent when the reported
    # initial PFAS concentration is exactly zero.  Keep the row in the audit
    # trail, but prevent a removal-derived Kd from entering Equilibrium_Data.
    c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
    if c0 is None:
        c0 = _coerce_number(row.get("PFAS_C0_value"))
    if c0 == 0 and _has_removal_result(row):
        _add_reason_label(labels, "Removal_reported_with_zero_PFAS_C0")
    # A study-local adsorbent ID is sufficient for model eligibility.  Missing
    # canonical names indicate that property enrichment could not be matched;
    # retain that as a warning rather than excluding an otherwise valid record.

    kd_final = _coerce_number(row.get("Kd_final_L/g"))
    log_kd = _coerce_number(row.get("Kd_final_log10(L/g)"))
    source = _review_value(row.get("Kd_final_source"))
    if kd_final is None or log_kd is None:
        for label in _normalized_review_reason_labels(_review_message(row)):
            if (
                "→Kd_" in label
                or label.startswith("Kd→Ce_")
            ):
                _add_reason_label(labels, label)
        if not any(
            "→Kd_" in label
            or label.startswith("Kd→Ce_")
            for label in labels
        ):
            _add_reason_label(labels, "Kd_final_missing")
    elif source not in EQUILIBRIUM_ALLOWED_KD_FINAL_SOURCES:
        _add_reason_label(labels, "Kd_final_invalid source")
    return labels


def _review_rows_with_labels(
    df_in: pd.DataFrame,
    keep_labels_by_index: dict,
    count_col: str,
    label_col: str,
) -> pd.DataFrame:
    context_cols = list(df_in.columns)
    output_cols = [count_col, label_col, "review_message"] + context_cols
    if not keep_labels_by_index:
        return pd.DataFrame(columns=output_cols)

    keep_index = list(keep_labels_by_index.keys())
    review = df_in.loc[keep_index, context_cols].copy()
    label_series = pd.Series(
        {idx: "; ".join(labels) for idx, labels in keep_labels_by_index.items()},
        dtype="object",
    )
    count_series = pd.Series(
        {idx: len(labels) for idx, labels in keep_labels_by_index.items()},
        dtype="object",
    )
    message_series = pd.Series(
        {
            idx: (_review_message(df_in.loc[idx]) or "; ".join(labels))
            for idx, labels in keep_labels_by_index.items()
        },
        dtype="object",
    )
    review.insert(0, "review_message", message_series.loc[keep_index])
    review.insert(0, label_col, label_series.loc[keep_index])
    review.insert(0, count_col, count_series.loc[keep_index])
    return review

def build_model_blocking_review(df_in: pd.DataFrame) -> pd.DataFrame:
    equilibrium = build_equilibrium_data(df_in)
    labels_by_index = {}
    for idx, row in equilibrium.iterrows():
        labels = _model_blocking_reason_labels(row)
        if labels:
            labels_by_index[idx] = labels
    return _review_rows_with_labels(
        equilibrium,
        labels_by_index,
        "blocking_reason_count",
        "blocking_reason_labels",
    )

def build_equilibrium_warning_review(df_in: pd.DataFrame) -> pd.DataFrame:
    equilibrium = build_equilibrium_data(df_in)
    labels_by_index = {}
    for idx, row in equilibrium.iterrows():
        if _model_blocking_reason_labels(row):
            continue
        message = _review_message(row)
        if not message:
            continue
        labels = _normalized_review_reason_labels(message)
        if labels:
            labels_by_index[idx] = labels
    return _review_rows_with_labels(
        equilibrium,
        labels_by_index,
        "warning_reason_count",
        "warning_reason_labels",
    )


REVIEW_REQUIRED_COLUMNS = [
    "standardized_row_id",
    "source_row_index",
    "extraction_dataset",
    "study_no",
    "DOI",
    "performance_id",
    "adsorbent_id",
    "PFAS_name",
    RECORD_STATUS_COLUMN,
    STATUS_REASON_COLUMN,
    REVIEW_OUTCOME_COLUMN,
]


def build_review_required(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return the compact queue for records that need a human decision."""
    columns = _existing_unique(REVIEW_REQUIRED_COLUMNS, df_in.columns)
    if RECORD_STATUS_COLUMN not in df_in.columns:
        return pd.DataFrame(columns=columns)
    review = df_in.loc[df_in[RECORD_STATUS_COLUMN].eq("review_required"), columns].copy()
    return review.reset_index(drop=True)


def _is_scalar_numeric_cell(value) -> bool:
    """
    Strict scalar-numeric check for ML operating-condition fields.

    This intentionally rejects lists, ranges, and uncertainty expressions,
    because those are not single operating conditions.
    """
    if value in ("", None):
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True

    text = str(value).strip()
    if not text:
        return False
    if "&&" in text or ";" in text:
        return False
    if "," in text:
        return False
    if re.search(r"\s(?:to)\s|[–—]|±|\+/-|\+⁄-", text, flags=re.I):
        return False
    return re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text) is not None


def _row_has_numeric_in_any(row, columns) -> bool:
    for col in columns:
        if _coerce_number(row.get(col)) is not None:
            return True
    return False


def _reported_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _has_reported_value(value) -> bool:
    if _coerce_number(value) is not None:
        return True
    text = _reported_text(value)
    if not text:
        return False
    return text.lower() not in {"na", "n/a", "nan", "none", "not reported", "not applicable", "unclassified"}


def _row_has_reported_in_any(row, columns) -> bool:
    for col in columns:
        if _has_reported_value(row.get(col)):
            return True
    return False


def _is_kinetic_row(row) -> bool:
    test_mode = str(row.get("Test_mode", "")).strip().lower()
    if test_mode.startswith("kinet"):
        return True
    return _row_has_reported_in_any(row, KINETIC_INDICATOR_COLUMNS)


def _has_removal_result(row) -> bool:
    return _has_reported_value(row.get("Removal_rate")) or _coerce_number(row.get("Removal_rate_avg_fraction")) is not None


def _has_equilibrium_endpoint(row) -> bool:
    kd_source = _reported_text(row.get("Kd_final_source"))
    if kd_source in EQUILIBRIUM_ENDPOINT_KD_FINAL_SOURCES:
        return True
    return _row_has_reported_in_any(row, EQUILIBRIUM_ENDPOINT_INDICATOR_COLUMNS)


def _has_unclear_endpoint_signal(row) -> bool:
    test_mode = _reported_text(row.get("Test_mode")).lower()
    return (
        test_mode.startswith("iso")
        or _row_has_reported_in_any(row, PERFORMANCE_INDICATOR_COLUMNS)
    )


def _equilibrium_reached_code(value) -> str:
    text = _reported_text(value).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text in {"not sure", "unclear", "unknown", "unsure", "maybe"}:
        return "not_sure"
    if re.match(r"^yes\b", text) or text in {"y", "true", "reached", "equilibrium", "equilibrium reached"}:
        return "yes"
    if re.match(r"^no\b", text) or text in {"n", "false", "not reached", "not equilibrium", "non equilibrium"}:
        return "no"
    return ""


def _scalar_contact_time_hours(row):
    value = row.get("Contact_time_(h)")
    if _is_scalar_numeric_cell(value):
        return _coerce_number(value)
    return None


def _group_value(value) -> str:
    if _is_scalar_numeric_cell(value):
        num = _coerce_number(value)
        if num is not None:
            return f"{num:.12g}"
    text = _reported_text(value).lower()
    return re.sub(r"\s+", " ", text)


def _removal_time_group_columns(df_in: pd.DataFrame) -> list[str]:
    columns = list(REMOVAL_TIME_GROUP_COLUMNS)
    columns.extend([c for c in df_in.columns if str(c).startswith("Temperature_(")])
    return _existing_unique(columns, df_in.columns)


def _assign_removal_time_assumptions(
    df_in: pd.DataFrame,
    determinations: pd.Series,
    pending_mask: pd.Series,
) -> None:
    pending_indices = [idx for idx in df_in.index if bool(pending_mask.loc[idx])]
    if not pending_indices:
        return

    group_columns = _removal_time_group_columns(df_in)
    groups = defaultdict(list)
    scalar_times = {}

    for idx in pending_indices:
        row = df_in.loc[idx]
        scalar_times[idx] = _scalar_contact_time_hours(row)
        key = tuple(_group_value(row.get(c)) for c in group_columns) if group_columns else (idx,)
        groups[key].append(idx)

    for group_indices in groups.values():
        scalar_indices = [idx for idx in group_indices if scalar_times.get(idx) is not None]
        nonscalar_indices = [idx for idx in group_indices if scalar_times.get(idx) is None]

        for idx in nonscalar_indices:
            determinations.at[idx] = "removal_equilibrium_unclear"

        if not scalar_indices:
            continue

        unique_times = sorted({round(float(scalar_times[idx]), 12) for idx in scalar_indices})
        if len(unique_times) == 1:
            for idx in scalar_indices:
                determinations.at[idx] = "removal_single_timepoint_assumed"
            continue

        last_time = unique_times[-1]
        for idx in scalar_indices:
            if abs(float(scalar_times[idx]) - last_time) <= 1e-9:
                determinations.at[idx] = "removal_last_timepoint_assumed"
            else:
                determinations.at[idx] = "removal_intermediate_timepoint"


def assign_equilibrium_determination(df_in: pd.DataFrame) -> None:
    """
    Add a compact audit label explaining why a row is routed to an
    equilibrium, non-equilibrium, or unknown output sheet.
    """
    _ensure_columns(df_in, [EQUILIBRIUM_DETERMINATION_COLUMN], "")
    if df_in.empty:
        return

    determinations = pd.Series("", index=df_in.index, dtype="object")

    kinetic = df_in.apply(_is_kinetic_row, axis=1)
    determinations.loc[kinetic] = "kinetic_model_reported"

    endpoints = determinations.eq("") & df_in.apply(_has_equilibrium_endpoint, axis=1)
    determinations.loc[endpoints] = "equilibrium_endpoint_reported"

    removal = determinations.eq("") & df_in.apply(_has_removal_result, axis=1)
    for idx, row in df_in.loc[removal].iterrows():
        reached = _equilibrium_reached_code(row.get("Equilibrium_reached"))
        if reached == "yes":
            determinations.at[idx] = "removal_explicit_equilibrium"
        elif reached == "no":
            determinations.at[idx] = "removal_explicit_non_equilibrium"

    pending_removal = removal & determinations.eq("")
    _assign_removal_time_assumptions(df_in, determinations, pending_removal)

    unclear_endpoint = determinations.eq("") & df_in.apply(_has_unclear_endpoint_signal, axis=1)
    determinations.loc[unclear_endpoint] = "endpoint_unclear"

    df_in[EQUILIBRIUM_DETERMINATION_COLUMN] = determinations


def _determination_mask(df_in: pd.DataFrame, labels: set[str]) -> pd.Series:
    if EQUILIBRIUM_DETERMINATION_COLUMN not in df_in.columns:
        return pd.Series(False, index=df_in.index)
    return df_in[EQUILIBRIUM_DETERMINATION_COLUMN].astype(str).str.strip().isin(labels)


def build_equilibrium_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """Rows determined to represent equilibrium adsorption results."""
    if df_in.empty:
        return df_in.copy()
    return df_in.loc[_determination_mask(df_in, EQUILIBRIUM_DETERMINATIONS)].copy()


def build_non_equilibrium_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """Rows determined to represent kinetic or pre-equilibrium results."""
    if df_in.empty:
        return df_in.copy()
    return df_in.loc[_determination_mask(df_in, NON_EQUILIBRIUM_DETERMINATIONS)].copy()


def build_equilibrium_status_review(df_in: pd.DataFrame) -> pd.DataFrame:
    """Rows with in-scope performance signals but unclear equilibrium status."""
    if df_in.empty:
        return df_in.copy()
    return df_in.loc[_determination_mask(df_in, UNKNOWN_EQUILIBRIUM_DETERMINATIONS)].copy()


def derive_endpoint_columns(df: pd.DataFrame) -> None:
    """Apply endpoint derivation stages in the historical order."""
    compute_kd_from_removal(df)
    compute_kd_from_qe(df)
    select_final_ce(df)
    compute_ce_from_kd(df)
    compute_ce_from_freundlich(df)
    compute_ce_from_langmuir(df)
    select_final_ce(df)
    compute_kd_from_isotherm_ce(df)
    select_final_kd(df)
    align_final_ce_to_kd_source(df)
    compute_apparent_removal_and_logkd_conversion_instability_flags(df)
    mark_not_applicable_adsorbent_fields(df)
    _add_log10_kd_columns(df)
    df.drop(columns=["Near_complete_removal"], inplace=True, errors="ignore")


MODEL_READY_RAW_TRACE_EXEMPTIONS = {"AC_raw_material"}

# Closed schema for the model-facing sheet. Standardized_Full remains the
# catch-all audit table for arbitrary extracted headers and conversion traces.
MODEL_READY_EQUILIBRIUM_COLUMNS = [
    MODEL_FAMILY_COL,
    "standardized_row_id",
    "source_row_index",
    "extraction_dataset",
    "study_no",
    "performance_id",
    "adsorbent_id",
    "adsorbent_identity_status",
    "adsorbent_identity_basis",
    "adsorbent_identity_key",
    "adsorbent_study_instance_key",
    "PFAS_name",
    "Second_Class",
    "Mw (g/mol)",
    "name_full",
    "name_commercial",
    "name_abbreviation",
    "adsorbent_category",
    "adsorbent_subcategory",
    "Test_mode",
    "Differentiating_Condition",
    "Equilibrium_reached",
    "equilibrium_reached",
    EQUILIBRIUM_DETERMINATION_COLUMN,
    "Kd_final_L/g",
    "Kd_final_log10(L/g)",
    "Kd_final_source",
    "Freundlich_exponent_greater_than_1",
    "apparent_removal_fraction",
    "apparent_removal_source",
    "logKd_conversion_spike_flag",
    "logKd_conversion_dip_flag",
    "low_Ce_detection_limit_flag",
    "concentration_difference_C0_minus_Ce_mg/L",
    "small_concentration_difference_flag",
    "low_removal_rate_flag",
    "high_apparent_removal_flag",
    "PFAS_C0_value_mg/L",
    "Adsorbent_dosage_value_mg/L",
    "pH",
    "Temperature_(°C)",
    "Contact_time_(h)",
    "Solution_volume_(mL)",
    "Mixing_speed_(rpm)",
    "Water_type",
    "organic_carbon_mg/L",
    "TDS_(mg/L)",
    "Ionic_strength_(mol/L)",
    "organic_matter_present",
    "organic_matter_class",
    "inorganic_matter_present",
    "contains_Na",
    "contains_K",
    "contains_Ca",
    "contains_Mg",
    "contains_Cl",
    "contains_HCO3",
    "contains_SO4",
    "contains_phosphate",
    "AC_raw_material",
    "AC_activation_method",
    "polymer_matrix",
    *FUNCTIONAL_GROUP_INDICATOR_COLUMNS.values(),
    "elemental_composition_method",
    *PORE_CLASS_INDICATOR_COLUMNS.values(),
    "average_pore_diameter_(angstrom)",
    "ssa_(m2/g)_avg",
    "phpzc",
    "Nominal_grain_size_value_mm",
    "porosity_total_value",
    "porosity_micro_value_avg",
    "porosity_meso_value_avg",
    "porosity_macro_value_avg",
    "pore_volume_total_value_cm3/g",
    "pore_volume_micro_value_avg_cm3/g",
    "pore_volume_meso_value_avg_cm3/g",
    "pore_volume_macro_value_avg_cm3/g",
    "element_C_value",
    "element_N_value",
    "element_O_value",
    ZETA_POTENTIAL_NEAR_PH_7_COLUMN,
    "ion_exchange_capacity_value_meq/g",
    "ion_exchange_capacity_value_meq/L",
    "iodine_number_mg/g",
]


def _is_model_ready_trace_column(column: object) -> bool:
    text = str(column)
    if text in MODEL_READY_RAW_TRACE_EXEMPTIONS:
        return False
    return bool(
        re.search(r"(?:^|_)raw(?:_|$)", text)
        or re.search(r"(?:^|_)context(?:_|$)", text)
    )


def project_model_ready_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return only model-facing columns for Equilibrium_Data.

    The raw paired zeta-potential fields and their converted pH-7 value are
    materialized in Standardized_Full. The fallback preserves this projection's
    behavior for isolated callers that have not run normalization.
    """
    model_ready = df_in.copy()
    if ZETA_POTENTIAL_NEAR_PH_7_COLUMN not in model_ready.columns:
        model_ready[ZETA_POTENTIAL_NEAR_PH_7_COLUMN] = zeta_potential_near_ph_7(df_in)
    columns = _existing_unique(MODEL_READY_EQUILIBRIUM_COLUMNS, model_ready.columns)
    return model_ready.loc[:, columns].copy()


REFERENCE_DOSE_PREFIX_COLUMNS = [
    "reference_dosage_target_mg/L",
    "reference_dosage_tolerance_mg/L",
    "reference_dose_basis",
    "reference_dose_method",
    "reference_dose_exact_flag",
    "reference_isotherm_R2",
    "experimental_dosage_delta_from_reference_mg/L",
    "search_dosage_mg/L",
    "search_Kd_L/g",
    "search_logKd_log10(L/g)",
    "search_Ce_mg/L",
    "search_removal_fraction",
]

REFERENCE_DOSE_ISOTHERM_COLUMNS = [
    "Langmuir_Qm_value_mg/g",
    "Langmuir_KL_value_L/mg",
    "Langmuir_R2",
    "Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)",
    "Freundlich_exponent_value",
    "Freundlich_R2",
]

REFERENCE_DOSE_CITATION_COLUMNS = [
    "DOI",
    "authors",
    "article_title",
    "source_title",
    "publication_year",
]


def _reference_dose_isotherm_candidate(
    row: pd.Series,
    source: str,
    reference_dosage_mg_l: float,
) -> dict[str, float | str] | None:
    """Return one fit-qualified endpoint recalculated at the reference dose."""
    c0 = _coerce_number(row.get("PFAS_C0_value_mg/L"))
    if c0 is None or not math.isfinite(c0) or c0 <= 0:
        return None

    r2 = _coerce_number(row.get(f"{source}_R2"))
    if r2 is None or not math.isfinite(r2) or r2 < ISOTHERM_R2_MINIMUM:
        return None

    if source == "Langmuir":
        qmax = _coerce_number(row.get("Langmuir_Qm_value_mg/g"))
        kl = _coerce_number(row.get("Langmuir_KL_value_L/mg"))
        if (
            qmax is None
            or kl is None
            or not math.isfinite(qmax)
            or not math.isfinite(kl)
            or qmax <= 0
            or kl <= 0
        ):
            return None

        def residual(ce: float) -> float:
            qe = qmax * kl * ce / (1.0 + kl * ce)
            return ce + (reference_dosage_mg_l / 1000.0) * qe - c0

        ce = _bisect_root(residual, 0.0, c0)
        if ce is None or not math.isfinite(ce) or ce <= 0:
            return None
        qe = qmax * kl * ce / (1.0 + kl * ce)
    elif source == "Freundlich":
        kf = _coerce_number(
            row.get("Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)")
        )
        exponent = _coerce_number(row.get("Freundlich_exponent_value"))
        if (
            kf is None
            or exponent is None
            or not math.isfinite(kf)
            or not math.isfinite(exponent)
            or kf <= 0
            or exponent <= 0
        ):
            return None

        def residual(ce: float) -> float:
            qe = kf * (ce ** exponent)
            return ce + (reference_dosage_mg_l / 1000.0) * qe - c0

        ce = _bisect_root(residual, 0.0, c0)
        if ce is None or not math.isfinite(ce) or ce <= 0:
            return None
        qe = kf * (ce ** exponent)
    else:
        return None

    kd = qe / ce
    if not math.isfinite(kd) or kd <= 0:
        return None
    removal = 1.0 - (ce / c0)
    return {
        "source": source,
        "r2": r2,
        "kd": kd,
        "log_kd": math.log10(kd),
        "ce": ce,
        "removal": removal,
    }


def _reference_dose_isotherm_result(
    row: pd.Series,
    reference_dosage_mg_l: float,
) -> dict[str, float | str] | None:
    """Select the final-source isotherm, else the best fit-qualified model."""
    candidates = [
        candidate
        for source in ("Langmuir", "Freundlich")
        if (
            candidate := _reference_dose_isotherm_candidate(
                row,
                source,
                reference_dosage_mg_l,
            )
        )
        is not None
    ]
    if not candidates:
        return None

    final_source = str(row.get("Kd_final_source", "")).strip()
    for candidate in candidates:
        if candidate["source"] == final_source:
            return candidate

    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["r2"]),
            candidate["source"] == "Langmuir",
        ),
    )


def build_reference_dose_search_data(
    df_full: pd.DataFrame,
    equilibrium_data: pd.DataFrame,
    *,
    reference_dosage_mg_l: float = 25.0,
    reference_dosage_tolerance_mg_l: float = 5.0,
) -> pd.DataFrame:
    """Build an isolated search sheet without modifying existing projections.

    A row is included when a fit-qualified Langmuir/Freundlich model can be
    recalculated exactly at the reference dose, or when its experimental dose
    lies within the configured tolerance. Exact isotherm results take priority
    over near-dose observations. The ``search_*`` fields intentionally identify
    whether the searchable value is exact or approximate through basis/method
    columns rather than silently labelling a near-dose value as a 25 mg/L result.
    """
    if not math.isfinite(reference_dosage_mg_l) or reference_dosage_mg_l <= 0:
        raise ValueError("reference_dosage_mg_l must be a positive finite number")
    if (
        not math.isfinite(reference_dosage_tolerance_mg_l)
        or reference_dosage_tolerance_mg_l < 0
    ):
        raise ValueError(
            "reference_dosage_tolerance_mg_l must be a non-negative finite number"
        )

    output_columns = _existing_unique(
        REFERENCE_DOSE_PREFIX_COLUMNS
        + list(equilibrium_data.columns)
        + REFERENCE_DOSE_ISOTHERM_COLUMNS
        + REFERENCE_DOSE_CITATION_COLUMNS,
        REFERENCE_DOSE_PREFIX_COLUMNS
        + list(equilibrium_data.columns)
        + REFERENCE_DOSE_ISOTHERM_COLUMNS
        + REFERENCE_DOSE_CITATION_COLUMNS,
    )
    if equilibrium_data.empty or "standardized_row_id" not in df_full.columns:
        return pd.DataFrame(columns=output_columns)

    full_by_id = df_full.set_index("standardized_row_id", drop=False)
    records: list[dict[str, object]] = []
    for _, equilibrium_row in equilibrium_data.iterrows():
        row_id = equilibrium_row.get("standardized_row_id")
        if row_id not in full_by_id.index:
            continue
        full_row = full_by_id.loc[row_id]
        if isinstance(full_row, pd.DataFrame):
            full_row = full_row.iloc[0]

        experimental_dose = _coerce_number(
            full_row.get("Adsorbent_dosage_value_mg/L")
        )
        near_reference_dose = (
            experimental_dose is not None
            and math.isfinite(experimental_dose)
            and abs(experimental_dose - reference_dosage_mg_l)
            <= reference_dosage_tolerance_mg_l
        )
        isotherm = _reference_dose_isotherm_result(
            full_row,
            reference_dosage_mg_l,
        )
        if isotherm is None and not near_reference_dose:
            continue

        if isotherm is not None:
            search_dose = reference_dosage_mg_l
            search_kd = float(isotherm["kd"])
            search_log_kd = float(isotherm["log_kd"])
            search_ce = float(isotherm["ce"])
            search_removal = float(isotherm["removal"])
            basis = "isotherm_recalculated_at_reference_dose"
            method = str(isotherm["source"])
            exact = True
            isotherm_r2: object = float(isotherm["r2"])
        else:
            search_kd = _coerce_number(full_row.get("Kd_final_L/g"))
            if search_kd is None or not math.isfinite(search_kd) or search_kd <= 0:
                continue
            search_log_kd = _coerce_number(full_row.get("Kd_final_log10(L/g)"))
            if search_log_kd is None or not math.isfinite(search_log_kd):
                search_log_kd = math.log10(search_kd)
            search_dose = float(experimental_dose)
            search_ce = _coerce_number(full_row.get("Ce_final_mg/L"))
            search_removal = _coerce_number(
                full_row.get("apparent_removal_fraction")
            )
            basis = "observed_dose_within_tolerance"
            method = f"Observed {str(full_row.get('Kd_final_source', '')).strip()}"
            exact = math.isclose(
                search_dose,
                reference_dosage_mg_l,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            isotherm_r2 = ""

        record = equilibrium_row.to_dict()
        record.update(
            {
                "reference_dosage_target_mg/L": reference_dosage_mg_l,
                "reference_dosage_tolerance_mg/L": reference_dosage_tolerance_mg_l,
                "reference_dose_basis": basis,
                "reference_dose_method": method,
                "reference_dose_exact_flag": exact,
                "reference_isotherm_R2": isotherm_r2,
                "experimental_dosage_delta_from_reference_mg/L": (
                    experimental_dose - reference_dosage_mg_l
                    if experimental_dose is not None
                    else ""
                ),
                "search_dosage_mg/L": search_dose,
                "search_Kd_L/g": search_kd,
                "search_logKd_log10(L/g)": search_log_kd,
                "search_Ce_mg/L": search_ce if search_ce is not None else "",
                "search_removal_fraction": (
                    search_removal if search_removal is not None else ""
                ),
            }
        )
        for column in REFERENCE_DOSE_ISOTHERM_COLUMNS:
            record[column] = full_row.get(column, "")
        for column in REFERENCE_DOSE_CITATION_COLUMNS:
            record[column] = full_row.get(column, "")
        records.append(record)

    return pd.DataFrame.from_records(records, columns=output_columns)


def write_output_workbook(
    df: pd.DataFrame,
    output_path: str,
    order_sheet1_csv: str,
    pinned_sheet1_columns: list[str],
    audit_output_columns: list[str],
    chemistry_output_columns: list[str],
    range_trace_columns: list[str],
    *,
    reference_dosage_mg_l: float = 25.0,
    reference_dosage_tolerance_mg_l: float = 5.0,
) -> None:
    """Write the normalized workbook and review sheets."""
    assign_equilibrium_determination(df)
    materialize_record_status(df)
    auto_side_by_side_cols = list(df.columns)
    df_ordered_auto = df[auto_side_by_side_cols]

    order_sheet1 = load_column_order(order_sheet1_csv)

    df_ordered = reorder_columns_by_family(
        df_ordered_auto,
        preferred_order=order_sheet1,
        pinned=pinned_sheet1_columns,
    )

    trace_cols = set(range_trace_columns)
    trace_cols.update(
        c for c in df_ordered.columns
        if _is_model_ready_trace_column(c)
    )
    drop_from_model_ready = (
        set(audit_output_columns + chemistry_output_columns)
        | trace_cols
    ) - set(REVIEW_SIGNAL_COLUMNS)
    df_model_ready_base = df_ordered[
        [c for c in df_ordered.columns if c not in drop_from_model_ready]
    ].copy()

    equilibrium_data = reorder_columns_by_family(
        build_model_ready_equilibrium_data(df_model_ready_base),
        preferred_order=order_sheet1,
        pinned=pinned_sheet1_columns,
    )
    review_required = build_review_required(df_ordered)
    non_equilibrium_data = reorder_columns_by_family(
        build_non_equilibrium_data(df_ordered),
        preferred_order=order_sheet1,
        pinned=pinned_sheet1_columns,
    )
    equilibrium_status_review = reorder_columns_by_family(
        build_equilibrium_status_review(df_ordered),
        preferred_order=order_sheet1,
        pinned=pinned_sheet1_columns,
    )
    reference_dose_search = build_reference_dose_search_data(
        df_ordered,
        equilibrium_data,
        reference_dosage_mg_l=reference_dosage_mg_l,
        reference_dosage_tolerance_mg_l=reference_dosage_tolerance_mg_l,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_ordered.to_excel(writer, sheet_name=STANDARDIZED_FULL_SHEET_NAME, index=False)
        equilibrium_data.to_excel(writer, sheet_name=EQUILIBRIUM_DATA_SHEET_NAME, index=False)
        review_required.to_excel(writer, sheet_name=REVIEW_REQUIRED_SHEET_NAME, index=False)
        non_equilibrium_data.to_excel(writer, sheet_name=NON_EQUILIBRIUM_DATA_SHEET_NAME, index=False)
        equilibrium_status_review.to_excel(writer, sheet_name=EQUILIBRIUM_STATUS_REVIEW_SHEET_NAME, index=False)
        reference_dose_search.to_excel(
            writer,
            sheet_name=REFERENCE_DOSE_SEARCH_SHEET_NAME,
            index=False,
        )

    print("Saved standardized data with six sheets:")
    print(f"  - {STANDARDIZED_FULL_SHEET_NAME} (full normalized data, family-grouped by {os.path.basename(order_sheet1_csv)} if present)")
    print(f"  - {EQUILIBRIUM_DATA_SHEET_NAME} ({len(equilibrium_data)} model-ready equilibrium rows)")
    print(f"  - {REVIEW_REQUIRED_SHEET_NAME} ({len(review_required)} records requiring review)")
    print(f"  - {NON_EQUILIBRIUM_DATA_SHEET_NAME} ({len(non_equilibrium_data)} non-equilibrium rows)")
    print(f"  - {EQUILIBRIUM_STATUS_REVIEW_SHEET_NAME} ({len(equilibrium_status_review)} rows with unclear equilibrium status)")
    print(
        f"  - {REFERENCE_DOSE_SEARCH_SHEET_NAME} "
        f"({len(reference_dose_search)} rows at {reference_dosage_mg_l:g} mg/L "
        f"or within ±{reference_dosage_tolerance_mg_l:g} mg/L)"
    )
    print(f"File: {output_path}")
