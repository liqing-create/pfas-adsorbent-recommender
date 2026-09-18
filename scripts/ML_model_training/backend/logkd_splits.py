"""Scenario-valid outer splits and target-free inner-fold allocation.

Final testing assignments are always seeded random splits.  ``evidence_balanced``
is deliberately limited to inner validation folds: it retains, relative to the
complete frozen outer-training cohort, the reference entities, reference value
support, and reference within-study contrasts that can inform tuning.  It never
uses targets, predictions, model inputs selected in a fold, or feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import logkd_config as cfg
from .logkd_coverage import (
    REFERENCE_RETENTION_BLOCKS,
    REFERENCE_RETENTION_BLOCK_SLUGS,
    _fixed_reference_value_labels,
    availability_unit_keys,
)
from .logkd_data import clean_text, normalize_pfas_key


TRAINING = "training"
TESTING = "testing"
_VALID_SPLITS = {TRAINING, TESTING}


@dataclass
class SplitResult:
    assignments: pd.DataFrame
    diagnostics: pd.DataFrame
    validation: dict[str, Any]


def _clean_column(df: pd.DataFrame, column: str, missing: str = "<missing>") -> pd.Series:
    if column not in df.columns:
        return pd.Series(missing, index=df.index, dtype="object")
    return df[column].map(clean_text).replace("", missing)


def _pfas_keys(df: pd.DataFrame) -> pd.Series:
    if "PFAS_name" not in df.columns:
        raise ValueError("Split construction requires PFAS_name.")
    return df["PFAS_name"].map(normalize_pfas_key).replace("", "<missing>")


def _adsorbent_ids(df: pd.DataFrame) -> pd.Series:
    if "adsorbent_id" not in df.columns:
        raise ValueError("Split construction requires adsorbent_id.")
    return _clean_column(df, "adsorbent_id")


def _adsorbent_keys(df: pd.DataFrame) -> pd.Series:
    return _clean_column(df, "adsorbent_identity_key") if "adsorbent_identity_key" in df.columns else _adsorbent_ids(df)


def split_row_ids(df: pd.DataFrame) -> tuple[str, pd.Series]:
    for column in cfg.SPLIT_ROW_ID_CANDIDATES:
        if column in df.columns:
            values = df[column].map(clean_text)
            if values.ne("").all() and values.is_unique:
                return column, values
    raise ValueError(f"A frozen split requires a unique ID from {cfg.SPLIT_ROW_ID_CANDIDATES}.")


def source_row_groups(df: pd.DataFrame) -> pd.Series:
    if "source_row_index" not in df.columns:
        raise ValueError("Split construction requires source_row_index for source-record cohesion.")
    values = df["source_row_index"].map(clean_text)
    if values.eq("").any():
        raise ValueError("Split construction requires a nonblank source_row_index for every row.")
    return values.map(lambda value: f"source_row::{value}")


def split_unit_values(df: pd.DataFrame, strategy: str) -> tuple[str, pd.Series]:
    if strategy not in cfg.SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy!r}.")
    if strategy == "random_row":
        return "source_row_index", source_row_groups(df)
    if strategy == "study":
        unit_type, units = "study_no", _clean_column(df, "study_no")
    elif strategy == "pfas":
        unit_type, units = "PFAS_name", _pfas_keys(df)
    elif strategy == "adsorbent":
        unit_type, units = "adsorbent_identity_key", _adsorbent_keys(df)
    else:
        pfas, adsorbent = _pfas_keys(df), _adsorbent_keys(df)
        if pfas.eq("<missing>").any() or adsorbent.eq("<missing>").any():
            raise ValueError("Combination split requires nonmissing PFAS and adsorbent identities.")
        unit_type, units = "PFAS_name x adsorbent_identity_key", pfas + "\x1f" + adsorbent
    source_units = source_row_groups(df)
    crosses = pd.DataFrame({"source_row": source_units, "split_unit": units}).groupby("source_row", dropna=False)["split_unit"].nunique()
    if crosses.gt(1).any():
        raise ValueError(f"Split strategy {strategy!r} assigns one source row to multiple split units.")
    return unit_type, units


def _group_table(df: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.Series]:
    unit_type, units = split_unit_values(df, strategy)
    work = pd.DataFrame({
        "split_unit": units.astype(str),
        "pfas_key": _pfas_keys(df).astype(str),
        "adsorbent_identity_key": _adsorbent_keys(df).astype(str),
        "row_count": 1,
    })
    groups = work.groupby("split_unit", as_index=False, dropna=False).agg(
        row_count=("row_count", "sum"),
        pfas_key=("pfas_key", "first"),
        adsorbent_identity_key=("adsorbent_identity_key", "first"),
    )
    groups.attrs["split_unit_type"] = unit_type
    return groups, units


def outer_allocation_protocol_metadata(split_strategy: str, allocation_method: str) -> dict[str, Any]:
    if allocation_method not in cfg.OUTER_ALLOCATION_METHODS:
        raise ValueError(f"Unknown outer allocation method {allocation_method!r}.")
    if allocation_method == "random_row":
        if split_strategy != "random_row":
            raise ValueError("random_row allocation requires split_strategy='random_row'.")
        return {
            "allocation_method": allocation_method,
            "evaluation_protocol": "seeded_random_source_row_split",
            "split_construction": "seeded_random_source_row_allocation",
            "group_integrity_enforced": True,
            "coverage_objective_used_for_allocation": False,
        }
    if split_strategy == "random_row":
        raise ValueError(f"{allocation_method} allocation requires a grouped split strategy.")
    if allocation_method == "size_matched_random_group":
        return {
            "allocation_method": allocation_method,
            "evaluation_protocol": "seeded_size_matched_random_group_split",
            "split_construction": "seeded_size_matched_random_group_allocation",
            "testing_size_policy": "nearest_feasible_fraction_of_rows",
            "group_integrity_enforced": True,
            "coverage_objective_used_for_allocation": False,
            "allocation_sampling_method": "seeded_random_cost_milp_feasible_selection",
            "allocation_sampling_uniform_over_feasible_group_sets": False,
        }
    return {
        "allocation_method": allocation_method,
        "evaluation_protocol": "seeded_random_group_split",
        "split_construction": "seeded_random_group_allocation",
        "testing_size_policy": "fixed_fraction_of_groups; realized_row_fraction_reported",
        "group_integrity_enforced": True,
        "coverage_objective_used_for_allocation": False,
    }


def _assignments_frame(df: pd.DataFrame, strategy: str, test_units: set[str], construction: str) -> pd.DataFrame:
    unit_type, units = split_unit_values(df, strategy)
    row_id_column, row_ids = split_row_ids(df)
    return pd.DataFrame({
        "row_position": np.arange(len(df), dtype=int),
        "row_id_column": row_id_column,
        "row_id": row_ids.to_numpy(),
        "source_row_index": df["source_row_index"].to_numpy(),
        "split": np.where(units.astype(str).isin(test_units), TESTING, TRAINING),
        "split_strategy": strategy,
        "split_unit_type": unit_type,
        "split_unit": units.to_numpy(),
        "study_no": _clean_column(df, "study_no").to_numpy(),
        "PFAS_name": _clean_column(df, "PFAS_name").to_numpy(),
        "pfas_key": _pfas_keys(df).to_numpy(),
        "adsorbent_id": _adsorbent_ids(df).to_numpy(),
        "adsorbent_identity_key": _adsorbent_keys(df).to_numpy(),
        "split_construction": construction,
    })


def _combination_random_units(groups: pd.DataFrame, requested_units: int, random_seed: int) -> set[str]:
    """Randomly select feasible pairs while retaining every test component in training."""

    unit_count = len(groups)
    objective = np.random.default_rng(np.random.SeedSequence([random_seed, 8129])).random(unit_count)
    constraints: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    constraints.append(np.ones(unit_count))
    lower.append(float(requested_units))
    upper.append(float(requested_units))
    for column in ("pfas_key", "adsorbent_identity_key"):
        for _, indices in groups.groupby(column, sort=False).groups.items():
            row = np.zeros(unit_count)
            row[list(indices)] = 1.0
            constraints.append(row)
            lower.append(-np.inf)
            upper.append(float(len(indices) - 1))
    result = milp(
        c=objective,
        integrality=np.ones(unit_count, dtype=int),
        bounds=Bounds(np.zeros(unit_count), np.ones(unit_count)),
        constraints=LinearConstraint(np.vstack(constraints), np.asarray(lower), np.asarray(upper)),
        options={"disp": False},
    )
    if result.x is None or result.status not in {0, 1}:
        raise ValueError(f"No feasible random combination allocation: {result.message}")
    selected = np.asarray(result.x) >= 0.5
    if not selected.any() or selected.all():
        raise ValueError("Random combination allocation is empty or leaves no training units.")
    return set(groups.loc[selected, "split_unit"].astype(str))


def _random_test_units(
    groups: pd.DataFrame,
    strategy: str,
    test_fraction: float,
    random_seed: int,
    distinct_from_test_unit_sets: tuple[frozenset[str], ...],
) -> set[str]:
    requested_units = min(max(int(np.ceil(len(groups) * test_fraction)), 1), len(groups) - 1)
    if strategy == "combination":
        return _combination_random_units(groups, requested_units, random_seed)
    units = groups["split_unit"].astype(str).to_numpy()
    prior = set(distinct_from_test_unit_sets)
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 8129]))
    for _ in range(max(1000, len(units) * 100)):
        selected = frozenset(rng.choice(units, size=requested_units, replace=False).tolist())
        if selected not in prior:
            return set(selected)
    raise ValueError("Could not draw a distinct random allocation.")


def _append_constraint(
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
    values: np.ndarray,
    *,
    lb: float = -np.inf,
    ub: float = np.inf,
) -> None:
    rows.append(values)
    lower.append(lb)
    upper.append(ub)


def _grouped_selection_constraints(
    groups: pd.DataFrame,
    strategy: str,
    variable_count: int,
    distinct_from_test_unit_sets: tuple[frozenset[str], ...],
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Return target-free hard constraints for a valid grouped test set."""

    unit_count = len(groups)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    nonempty = np.zeros(variable_count)
    nonempty[:unit_count] = 1.0
    _append_constraint(rows, lower, upper, nonempty, lb=1.0, ub=float(unit_count - 1))

    # A combination holdout must leave each held-out PFAS and adsorbent
    # represented in outer training, exactly as the standard random method does.
    if strategy == "combination":
        for column in ("pfas_key", "adsorbent_identity_key"):
            for indices in groups.groupby(column, sort=False).groups.values():
                constraint = np.zeros(variable_count)
                constraint[list(indices)] = 1.0
                _append_constraint(
                    rows,
                    lower,
                    upper,
                    constraint,
                    ub=float(len(indices) - 1),
                )

    units = groups["split_unit"].astype(str).to_numpy()
    for prior in distinct_from_test_unit_sets:
        prior_mask = np.isin(units, list(prior))
        prior_count = int(prior_mask.sum())
        if prior_count == 0 or prior_count == unit_count:
            continue
        # Forbid exactly reproducing a retained test-unit membership. Any
        # differing membership lowers this signed total by at least one.
        constraint = np.zeros(variable_count)
        constraint[:unit_count] = np.where(prior_mask, 1.0, -1.0)
        _append_constraint(rows, lower, upper, constraint, ub=float(prior_count - 1))
    return rows, lower, upper


def _solve_milp(
    objective: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            np.vstack(rows), np.asarray(lower), np.asarray(upper)
        ),
        options={"disp": False},
    )
    metadata = {
        "optimization_solver_status": int(result.status),
        "optimization_solver_message": str(result.message),
        "optimization_solver_proved_optimal": bool(result.status == 0),
    }
    if result.x is None or result.status not in {0, 1}:
        raise ValueError(f"No valid size-matched grouped allocation: {result.message}")
    return np.asarray(result.x), metadata


def _size_matched_random_test_units(
    groups: pd.DataFrame,
    strategy: str,
    test_fraction: float,
    random_seed: int,
    distinct_from_test_unit_sets: tuple[frozenset[str], ...],
) -> tuple[set[str], dict[str, Any]]:
    """Draw a target-free grouped holdout closest to the requested row count.

    The first MILP finds the nearest feasible number of testing rows while
    retaining all scenario constraints. A second MILP uses only seeded random
    costs to choose among allocations at that fixed row count; no coverage,
    feature, target, or prediction information is used.
    """

    unit_count = len(groups)
    requested_test_rows = float(groups["row_count"].sum() * test_fraction)
    row_counts = groups["row_count"].to_numpy(dtype=float)

    # Minimize the absolute difference between the selected and requested row
    # counts. The final variable is a nonnegative continuous deviation.
    variable_count = unit_count + 1
    deviation_index = unit_count
    rows, lower, upper = _grouped_selection_constraints(
        groups, strategy, variable_count, distinct_from_test_unit_sets
    )
    testing_rows = np.zeros(variable_count)
    testing_rows[:unit_count] = row_counts
    _append_constraint(
        rows,
        lower,
        upper,
        testing_rows,
        lb=1.0,
        ub=float(row_counts.sum() - 1),
    )
    above_target = testing_rows.copy()
    above_target[deviation_index] = -1.0
    _append_constraint(rows, lower, upper, above_target, ub=requested_test_rows)
    below_target = -testing_rows.copy()
    below_target[deviation_index] = -1.0
    _append_constraint(rows, lower, upper, below_target, ub=-requested_test_rows)
    target_solution, target_solver = _solve_milp(
        np.eye(1, variable_count, deviation_index, dtype=float).ravel(),
        np.r_[np.ones(unit_count, dtype=int), 0],
        np.zeros(variable_count),
        np.r_[np.ones(unit_count), np.inf],
        rows,
        lower,
        upper,
    )
    target_test_rows = int(round(float(np.dot(target_solution[:unit_count], row_counts))))

    # Select one fixed-size feasible membership with a seeded random objective.
    rows, lower, upper = _grouped_selection_constraints(
        groups, strategy, unit_count, distinct_from_test_unit_sets
    )
    _append_constraint(
        rows,
        lower,
        upper,
        row_counts.copy(),
        lb=float(target_test_rows),
        ub=float(target_test_rows),
    )
    random_costs = np.random.default_rng(
        np.random.SeedSequence([random_seed, 24593])
    ).random(unit_count)
    selected_solution, selection_solver = _solve_milp(
        random_costs,
        np.ones(unit_count, dtype=int),
        np.zeros(unit_count),
        np.ones(unit_count),
        rows,
        lower,
        upper,
    )
    selected = selected_solution >= 0.5
    if not selected.any() or selected.all():
        raise ValueError("Size-matched grouped allocation is empty or leaves no training groups.")
    selected_rows = int(round(float(np.dot(selected, row_counts))))
    if selected_rows != target_test_rows:
        raise RuntimeError("Size-matched grouped allocation missed its fixed testing-row target.")
    return set(groups.loc[selected, "split_unit"].astype(str)), {
        "test_row_target_selection_method": (
            "proven_nearest_feasible_grouped_milp"
            if target_solver["optimization_solver_proved_optimal"]
            else "feasible_grouped_milp_target_not_proven_nearest"
        ),
        "test_row_target_requested_rows": requested_test_rows,
        "test_row_target_absolute_deviation_rows": abs(
            target_test_rows - requested_test_rows
        ),
        "test_row_target_proved_nearest": target_solver[
            "optimization_solver_proved_optimal"
        ],
        "test_row_target_solver_status": target_solver["optimization_solver_status"],
        "test_row_target_solver_message": target_solver["optimization_solver_message"],
        "selection_solver_status": selection_solver["optimization_solver_status"],
        "selection_solver_message": selection_solver["optimization_solver_message"],
    }


def build_testing_split(
    df: pd.DataFrame,
    split_strategy: str,
    test_fraction: float,
    random_seed: int,
    target: str,
    allocation_method: str,
    distinct_from_test_unit_sets: tuple[frozenset[str], ...] = (),
) -> SplitResult:
    if not 0 < test_fraction < 1 or len(df) < 2:
        raise ValueError("test_fraction must be between zero and one and at least two rows are required.")
    protocol = outer_allocation_protocol_metadata(split_strategy, allocation_method)
    groups, _ = _group_table(df, split_strategy)
    if len(groups) < 2:
        raise ValueError(f"Split strategy {split_strategy!r} needs at least two holdout units.")
    size_metadata: dict[str, Any]
    if allocation_method == "size_matched_random_group":
        test_units, size_metadata = _size_matched_random_test_units(
            groups,
            split_strategy,
            test_fraction,
            random_seed,
            distinct_from_test_unit_sets,
        )
    else:
        test_units = _random_test_units(
            groups,
            split_strategy,
            test_fraction,
            random_seed,
            distinct_from_test_unit_sets,
        )
        size_metadata = {
            "test_row_target_selection_method": (
                "standard_random_fraction_of_source_rows"
                if split_strategy == "random_row"
                else "standard_random_fraction_of_groups"
            ),
            "test_row_target_requested_rows": float(len(df) * test_fraction),
            "test_row_target_absolute_deviation_rows": None,
            "test_row_target_proved_nearest": False,
        }
    assignments = _assignments_frame(df, split_strategy, test_units, protocol["split_construction"])
    validation = validate_testing_split(assignments)
    observed_fraction = float(assignments["split"].eq(TESTING).mean())
    if size_metadata["test_row_target_absolute_deviation_rows"] is None:
        size_metadata["test_row_target_absolute_deviation_rows"] = abs(
            int(assignments["split"].eq(TESTING).sum())
            - float(size_metadata["test_row_target_requested_rows"])
        )
    validation.update({
        "requested_test_fraction": float(test_fraction),
        "requested_test_group_count": (
            None
            if allocation_method == "size_matched_random_group"
            else int(np.ceil(len(groups) * test_fraction))
        ),
        "observed_test_group_count": int(len(test_units)),
        "observed_test_group_fraction": float(len(test_units) / len(groups)),
        "observed_test_fraction": observed_fraction,
        "test_fraction_absolute_error": abs(observed_fraction - test_fraction),
        "random_seed": int(random_seed),
        "distinctness_reference_repeats": int(len(distinct_from_test_unit_sets)),
        **size_metadata,
        **protocol,
    })
    return SplitResult(assignments, split_diagnostics(df, assignments, target), validation)


def split_indices(assignments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        assignments.index[assignments["split"].eq(TRAINING)].to_numpy(int),
        assignments.index[assignments["split"].eq(TESTING)].to_numpy(int),
    )


def split_diagnostics(df: pd.DataFrame, assignments: pd.DataFrame, target: str) -> pd.DataFrame:
    work = assignments.copy()
    work[target] = pd.to_numeric(df[target].to_numpy(), errors="coerce")
    rows: list[dict[str, Any]] = []
    for label, part in work.groupby("split", sort=False):
        values = part[target].dropna()
        rows.append({
            "split": label,
            "row_count": int(len(part)),
            "row_fraction": float(len(part) / len(work)),
            "split_unit_count": int(part["split_unit"].nunique(dropna=False)),
            "source_row_count": int(part["source_row_index"].nunique(dropna=False)),
            "study_count": int(part["study_no"].nunique(dropna=False)),
            "pfas_count": int(part["pfas_key"].nunique(dropna=False)),
            "adsorbent_count": int(part["adsorbent_id"].nunique(dropna=False)),
            "target_mean": values.mean(),
            "target_sd": values.std(ddof=1),
            "target_min": values.min(),
            "target_max": values.max(),
        })
    return pd.DataFrame(rows)


def validate_testing_split(assignments: pd.DataFrame) -> dict[str, Any]:
    if assignments.empty:
        raise ValueError("Split assignments are empty.")
    strategy = str(assignments["split_strategy"].iloc[0])
    counts = assignments["split"].value_counts()
    checks: dict[str, Any] = {
        "split_strategy": strategy,
        "training_rows": int(counts.get(TRAINING, 0)),
        "testing_rows": int(counts.get(TESTING, 0)),
        "valid_split_labels": bool(assignments["split"].isin(_VALID_SPLITS).all()),
    }
    if not checks["valid_split_labels"] or not checks["training_rows"] or not checks["testing_rows"]:
        raise ValueError("Frozen split must contain valid nonempty training and testing partitions.")
    source_rows = assignments["source_row_index"].map(clean_text)
    if source_rows.eq("").any():
        raise ValueError("Frozen split has a blank source_row_index.")
    if assignments.assign(_source_row=source_rows).groupby("_source_row")["split"].nunique().gt(1).any():
        raise ValueError("A source record crosses training and testing.")
    if assignments.groupby("split_unit")["split"].nunique().gt(1).any():
        raise ValueError("A split unit crosses training and testing.")
    if strategy == "combination":
        training = assignments.loc[assignments["split"].eq(TRAINING)]
        testing = assignments.loc[assignments["split"].eq(TESTING)]
        pfas_missing = int((~testing["pfas_key"].isin(set(training["pfas_key"]))).sum())
        adsorbent_missing = int((~testing["adsorbent_identity_key"].isin(set(training["adsorbent_identity_key"]))).sum())
        checks.update({
            "combination_component_representation_enforced": True,
            "test_pfas_without_training_representation": pfas_missing,
            "test_adsorbents_without_training_representation": adsorbent_missing,
        })
        if pfas_missing or adsorbent_missing:
            raise ValueError("Combination split violates its train-only component representation rule.")
    return checks


def load_testing_assignments(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    assignments = pd.read_csv(path, keep_default_na=False)
    required = {"row_position", "row_id", "row_id_column", "split", "split_strategy", "split_unit", "split_construction"}
    missing = sorted(required.difference(assignments.columns))
    if missing or len(assignments) != len(df):
        raise ValueError(f"Split assignment does not match model rows; missing columns: {missing}")
    assignments = assignments.sort_values("row_position").reset_index(drop=True)
    if not np.array_equal(pd.to_numeric(assignments["row_position"], errors="raise").to_numpy(int), np.arange(len(df))):
        raise ValueError("Split assignment row positions do not match model rows.")
    column, values = split_row_ids(df)
    if assignments["row_id_column"].nunique() != 1 or assignments["row_id_column"].iloc[0] != column:
        raise ValueError("Split assignment ID column does not match the model frame.")
    if not np.array_equal(assignments["row_id"].map(clean_text), values):
        raise ValueError("Split assignment IDs do not match the model frame.")
    validate_testing_split(assignments)
    return assignments


def _balanced_group_folds(groups: pd.DataFrame, n_folds: int, random_seed: int) -> dict[str, int]:
    rng, work = np.random.default_rng(random_seed), groups.copy()
    work["tie_breaker"] = rng.random(len(work))
    work = work.sort_values(["row_count", "tie_breaker", "split_unit"], ascending=[False, True, True])
    loads, assignment = np.zeros(n_folds, dtype=int), {}
    for row in work.itertuples(index=False):
        fold = int(rng.choice(np.flatnonzero(loads == loads.min())))
        assignment[str(row.split_unit)] = fold
        loads[fold] += int(row.row_count)
    return assignment


def _combination_validation_folds(groups: pd.DataFrame, n_folds: int, random_seed: int) -> dict[str, int]:
    pfas_counts = groups.groupby("pfas_key")["split_unit"].nunique()
    adsorbent_counts = groups.groupby("adsorbent_identity_key")["split_unit"].nunique()
    eligible = groups.loc[
        groups["pfas_key"].map(pfas_counts).ge(2)
        & groups["adsorbent_identity_key"].map(adsorbent_counts).ge(2)
    ].copy()
    if len(eligible) < 2:
        raise ValueError("Combination validation has fewer than two pairs with represented components.")
    assignment = {str(unit): -1 for unit in groups["split_unit"]}
    assignment.update(_balanced_group_folds(eligible, min(n_folds, len(eligible)), random_seed))
    changed = True
    while changed:
        changed = False
        for fold in sorted({value for value in assignment.values() if value >= 0}):
            validation = groups[groups["split_unit"].map(assignment).eq(fold)]
            training = groups[~groups["split_unit"].map(assignment).eq(fold)]
            for column in ("pfas_key", "adsorbent_identity_key"):
                unsupported = set(validation[column]).difference(training[column])
                for unit in validation.loc[validation[column].isin(unsupported), "split_unit"]:
                    assignment[str(unit)] = -1
                    changed = True
    present = sorted({value for value in assignment.values() if value >= 0})
    if len(present) < 2:
        raise ValueError("Combination validation could not form two valid validation folds.")
    remap = {fold: index for index, fold in enumerate(present)}
    return {unit: remap[fold] if fold >= 0 else -1 for unit, fold in assignment.items()}


def _validate_validation_assignments(df: pd.DataFrame, strategy: str, assignments: pd.DataFrame) -> None:
    folds = sorted(value for value in assignments["validation_fold"].unique() if value >= 0)
    if len(folds) < 2:
        raise ValueError("Inner validation must contain at least two validation folds.")
    eligible = assignments["validation_fold"].ge(0)
    if assignments.loc[eligible].groupby("split_unit")["validation_fold"].nunique().gt(1).any():
        raise ValueError("A validation holdout unit crosses validation folds.")
    source_rows = source_row_groups(df)
    source_frame = assignments.assign(_source_row=source_rows.to_numpy())
    if source_frame.loc[eligible].groupby("_source_row")["validation_fold"].nunique().gt(1).any():
        raise ValueError("A source record crosses validation folds.")
    if strategy == "combination":
        pfas, adsorbent = _pfas_keys(df), _adsorbent_keys(df)
        for fold in folds:
            validation = assignments["validation_fold"].eq(fold)
            training = assignments["validation_fold"].ne(fold)
            if not set(pfas[validation]).issubset(pfas[training]) or not set(adsorbent[validation]).issubset(adsorbent[training]):
                raise ValueError("Combination validation violates its train-only component representation rule.")


_RETENTION_COMPONENTS = (
    ("entity_retention", "reference_entity_retention"),
    ("value_support_retention", "reference_value_support_retention"),
    ("contrast_retention", "reference_contrast_retention"),
)


def _allocation_manifest(reference_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep the fixed outer-training diagnostic inputs usable for allocation."""

    required = {"feature", "bucket", "kind"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Reference-retention manifest lacks required columns: {missing}")
    work = manifest.loc[:, ["feature", "bucket", "kind"]].copy()
    for column in work.columns:
        work[column] = work[column].map(clean_text)
    work = work.loc[
        work["feature"].ne("")
        & work["feature"].isin(reference_df.columns)
        & work["bucket"].isin(REFERENCE_RETENTION_BLOCKS)
        & work["kind"].isin({"numeric", "categorical"})
    ].drop_duplicates("feature", keep="first").reset_index(drop=True)
    if work.empty:
        raise ValueError("Evidence-balanced inner allocation has no usable reference-manifest features.")
    return work


class _ReferenceRetentionObjective:
    """Fast exact evaluation of the fixed-reference inner-fold objective.

    The source dataframe is the outer-training partition.  Counts are prepared
    once by split unit, so local allocation moves only perform small matrix
    operations; they do not rerun feature selection or inspect target values.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        split_strategy: str,
        reference_manifest: pd.DataFrame,
        numeric_bins: int,
    ) -> None:
        self.reference_df = reference_df.reset_index(drop=True)
        self.split_strategy = split_strategy
        self.manifest = _allocation_manifest(self.reference_df, reference_manifest)
        self.groups, units = _group_table(self.reference_df, split_strategy)
        self.unit_names = self.groups["split_unit"].astype(str).to_numpy()
        self.row_unit = pd.Categorical(
            units.astype(str), categories=self.unit_names, ordered=True
        ).codes.astype(int)
        if (self.row_unit < 0).any():
            raise ValueError("Could not map all rows to inner-allocation split units.")
        self.unit_rows = self.groups["row_count"].to_numpy(dtype=float)
        self.n_units = len(self.unit_names)
        self._feature_blocks = self.manifest["bucket"].tolist()
        self._build_entity_counts()
        self._build_value_counts(numeric_bins)
        self._build_contrast_counts(numeric_bins)

    @staticmethod
    def _unit_count_matrix(unit_codes: np.ndarray, stratum_codes: np.ndarray, n_units: int, n_strata: int) -> np.ndarray:
        matrix = np.zeros((n_units, n_strata), dtype=np.int32)
        valid = stratum_codes >= 0
        np.add.at(matrix, (unit_codes[valid], stratum_codes[valid]), 1)
        return matrix

    def _build_entity_counts(self) -> None:
        self.entity_matrices: dict[str, np.ndarray] = {}
        self.entity_totals: dict[str, np.ndarray] = {}
        for block in REFERENCE_RETENTION_BLOCKS:
            _, keys = availability_unit_keys(self.reference_df, block)
            codes, _ = pd.factorize(keys.where(keys.ne(""), np.nan), sort=False)
            matrix = self._unit_count_matrix(self.row_unit, codes.astype(int), self.n_units, int((codes >= 0).sum() and codes.max() + 1))
            self.entity_matrices[block] = matrix
            self.entity_totals[block] = matrix.sum(axis=0)

    def _build_value_counts(self, numeric_bins: int) -> None:
        matrices: list[np.ndarray] = []
        masses: list[float] = []
        self.value_ranges: list[tuple[int, int]] = []
        offset = 0
        for row in self.manifest.itertuples(index=False):
            feature = str(row.feature)
            block = str(row.bucket)
            labels, _, _, _ = _fixed_reference_value_labels(
                self.reference_df, self.reference_df, feature, str(row.kind), numeric_bins
            )
            codes, levels = pd.factorize(labels, sort=False)
            n_levels = len(levels)
            self.value_ranges.append((offset, offset + n_levels))
            offset += n_levels
            if not n_levels:
                continue
            matrices.append(self._unit_count_matrix(self.row_unit, codes.astype(int), self.n_units, n_levels))
            _, unit_keys = availability_unit_keys(self.reference_df, block)
            weight_frame = pd.DataFrame({"unit": unit_keys, "code": codes})
            weight_frame = weight_frame.loc[weight_frame["unit"].ne("") & weight_frame["code"].ge(0)]
            if weight_frame.empty:
                masses.extend([np.nan] * n_levels)
                continue
            per_unit = weight_frame.groupby(["unit", "code"], sort=False).size().rename("count").reset_index()
            per_unit["unit_total"] = per_unit.groupby("unit", sort=False)["count"].transform("sum")
            weighted = (per_unit.assign(weight=per_unit["count"] / per_unit["unit_total"])
                        .groupby("code", sort=False)["weight"].sum() / per_unit["unit"].nunique())
            masses.extend([float(weighted.get(code, 0.0)) for code in range(n_levels)])
        self.value_matrix = np.concatenate(matrices, axis=1) if matrices else np.zeros((self.n_units, 0), dtype=np.int32)
        self.value_totals = self.value_matrix.sum(axis=0)
        self.value_masses = np.asarray(masses, dtype=float)

    def _build_contrast_counts(self, numeric_bins: int) -> None:
        matrices: list[np.ndarray] = []
        group_features: list[int] = []
        offset = 0
        study = _clean_column(self.reference_df, "study_no")
        for feature_index, row in enumerate(self.manifest.itertuples(index=False)):
            labels, _, _, _ = _fixed_reference_value_labels(
                self.reference_df, self.reference_df, str(row.feature), str(row.kind), numeric_bins
            )
            valid = study.ne("") & labels.notna()
            pair = pd.Series(np.nan, index=self.reference_df.index, dtype="object")
            pair.loc[valid] = list(zip(study.loc[valid].astype(str), labels.loc[valid].astype(str)))
            pair_codes, pair_levels = pd.factorize(pair, sort=False)
            n_pairs = len(pair_levels)
            if not n_pairs:
                continue
            matrices.append(self._unit_count_matrix(self.row_unit, pair_codes.astype(int), self.n_units, n_pairs))
            study_levels = [str(value[0]) for value in pair_levels]
            local_groups, _ = pd.factorize(pd.Index(study_levels), sort=False)
            group_features.extend([feature_index] * int(local_groups.max() + 1))
            self_group_codes = local_groups + offset
            if not hasattr(self, "_contrast_pair_groups"):
                self._contrast_pair_groups = []
            self._contrast_pair_groups.extend(self_group_codes.tolist())
            offset += int(local_groups.max() + 1)
        self.contrast_matrix = np.concatenate(matrices, axis=1) if matrices else np.zeros((self.n_units, 0), dtype=np.int32)
        self.contrast_totals = self.contrast_matrix.sum(axis=0)
        self.contrast_pair_groups = np.asarray(getattr(self, "_contrast_pair_groups", []), dtype=int)
        self.contrast_group_features = np.asarray(group_features, dtype=int)
        self.n_contrast_groups = len(group_features)
        if self.n_contrast_groups:
            labels_per_group = np.bincount(
                self.contrast_pair_groups,
                weights=(self.contrast_totals > 0).astype(float),
                minlength=self.n_contrast_groups,
            )
            self.reference_contrast_groups = labels_per_group >= 2
        else:
            self.reference_contrast_groups = np.zeros(0, dtype=bool)

    def _heldout_counts(self, unit_folds: np.ndarray) -> np.ndarray:
        folds = sorted(int(value) for value in np.unique(unit_folds) if value >= 0)
        if folds != list(range(len(folds))) or len(folds) < 2:
            raise ValueError("Evidence-balanced allocation needs contiguous, nonempty validation folds.")
        membership = np.zeros((len(folds), self.n_units), dtype=np.int8)
        eligible_units = np.flatnonzero(unit_folds >= 0)
        membership[unit_folds[eligible_units], eligible_units] = 1
        return membership

    @staticmethod
    def _block_feature_means(values: np.ndarray, feature_blocks: list[str]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for block in REFERENCE_RETENTION_BLOCKS:
            indices = [index for index, value in enumerate(feature_blocks) if value == block]
            if indices:
                result[block] = np.nanmean(values[:, indices], axis=1)
            else:
                result[block] = np.full(values.shape[0], np.nan)
        return result

    def evaluate(self, unit_folds: np.ndarray) -> tuple[pd.DataFrame, tuple[float, float, float]]:
        membership = self._heldout_counts(unit_folds)
        n_folds = len(membership)
        records: dict[str, np.ndarray] = {"validation_fold": np.arange(n_folds, dtype=int)}

        for block in REFERENCE_RETENTION_BLOCKS:
            totals = self.entity_totals[block]
            if len(totals):
                retained = (totals[None, :] - membership @ self.entity_matrices[block] > 0).mean(axis=1)
            else:
                retained = np.full(n_folds, np.nan)
            records[f"validation_reference_entity_retention_{REFERENCE_RETENTION_BLOCK_SLUGS[block]}"] = retained

        value_by_feature = np.full((n_folds, len(self.manifest)), np.nan)
        heldout_value = membership @ self.value_matrix
        for feature_index, (start, end) in enumerate(self.value_ranges):
            if start == end:
                continue
            masses = self.value_masses[start:end]
            if np.isfinite(masses).all() and masses.sum() > 0:
                value_by_feature[:, feature_index] = ((self.value_totals[start:end][None, :] - heldout_value[:, start:end]) > 0) @ masses
        for block, values in self._block_feature_means(value_by_feature, self._feature_blocks).items():
            records[f"validation_reference_value_support_retention_{REFERENCE_RETENTION_BLOCK_SLUGS[block]}"] = values

        contrast_by_feature = np.full((n_folds, len(self.manifest)), np.nan)
        if self.n_contrast_groups:
            heldout_contrast = membership @ self.contrast_matrix
            represented = self.contrast_totals[None, :] - heldout_contrast > 0
            group_counts = np.zeros((n_folds, self.n_contrast_groups), dtype=np.int16)
            for fold in range(n_folds):
                np.add.at(group_counts[fold], self.contrast_pair_groups, represented[fold].astype(np.int16))
            retained_groups = group_counts >= 2
            for feature_index in range(len(self.manifest)):
                group_mask = (self.contrast_group_features == feature_index) & self.reference_contrast_groups
                if group_mask.any():
                    contrast_by_feature[:, feature_index] = retained_groups[:, group_mask].mean(axis=1)
        for block, values in self._block_feature_means(contrast_by_feature, self._feature_blocks).items():
            records[f"validation_reference_contrast_retention_{REFERENCE_RETENTION_BLOCK_SLUGS[block]}"] = values

        frame = pd.DataFrame(records)
        component_columns = [
            f"validation_{metric}_{slug}"
            for _, metric in _RETENTION_COMPONENTS
            for slug in REFERENCE_RETENTION_BLOCK_SLUGS.values()
            if f"validation_{metric}_{slug}" in frame
        ]
        scores = frame[component_columns].mean(axis=1, skipna=True)
        frame["validation_reference_retention_score"] = scores
        frame["validation_reference_retention_loss"] = 1.0 - scores
        losses = frame["validation_reference_retention_loss"].to_numpy(dtype=float)
        finite = losses[np.isfinite(losses)]
        if not len(finite):
            raise ValueError("Evidence-balanced allocation has no finite reference-retention objective values.")
        # The loss tuple deliberately follows the allocation priority.  The
        # allocator first improves overall evidence retained across folds,
        # then protects the weakest fold, then reduces fold-to-fold spread.
        return frame, (float(finite.mean()), float(finite.max()), float(finite.std(ddof=0)))


def _objective_improves(candidate: tuple[float, float, float], current: tuple[float, float, float], tolerance: float = 1e-12) -> bool:
    for candidate_value, current_value in zip(candidate, current, strict=True):
        if candidate_value < current_value - tolerance:
            return True
        if candidate_value > current_value + tolerance:
            return False
    return False


def _objective_loss_record(loss: tuple[float, float, float]) -> dict[str, float]:
    """Make the ordered inner-allocation objective explicit in run metadata."""

    return {
        "mean_fold_loss": float(loss[0]),
        "maximum_fold_loss": float(loss[1]),
        "fold_loss_sd": float(loss[2]),
    }


def _unit_folds_from_mapping(groups: pd.DataFrame, mapping: dict[str, int]) -> np.ndarray:
    return groups["split_unit"].astype(str).map(mapping).to_numpy(dtype=int)


def _evidence_balanced_folds(
    df_training: pd.DataFrame,
    strategy: str,
    groups: pd.DataFrame,
    baseline_mapping: dict[str, int],
    reference_manifest: pd.DataFrame,
    numeric_bins: int,
    random_seed: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Improve overall inner-fold retention, while protecting every fold."""

    objective = _ReferenceRetentionObjective(df_training, strategy, reference_manifest, numeric_bins)
    values = _unit_folds_from_mapping(groups, baseline_mapping)
    baseline_frame, baseline_objective = objective.evaluate(values)
    baseline_loads = np.bincount(values[values >= 0], weights=objective.unit_rows[values >= 0], minlength=int(values.max()) + 1)
    maximum_row_imbalance = float(baseline_loads.max() - baseline_loads.min())
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 94711]))
    candidate_moves = 0
    moves_applied = 0
    maximum_passes = max(1, min(5, len(values)))
    current_frame, current_objective = baseline_frame, baseline_objective

    def valid(values_to_check: np.ndarray) -> bool:
        eligible = values_to_check >= 0
        loads = np.bincount(values_to_check[eligible], weights=objective.unit_rows[eligible], minlength=int(values_to_check.max()) + 1)
        if not len(loads) or loads.min() <= 0 or loads.max() - loads.min() > maximum_row_imbalance + 1e-9:
            return False
        if strategy != "combination":
            return True
        row_values = pd.Categorical(
            split_unit_values(df_training, strategy)[1].astype(str),
            categories=objective.unit_names,
            ordered=True,
        ).codes
        assignments = pd.DataFrame({"validation_fold": values_to_check[row_values], "split_unit": split_unit_values(df_training, strategy)[1].to_numpy()})
        try:
            _validate_validation_assignments(df_training, strategy, assignments)
        except ValueError:
            return False
        return True

    for _ in range(maximum_passes):
        best_values: np.ndarray | None = None
        best_objective = current_objective
        unit_order = rng.permutation(np.flatnonzero(values >= 0))
        folds = np.unique(values[values >= 0])
        for unit_index in unit_order:
            current_fold = int(values[unit_index])
            for new_fold in folds:
                if new_fold == current_fold:
                    continue
                candidate = values.copy()
                candidate[unit_index] = new_fold
                candidate_moves += 1
                if not valid(candidate):
                    continue
                _, candidate_objective = objective.evaluate(candidate)
                if _objective_improves(candidate_objective, best_objective):
                    best_values, best_objective = candidate, candidate_objective
        # A perfectly row-balanced seeded layout can make every single-unit
        # move infeasible. Bounded deterministic exchanges preserve that hard
        # balance rule while still permitting local improvement.
        if best_values is None:
            eligible_units = np.flatnonzero(values >= 0)
            for unit_index in unit_order:
                other_units = eligible_units[values[eligible_units] != values[unit_index]]
                if not len(other_units):
                    continue
                for other_index in rng.permutation(other_units)[: min(8, len(other_units))]:
                    candidate = values.copy()
                    candidate[unit_index], candidate[other_index] = candidate[other_index], candidate[unit_index]
                    candidate_moves += 1
                    if not valid(candidate):
                        continue
                    _, candidate_objective = objective.evaluate(candidate)
                    if _objective_improves(candidate_objective, best_objective):
                        best_values, best_objective = candidate, candidate_objective
        if best_values is None:
            break
        values = best_values
        current_frame, current_objective = objective.evaluate(values)
        moves_applied += 1

    final_mapping = dict(zip(objective.unit_names, values.tolist(), strict=True))
    return final_mapping, {
        "allocation_method": "evidence_balanced",
        "reference_scope": "complete_frozen_outer_training_cohort",
        "reference_manifest_feature_count": int(len(objective.manifest)),
        "numeric_value_policy": f"fixed_reference_values_or_{numeric_bins}_quantile_bins",
        "objective_components": [metric for _, metric in _RETENTION_COMPONENTS],
        "block_aggregation": "mean_across_fixed_features; blocks_and_components_equally_weighted",
        "fold_objective": "mean_of_finite_block_component_retentions",
        "optimization_rule": "lexicographic_minimize(mean_fold_loss, maximum_fold_loss, fold_loss_sd)",
        "optimization_priority": [
            "maximize_mean_fold_reference_retention_score",
            "maximize_minimum_fold_reference_retention_score",
            "minimize_fold_reference_retention_score_sd",
        ],
        "row_imbalance_constraint": "not greater than seeded random baseline",
        "conditional_contrast_rule": "features without reference contrasting studies are omitted from the contrast component",
        "baseline_objective_loss": _objective_loss_record(baseline_objective),
        "optimized_objective_loss": _objective_loss_record(current_objective),
        "candidate_moves_considered": int(candidate_moves),
        "moves_applied": int(moves_applied),
        "optimization_status": "improved" if moves_applied else "baseline_retained_no_strict_feasible_improvement",
        "baseline_fold_diagnostics": baseline_frame.to_dict(orient="records"),
        "optimized_fold_diagnostics": current_frame.to_dict(orient="records"),
    }


def make_validation_assignments(
    df_training: pd.DataFrame,
    split_strategy: str,
    validation_folds: int,
    random_seed: int,
    allocation_method: str | None = None,
    reference_manifest: pd.DataFrame | None = None,
    numeric_bins: int = cfg.DEFAULT_EVIDENCE_COVERAGE_NUMERIC_BINS,
) -> pd.DataFrame:
    expected_random = "random_row" if split_strategy == "random_row" else "random_group"
    allocation_method = allocation_method or expected_random
    if allocation_method not in cfg.INNER_ALLOCATION_METHODS:
        raise ValueError(f"Unknown inner allocation method {allocation_method!r}.")
    if allocation_method != "evidence_balanced" and allocation_method != expected_random:
        raise ValueError(
            f"{allocation_method} inner allocation requires split_strategy="
            f"{'random_row' if allocation_method == 'random_row' else 'a grouped strategy'}"
        )
    groups, units = _group_table(df_training, split_strategy)
    folds = min(validation_folds, len(groups))
    if folds < 2:
        raise ValueError("Inner validation needs at least two eligible split units.")
    if split_strategy == "combination":
        mapping = _combination_validation_folds(groups, folds, random_seed)
    else:
        mapping = _balanced_group_folds(groups, folds, random_seed)
    allocation_metadata: dict[str, Any] = {
        "allocation_method": allocation_method,
        "reference_retention_objective_used": allocation_method == "evidence_balanced",
    }
    if allocation_method == "evidence_balanced":
        if reference_manifest is None:
            raise ValueError("evidence_balanced inner allocation requires an outer-training reference manifest.")
        mapping, allocation_metadata = _evidence_balanced_folds(
            df_training,
            split_strategy,
            groups,
            mapping,
            reference_manifest,
            numeric_bins,
            random_seed,
        )
    elif reference_manifest is not None:
        # The matching random layout receives the same diagnostic calculation.
        # It is reported for a paired comparison but is never optimized.
        objective = _ReferenceRetentionObjective(
            df_training, split_strategy, reference_manifest, numeric_bins
        )
        random_values = _unit_folds_from_mapping(groups, mapping)
        random_frame, random_objective = objective.evaluate(random_values)
        allocation_metadata.update({
            "reference_scope": "complete_frozen_outer_training_cohort",
            "reference_manifest_feature_count": int(len(objective.manifest)),
            "numeric_value_policy": f"fixed_reference_values_or_{numeric_bins}_quantile_bins",
            "objective_components": [metric for _, metric in _RETENTION_COMPONENTS],
            "block_aggregation": "mean_across_fixed_features; blocks_and_components_equally_weighted",
            "fold_objective": "mean_of_finite_block_component_retentions",
            "optimization_rule": "not_applied_seeded_random_baseline",
            "baseline_objective_loss": _objective_loss_record(random_objective),
            "optimized_objective_loss": _objective_loss_record(random_objective),
            "candidate_moves_considered": 0,
            "moves_applied": 0,
            "optimization_status": "random_baseline",
            "baseline_fold_diagnostics": random_frame.to_dict(orient="records"),
            "optimized_fold_diagnostics": random_frame.to_dict(orient="records"),
        })
    values = units.astype(str).map(mapping).fillna(-1).to_numpy(dtype=int)
    assignments = pd.DataFrame({
        "row_position": np.arange(len(units)),
        "validation_fold": values,
        "split_unit": units.to_numpy(),
        "validation_eligible": values >= 0,
    })
    _validate_validation_assignments(df_training, split_strategy, assignments)
    assignments.attrs["reference_retention_allocation"] = allocation_metadata
    if "optimized_fold_diagnostics" in allocation_metadata:
        assignments.attrs["reference_retention_fold_diagnostics"] = pd.DataFrame(
            allocation_metadata["optimized_fold_diagnostics"]
        )
    return assignments


def validation_diagnostics(df_training: pd.DataFrame, assignments: pd.DataFrame, target: str) -> pd.DataFrame:
    work = assignments.copy()
    work[target] = pd.to_numeric(df_training[target].to_numpy(), errors="coerce")
    rows = []
    for fold in sorted(value for value in work["validation_fold"].unique() if value >= 0):
        validation = work.loc[work["validation_fold"].eq(fold)]
        values = validation[target].dropna()
        rows.append({
            "validation_fold": fold,
            "validation_rows": int(len(validation)),
            "validation_split_units": int(validation["split_unit"].nunique()),
            "training_rows": int(len(work) - len(validation)),
            "training_only_rows": int(work["validation_fold"].eq(-1).sum()),
            "target_mean": values.mean(),
            "target_sd": values.std(ddof=1),
        })
    diagnostics = pd.DataFrame(rows)
    retention = assignments.attrs.get("reference_retention_fold_diagnostics")
    if isinstance(retention, pd.DataFrame) and not retention.empty:
        diagnostics = diagnostics.merge(
            retention,
            on="validation_fold",
            how="left",
            validate="one_to_one",
        )
    return diagnostics
