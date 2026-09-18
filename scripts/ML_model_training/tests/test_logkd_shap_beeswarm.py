"""Checks for the representative-repeat SHAP beeswarm and its run selection."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import logkd_shap_outer_robustness as robustness  # noqa: E402
from backend.logkd_shap_common import (  # noqa: E402
    BeeswarmDisplay,
    PipelineExplanation,
    BEESWARM_MIN_POINT_AREA,
    BEESWARM_POINT_AREA,
    beeswarm_point_areas,
    beeswarm_row_offsets,
    build_beeswarm_display,
    level_display_label,
    make_feature_map,
    original_importance_summary,
    percentile_scaled_colors,
    save_beeswarm,
)


def unit_weights(explanation: PipelineExplanation) -> pd.Series:
    """Weights for a fixture whose rows are all unexpanded source records."""

    return pd.Series(np.ones(len(explanation.X)))


def write_testing_metrics(run_dir: Path, **metrics: float) -> Path:
    """Write the one trainer artifact the representative selection reads."""

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"split": "training_refit", "mae": 0.05, "rmse": 0.08, "r2": 0.99, "spearman": 0.99},
            {"split": "testing", **metrics},
        ]
    ).to_csv(run_dir / "metrics_summary.csv", index=False)
    return run_dir


class RepresentativeRunSelectionTests(unittest.TestCase):
    def test_selects_the_repeat_closest_to_the_mean_of_every_metric(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = [
                write_testing_metrics(root / "run_0", mae=0.5, rmse=0.7, r2=0.60, spearman=0.80),
                write_testing_metrics(root / "run_1", mae=0.4, rmse=0.6, r2=0.70, spearman=0.85),
                write_testing_metrics(root / "run_2", mae=0.3, rmse=0.5, r2=0.80, spearman=0.90),
            ]
            selection = robustness.representative_run_table(run_dirs)
            self.assertEqual(int(selection["selected"].sum()), 1)
            self.assertTrue(bool(selection.loc[1, "selected"]))
            self.assertEqual(robustness._selected_run_dir(selection), run_dirs[1])
            self.assertAlmostEqual(float(selection.loc[1, "typicality_distance"]), 0.0, places=9)
            self.assertEqual(
                selection.loc[0, "metrics_used_for_selection"], "rmse, mae, r2, spearman"
            )

    def test_constant_metrics_do_not_standardize_floating_point_noise(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = [
                write_testing_metrics(
                    root / f"run_{index}", mae=0.4, rmse=0.6, r2=0.7, spearman=0.85
                )
                for index in range(3)
            ]
            selection = robustness.representative_run_table(run_dirs)
            self.assertTrue(selection["metrics_used_for_selection"].eq("").all())
            self.assertTrue(selection["typicality_distance"].isna().all())
            self.assertTrue(bool(selection.loc[0, "selected"]))

    def test_metrics_missing_from_any_repeat_are_excluded_from_the_score(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = [
                write_testing_metrics(root / "run_0", mae=0.3, rmse=0.6),
                write_testing_metrics(root / "run_1", mae=0.4, rmse=0.6),
                write_testing_metrics(root / "run_2", mae=0.5, rmse=0.6),
            ]
            selection = robustness.representative_run_table(run_dirs)
            self.assertTrue(selection["metrics_used_for_selection"].eq("mae").all())
            self.assertTrue(bool(selection.loc[1, "selected"]))

    def test_absent_metrics_summary_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                robustness.representative_run_table([Path(directory) / "missing_run"])


class OutputDirectoryTests(unittest.TestCase):
    def test_a_new_directory_is_created(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "shap" / "scenario"
            self.assertEqual(robustness.prepare_output_dir(target, overwrite=False), [])
            self.assertTrue(target.is_dir())

    def test_an_existing_directory_is_refused_without_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                robustness.prepare_output_dir(Path(directory), overwrite=False)

    def test_rerunning_an_analysis_replaces_it_by_default(self) -> None:
        """A rerun exists to correct an earlier run, so it must not need a flag."""

        signature = inspect.signature(robustness.generate_robustness_summary)
        self.assertIs(signature.parameters["overwrite"].default, True)
        with patch.object(sys, "argv", ["prog", "--run-dirs", "a", "b", "--output-dir", "out"]):
            self.assertTrue(robustness.parse_args().overwrite)
        with patch.object(
            sys, "argv", ["prog", "--run-dirs", "a", "b", "--output-dir", "out", "--no-overwrite"]
        ):
            self.assertFalse(robustness.parse_args().overwrite)

    def test_overwrite_replaces_only_this_workflow_s_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "shap_robustness_summary.csv").write_text("stale", encoding="utf-8")
            (target / "shap_representative_beeswarm.png").write_text("stale", encoding="utf-8")
            (target / "shap_retired_output.csv").write_text("stale schema", encoding="utf-8")
            (target / "notes.md").write_text("keep me", encoding="utf-8")
            (target / "subfolder").mkdir()

            replaced = robustness.prepare_output_dir(target, overwrite=True)

            self.assertEqual(
                replaced,
                [
                    "shap_representative_beeswarm.png",
                    "shap_retired_output.csv",
                    "shap_robustness_summary.csv",
                ],
            )
            self.assertFalse((target / "shap_robustness_summary.csv").exists())
            self.assertTrue((target / "notes.md").exists())
            self.assertTrue((target / "subfolder").is_dir())


class BeeswarmGeometryTests(unittest.TestCase):
    def test_offsets_are_deterministic_centred_and_bounded(self) -> None:
        values = np.random.default_rng(0).normal(size=200)
        first = beeswarm_row_offsets(values)
        self.assertTrue(np.array_equal(first, beeswarm_row_offsets(values)))
        self.assertLessEqual(float(np.abs(first).max()), 0.4 + 1e-9)
        self.assertAlmostEqual(float(first.sum()), 0.0, places=6)

    def test_degenerate_inputs_stay_inside_the_row_band(self) -> None:
        self.assertEqual(len(beeswarm_row_offsets(np.zeros(0))), 0)
        constant = beeswarm_row_offsets(np.full(9, 2.0))
        self.assertLessEqual(float(np.abs(constant).max()), 0.4 + 1e-9)

    def test_only_numeric_feature_values_receive_a_colour_position(self) -> None:
        scaled = percentile_scaled_colors(pd.Series([1.0, 2.0, 3.0, np.nan, 100.0]))
        self.assertTrue(np.isnan(scaled[3]))
        finite = scaled[np.isfinite(scaled)]
        self.assertGreaterEqual(float(finite.min()), 0.0)
        self.assertLessEqual(float(finite.max()), 1.0)
        self.assertTrue(np.isnan(percentile_scaled_colors(pd.Series(["GAC", "PAC", "GAC"]))).all())


def build_explanation(
    rows: int = 120,
) -> tuple[PipelineExplanation, list[str], list[str]]:
    """Build a small explanation whose layout mirrors the fitted pipeline's.

    One numeric feature carries missing values, and one three-level categorical
    is one-hot encoded the way the fitted ColumnTransformer encodes it.
    """

    rng = np.random.default_rng(7)
    numeric = ["dosage", "pore_volume"]
    categorical = ["raw_material"]
    levels = ["coconut", "coal", "<missing>"]
    processed_names = [*numeric, *(f"raw_material_{level}" for level in levels)]

    X = pd.DataFrame(
        {
            "dosage": rng.normal(size=rows),
            "pore_volume": rng.normal(size=rows),
            "raw_material": rng.choice(levels, size=rows),
        }
    )
    X.loc[rng.random(rows) < 0.25, "pore_volume"] = np.nan
    indicators = np.stack([(X["raw_material"] == level).to_numpy(dtype=float) for level in levels], axis=1)
    X_processed = np.column_stack(
        [np.nan_to_num(X["dosage"].to_numpy()), np.nan_to_num(X["pore_volume"].to_numpy()), indicators]
    )
    processed_values = rng.normal(
        0, np.array([0.6, 0.3, 0.2, 0.15, 0.1]), size=(rows, len(processed_names))
    )
    explanation = PipelineExplanation(
        X=X,
        X_processed=X_processed,
        processed_values=processed_values,
        feature_map=make_feature_map(processed_names, [*numeric, *categorical], numeric, categorical),
        original_values=pd.DataFrame(),
        base_values=np.full(rows, 2.0),
        predictions=np.linspace(0.0, 1.0, rows),
        additivity_predictions=np.linspace(0.0, 1.0, rows),
        additivity_reference="test",
        backend="test",
        max_additivity_error=0.0,
    )
    return explanation, numeric, categorical


class CategoricalLevelDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explanation, self.numeric, self.categorical = build_explanation()

    def test_levels_become_their_own_fully_coloured_rows(self) -> None:
        display = build_beeswarm_display(
            self.explanation, self.categorical, sample_weight=unit_weights(self.explanation)
        )
        level_rows = display.layout.loc[display.layout["row_kind"].eq("categorical_level")]
        self.assertEqual(
            sorted(level_rows["display_feature"]),
            [
                "raw_material = <missing>",
                "raw_material = coal",
                "raw_material = coconut",
            ],
        )
        # Every one-hot point is 0 or 1, so no categorical point stays grey.
        for feature in level_rows["display_feature"]:
            colors = display.color_positions[feature]
            self.assertFalse(colors.isna().any())
            self.assertEqual(sorted(colors.unique().tolist()), [0.0, 1.0])

    def test_level_colour_marks_exactly_the_rows_holding_that_category(self) -> None:
        display = build_beeswarm_display(
            self.explanation, self.categorical, sample_weight=unit_weights(self.explanation)
        )
        expected = (self.explanation.X["raw_material"] == "coconut").to_numpy(dtype=float)
        np.testing.assert_array_equal(
            display.color_positions["raw_material = coconut"].to_numpy(dtype=float), expected
        )

    def test_level_shap_values_sum_to_the_aggregated_categorical_row(self) -> None:
        split = build_beeswarm_display(
            self.explanation, self.categorical, sample_weight=unit_weights(self.explanation)
        )
        aggregated = build_beeswarm_display(
            self.explanation,
            self.categorical,
            sample_weight=unit_weights(self.explanation),
            split_categorical_levels=False,
        )
        levels = [name for name in split.values.columns if name.startswith("raw_material = ")]
        np.testing.assert_allclose(
            split.values[levels].sum(axis=1).to_numpy(),
            aggregated.values["raw_material"].to_numpy(),
        )

    def test_grey_is_reserved_for_missing_numeric_values(self) -> None:
        display = build_beeswarm_display(
            self.explanation, self.categorical, sample_weight=unit_weights(self.explanation)
        )
        uncoloured = display.color_positions.isna()
        self.assertFalse(uncoloured["dosage"].any())
        self.assertEqual(
            int(uncoloured["pore_volume"].sum()),
            int(self.explanation.X["pore_volume"].isna().sum()),
        )
        level_columns = [name for name in display.color_positions.columns if " = " in name]
        self.assertFalse(uncoloured[level_columns].to_numpy().any())

    def test_aggregated_categorical_rows_stay_uncoloured(self) -> None:
        display = build_beeswarm_display(
            self.explanation,
            self.categorical,
            sample_weight=unit_weights(self.explanation),
            split_categorical_levels=False,
        )
        self.assertTrue(display.color_positions["raw_material"].isna().all())
        self.assertEqual(list(display.values.columns), ["dosage", "pore_volume", "raw_material"])

    def test_infrequent_grouped_level_is_labelled_readably(self) -> None:
        self.assertEqual(
            level_display_label("raw_material_infrequent_sklearn", "raw_material"),
            "raw_material = rare levels (grouped)",
        )
        self.assertEqual(level_display_label("raw_material_coconut", "raw_material"), "raw_material = coconut")


class BeeswarmPointAreaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explanation, self.numeric, self.categorical = build_explanation()
        self.rows = len(self.explanation.X)

    def test_omitted_weights_are_refused_rather_than_defaulted(self) -> None:
        # An unweighted SHAP result would contradict the weighted numbers it is
        # read against, so no entry point may reach one by omission.
        with self.assertRaises(ValueError):
            beeswarm_point_areas(None, 5)
        with self.assertRaises(ValueError):
            original_importance_summary(pd.DataFrame({"a": [1.0, 2.0]}), None)
        with self.assertRaises(TypeError):
            build_beeswarm_display(self.explanation, self.categorical)  # no weights
        with self.assertRaises(TypeError):
            original_importance_summary(pd.DataFrame({"a": [1.0, 2.0]}))  # no weights

    def test_an_unexpanded_record_keeps_the_nominal_point_area(self) -> None:
        areas = beeswarm_point_areas(np.ones(5), 5)
        self.assertTrue(np.array_equal(areas, np.full(5, BEESWARM_POINT_AREA)))

    def test_area_is_proportional_to_weight(self) -> None:
        weights = np.array([0.5, 1.0, 2.0])
        areas = beeswarm_point_areas(weights, 3)
        self.assertTrue(np.allclose(areas, BEESWARM_POINT_AREA * weights))
        # Equal ink: two half-weight rows spend one whole-weight row's area.
        self.assertAlmostEqual(2.0 * areas[0], areas[1], places=12)

    def test_a_heavily_expanded_record_stays_visible(self) -> None:
        areas = beeswarm_point_areas(np.array([1e-6, 2.0]), 2)
        self.assertEqual(areas[0], BEESWARM_MIN_POINT_AREA)

    def test_invalid_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            beeswarm_point_areas(np.array([1.0, 1.0]), 3)
        with self.assertRaises(ValueError):
            beeswarm_point_areas(np.array([1.0, -1.0]), 2)
        with self.assertRaises(ValueError):
            beeswarm_point_areas(np.array([1.0, np.nan]), 2)

    def test_display_carries_the_supplied_weights(self) -> None:
        weights = pd.Series(np.linspace(0.5, 1.5, self.rows))
        display = build_beeswarm_display(
            self.explanation, self.categorical, sample_weight=weights
        )
        self.assertIsNotNone(display.weights)
        self.assertTrue(np.allclose(display.weights.to_numpy(), weights.to_numpy()))
        self.assertTrue(display.weights.index.equals(display.values.index))

    def test_a_display_cannot_be_built_without_weights(self) -> None:
        with self.assertRaises(ValueError):
            build_beeswarm_display(
                self.explanation, self.categorical, sample_weight=None
            )

    def test_weighted_display_still_plots(self) -> None:
        display = build_beeswarm_display(
            self.explanation,
            self.categorical,
            sample_weight=pd.Series(np.linspace(0.5, 1.5, self.rows)),
        )
        order = display.values.abs().mean().sort_values(ascending=False).index.tolist()
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "beeswarm.png"
            save_beeswarm(
                display,
                output_path,
                title="Test beeswarm",
                max_display=3,
                feature_order=order,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class BeeswarmFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explanation, self.numeric, self.categorical = build_explanation()
        self.display = build_beeswarm_display(
            self.explanation, self.categorical, sample_weight=unit_weights(self.explanation)
        )
        self.rows = len(self.explanation.X)
        self.order = (
            self.display.values.abs().mean().sort_values(ascending=False).index.tolist()
        )

    def test_figure_honours_the_supplied_order_and_display_limit(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "beeswarm.png"
            displayed = save_beeswarm(
                self.display,
                output_path,
                title="Test beeswarm",
                max_display=3,
                feature_order=self.order,
                subtitle="120 held-out rows",
            )
            self.assertEqual(displayed, self.order[:3])
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_aggregated_display_also_plots(self) -> None:
        display = build_beeswarm_display(
            self.explanation,
            self.categorical,
            sample_weight=unit_weights(self.explanation),
            split_categorical_levels=False,
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "beeswarm.png"
            save_beeswarm(
                display,
                output_path,
                title="Test beeswarm",
                max_display=3,
                feature_order=list(display.values.columns),
            )
            self.assertTrue(output_path.exists())

    def test_ordering_a_feature_without_shap_values_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_beeswarm(
                    self.display,
                    Path(directory) / "beeswarm.png",
                    title="Test beeswarm",
                    max_display=3,
                    feature_order=[*self.order, "never_explained"],
                )

    def test_mismatched_colour_rows_are_rejected(self) -> None:
        broken = BeeswarmDisplay(
            values=self.display.values,
            color_positions=self.display.color_positions.head(10),
            layout=self.display.layout,
            weights=self.display.weights,
        )
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_beeswarm(
                    broken,
                    Path(directory) / "beeswarm.png",
                    title="Test beeswarm",
                    max_display=3,
                    feature_order=self.order,
                )

    def test_exported_row_values_cover_every_plotted_point(self) -> None:
        exported = robustness.representative_row_values(
            self.display,
            self.explanation.X,
            pd.DataFrame({"source_row_index": [f"s{index}" for index in range(self.rows)]}),
            pd.Series(np.ones(self.rows)),
            self.explanation.predictions,
            self.explanation.base_values,
        )
        plotted = list(self.display.values.columns)
        self.assertEqual(len(exported), self.rows * len(plotted))
        self.assertEqual(exported["source_row_index"].nunique(), self.rows)
        first_row = exported.loc[exported["testing_row_position"].eq(0)]
        self.assertEqual(first_row["feature"].tolist(), plotted)
        level = first_row.loc[first_row["feature"].eq("raw_material = coconut")].iloc[0]
        self.assertEqual(level["parent_feature"], "raw_material")
        self.assertEqual(level["level"], "coconut")
        self.assertEqual(level["row_kind"], "categorical_level")
        # The indicator colours the point; the raw category stays alongside it.
        self.assertEqual(
            float(level["feature_value"]),
            float(self.explanation.X.iloc[0]["raw_material"] == "coconut"),
        )
        self.assertEqual(level["model_input_value"], self.explanation.X.iloc[0]["raw_material"])


if __name__ == "__main__":
    unittest.main()
