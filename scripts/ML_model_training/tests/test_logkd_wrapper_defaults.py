"""Regression checks for the baseline used by active experiment wrappers."""

from __future__ import annotations

from argparse import Namespace
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_allocation as allocation  # noqa: E402
import run_logkd_algorithm_comparison as algorithm_comparison  # noqa: E402
import run_logkd_feature_family_only as feature_family  # noqa: E402
import run_logkd_feature_threshold_comparison as feature_threshold  # noqa: E402
import run_logkd_missing_strategy_comparison as missing_strategy  # noqa: E402
import run_logkd_split_strategy_comparison as split_strategy  # noqa: E402
from backend import logkd_config as cfg  # noqa: E402
import train_logkd_model as trainer  # noqa: E402


class WrapperDefaultsTests(unittest.TestCase):
    def test_feature_family_wrapper_combines_single_family_and_ablation_cases(self) -> None:
        configurations = feature_family.DEFAULT_FEATURE_FAMILY_CONFIGURATIONS
        self.assertEqual(
            set(configurations),
            {
                "full",
                "only_adsorbent_properties",
                "only_experimental_conditions",
                "only_pfas_characteristics",
                "without_adsorbent_properties",
                "without_experimental_conditions",
                "without_pfas_characteristics",
            },
        )
        self.assertEqual(
            configurations["without_adsorbent_properties"]["excluded_feature_buckets"],
            ["Adsorbent properties"],
        )
        self.assertEqual(
            configurations["without_experimental_conditions"]["excluded_feature_buckets"],
            ["Experimental conditions"],
        )
        self.assertEqual(
            configurations["without_pfas_characteristics"]["excluded_feature_buckets"],
            ["PFAS characteristics"],
        )
        self.assertEqual(
            feature_family.FEATURE_FAMILY_COMPARISON_PLOT_ORDER,
            (
                "full",
                "only_adsorbent_properties",
                "only_experimental_conditions",
                "only_pfas_characteristics",
                "without_adsorbent_properties",
                "without_experimental_conditions",
                "without_pfas_characteristics",
            ),
        )

        with patch.object(feature_family, "run_feature_policy_comparison") as runner:
            feature_family.main()

        self.assertEqual(runner.call_args.kwargs["default_configurations"], configurations)
        self.assertEqual(runner.call_args.kwargs["default_reference_configuration"], "full")
        self.assertEqual(
            runner.call_args.kwargs["automatic_plot_group_column"],
            "threshold_configuration",
        )
        self.assertEqual(
            runner.call_args.kwargs["automatic_plot_groups"],
            feature_family.FEATURE_FAMILY_COMPARISON_PLOT_ORDER,
        )

    def test_automatic_plot_settings_reach_the_figure_generator(self) -> None:
        configurations = {
            "full": {
                **feature_threshold.BASELINE_THRESHOLDS,
                "feature_screening_mode": "standard",
                "excluded_feature_buckets": [],
            },
        }
        args = Namespace(
            model="AC",
            reference_configuration="full",
            outer_repeats=1,
            n_trials=0,
            xgb_search_space="compact",
            random_seed=1,
            model_random_seed=1,
            test_fraction=0.2,
            validation_folds=5,
            reuse_outer_assignments_from=None,
        )
        with TemporaryDirectory() as directory, patch.object(
            feature_threshold,
            "write_testing_prediction_detail",
            return_value=None,
        ), patch.object(
            feature_threshold,
            "generate_aggregate_figure",
            return_value=["figures/configuration_performance_comparison.png"],
        ) as generator:
            feature_threshold.write_outputs(
                Path(directory),
                runs=pd.DataFrame(),
                assignments=pd.DataFrame(),
                discarded=pd.DataFrame(),
                feature_rows=pd.DataFrame(),
                configurations=configurations,
                args=args,
                failures=[],
                comparison_name="test",
                configuration_definition_filename="configuration_definitions.csv",
                include_excluded_feature_buckets=True,
                output_description="test comparison",
                automatic_plot_group_column="threshold_configuration",
                automatic_plot_groups=feature_family.FEATURE_FAMILY_COMPARISON_PLOT_ORDER,
            )

        self.assertEqual(
            generator.call_args.kwargs["group_column"],
            "threshold_configuration",
        )
        self.assertEqual(
            generator.call_args.kwargs["groups"],
            feature_family.FEATURE_FAMILY_COMPARISON_PLOT_ORDER,
        )

    def test_split_strategy_figures_use_the_concise_performance_summary(self) -> None:
        with TemporaryDirectory() as directory, patch.object(
            split_strategy,
            "generate_aggregate_figure",
            return_value=[],
        ) as generator:
            figure_paths, failures = split_strategy._generate_figures(
                Path(directory),
                prediction_target_label="logKd",
            )

        self.assertEqual(figure_paths, [])
        self.assertEqual(failures, [])
        comparison_calls = [
            call
            for call in generator.call_args_list
            if call.kwargs["plot_type"] == "comparison"
        ]
        self.assertEqual(len(comparison_calls), 2)
        self.assertTrue(all(
            call.kwargs["input_relative_path"]
            == Path("summary") / "performance_summary.csv"
            for call in comparison_calls
        ))
        self.assertFalse(any(
            call.kwargs["plot_type"] == "retention-profile"
            for call in generator.call_args_list
        ))

    def test_missing_strategy_figures_group_by_missing_value_strategy(self) -> None:
        args = Namespace(
            split_strategies=["random_row"],
            grouped_outer_allocation_methods=[],
        )
        with TemporaryDirectory() as directory, patch.object(
            missing_strategy,
            "generate_aggregate_figure",
            return_value=[],
        ) as generator:
            figure_paths, failures = missing_strategy._generate_figures(
                Path(directory),
                args,
            )

        self.assertEqual(figure_paths, [])
        self.assertEqual(failures, [])
        comparison_call = next(
            call
            for call in generator.call_args_list
            if call.kwargs["plot_type"] == "comparison"
        )
        self.assertEqual(
            comparison_call.kwargs["group_column"],
            "numeric_missing_strategy",
        )
        self.assertEqual(
            comparison_call.kwargs["groups"],
            missing_strategy.MISSING_STRATEGIES,
        )

    def test_every_active_wrapper_uses_the_central_baseline(self) -> None:
        parsers = {
            "allocation": allocation.parse_args,
            "algorithm comparison": algorithm_comparison.parse_args,
            "missing-strategy comparison": missing_strategy.parse_args,
            "split-strategy comparison": split_strategy.parse_args,
        }
        for name, parser in parsers.items():
            with self.subTest(wrapper=name), patch.object(sys, "argv", ["wrapper.py"]):
                args = parser()
                self.assertEqual(args.outer_repeats, cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS)
                self.assertEqual(args.n_trials, cfg.DEFAULT_EXPERIMENT_N_TRIALS)
                self.assertEqual(
                    args.early_stop_se_multiplier,
                    cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER,
                )
                self.assertEqual(
                    args.early_stop_min_delta_floor,
                    cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR,
                )
                self.assertEqual(args.data_mode, cfg.DEFAULT_DATA_MODE)

        with patch.object(sys, "argv", ["wrapper.py"]):
            threshold_args = feature_threshold.parse_args("test parser")
        self.assertEqual(
            threshold_args.outer_repeats,
            cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS,
        )
        self.assertEqual(threshold_args.n_trials, cfg.DEFAULT_EXPERIMENT_N_TRIALS)
        self.assertEqual(
            threshold_args.early_stop_se_multiplier,
            cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER,
        )
        self.assertEqual(
            threshold_args.early_stop_min_delta_floor,
            cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR,
        )
        self.assertEqual(threshold_args.data_mode, cfg.DEFAULT_DATA_MODE)

    def test_comparison_scope_defaults_to_random_row(self) -> None:
        for name, parser in {
            "algorithm comparison": algorithm_comparison.parse_args,
            "missing-strategy comparison": missing_strategy.parse_args,
        }.items():
            with self.subTest(wrapper=name), patch.object(sys, "argv", ["wrapper.py"]):
                self.assertEqual(parser().split_strategies, ["random_row"])

        self.assertEqual(
            algorithm_comparison.NUMERIC_MISSING_STRATEGY,
            cfg.ALGORITHM_COMPARISON_NUMERIC_MISSING_STRATEGY,
        )

        with patch.object(sys, "argv", ["wrapper.py"]):
            split_args = split_strategy.parse_args()
        self.assertEqual(
            split_args.grouped_outer_allocation_methods,
            list(cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS),
        )

        with patch.object(sys, "argv", ["wrapper.py"]):
            allocation_args = allocation.parse_args()
        self.assertEqual(
            allocation_args.grouped_outer_allocation_methods,
            list(cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS),
        )
        self.assertEqual(
            allocation._outer_allocation_methods_for_strategy(
                "random_row", allocation_args.grouped_outer_allocation_methods
            ),
            ("random_row",),
        )
        self.assertEqual(
            allocation._outer_allocation_methods_for_strategy(
                "study", allocation_args.grouped_outer_allocation_methods
            ),
            tuple(cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS),
        )

    def test_trainer_inherits_the_same_experiment_baseline(self) -> None:
        with patch.object(sys, "argv", ["trainer.py"]):
            args = trainer.parse_args(require_outer_allocation_method=False)
        self.assertEqual(args.n_trials, cfg.DEFAULT_EXPERIMENT_N_TRIALS)
        self.assertEqual(
            args.early_stop_se_multiplier,
            cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER,
        )
        self.assertEqual(
            args.early_stop_min_delta_floor,
            cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR,
        )
        self.assertEqual(args.data_mode, cfg.DEFAULT_DATA_MODE)
        self.assertEqual(
            args.numeric_missing_strategy,
            cfg.DEFAULT_EXPERIMENT_NUMERIC_MISSING_STRATEGY,
        )


if __name__ == "__main__":
    unittest.main()
