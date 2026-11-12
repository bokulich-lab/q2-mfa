# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from rachis import Metadata
from rachis.plugin.testing import TestPluginBase
from rachis.sdk.usage import ExecutionUsage

import q2_mfa
from q2_mfa.pls import PLSComponentSelectionDirFmt, PLSFeatureSelectionDirFmt
from q2_mfa.plugin_setup import plugin


class TestUsageExampleCoverage(unittest.TestCase):
    def test_all_usage_examples_have_tests(self):
        """Check that every registered action example has a named test.

        Test methods must be named ``test_<action_id>_<example_name>``,
        using the action ID and example name exactly as registered on the
        plugin, for example
        ``test_select_features_block_splsda_nutrimouse``.
        """
        package_dir = Path(q2_mfa.__file__).resolve().parent
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(package_dir),
            pattern="test*.py",
            top_level_dir=str(package_dir.parent),
        )

        # Get names of all tests
        pending = [suite]
        test_names = set()
        while pending:
            test = pending.pop()
            if isinstance(test, unittest.TestSuite):
                pending.extend(test)
            else:
                test_names.add(test.id().rsplit(".", 1)[-1])

        # Check if all usage examples have a test
        missing = []
        for action_id, action in sorted(plugin.actions.items()):
            for example_name in sorted(action.examples):
                expected = f"test_{action_id}_{example_name}"
                if expected not in test_names:
                    missing.append(f"{action_id}/{example_name}: expected {expected}")
        self.assertFalse(
            missing,
            "Usage examples missing tests:\n" + "\n".join(missing),
        )


class TestUsageExamples(TestPluginBase):
    package = "q2_mfa.pls.tests"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        instance = cls()
        cls.expected_error_rates = pd.read_csv(
            instance.get_data_path("feature-selection/nutrimouse-error-rates.tsv"),
            sep="\t",
            index_col="id",
            dtype={"id": str},
        )
        cls.expected_selected_features = pd.read_csv(
            instance.get_data_path(
                "feature-selection/nutrimouse-selected-features.tsv"
            ),
            sep="\t",
            index_col="id",
            dtype={"id": str},
        )
        cls.expected_component_tuning = {}
        for vote in ("weighted", "majority"):
            for table in ("error-rates", "choices"):
                expected = pd.read_csv(
                    instance.get_data_path(
                        f"tune-components/nutrimouse-{vote}-{table}.tsv"
                    ),
                    sep="\t",
                    index_col="id",
                    dtype={"id": str, "component": float},
                )
                cls.expected_component_tuning[vote, table] = (
                    expected.astype(float) if table == "choices" else expected
                )

    def test_tune_components_block_splsda_nutrimouse(self):
        """Check the Nutrimouse tuning metrics, printed choices, and report."""
        use = ExecutionUsage()
        action = self.plugin.pipelines["tune_components_block_splsda"]
        stdout = StringIO()
        with redirect_stdout(stdout):
            action.examples["nutrimouse"](use)
        outputs = use.render()
        tuning = outputs["tuned_components_nutrimouse"].execute()
        data = tuning.view(PLSComponentSelectionDirFmt)

        expected_stdout = []
        for vote in ("weighted", "majority"):
            expected_rates = self.expected_component_tuning[vote, "error-rates"]
            expected_choices = self.expected_component_tuning[vote, "choices"]
            rates = getattr(data, f"error_rate_{vote}").view(Metadata).to_dataframe()
            choice_file = getattr(data, f"choice_matrix_{vote}")
            choices = choice_file.view(Metadata).to_dataframe()
            with self.subTest(vote=vote):
                assert_frame_equal(rates, expected_rates, rtol=1e-8, atol=1e-10)
                assert_frame_equal(choices, expected_choices, check_exact=True)
            printed_choices = expected_choices.astype("Int64").to_string()
            expected_stdout.append(
                f"{vote.capitalize()}Vote component-choice matrix:\n\n"
                f"{printed_choices}\n\n"
            )
        self.assertEqual(stdout.getvalue(), "".join(expected_stdout))

        report = outputs["component_tuning_report"].execute()
        with (report._archiver.data_dir / "subfigures" / "index.json").open() as fh:
            report_index = json.load(fh)
        self.assertEqual(list(report_index), ["Error rates", "Component choices"])
        for section in ("Error rates", "Component choices"):
            self.assertEqual(
                list(report_index[section]["children"]),
                ["Weighted vote", "Majority vote"],
            )

    def test_select_features_block_splsda_nutrimouse(self):
        use = ExecutionUsage()
        action = self.plugin.pipelines["select_features_block_splsda"]
        stdout = StringIO()
        with redirect_stdout(stdout):
            action.examples["nutrimouse"](use)
        expected_printed_features = self.expected_selected_features.reset_index(
            drop=True
        ).astype(int)
        self.assertEqual(
            stdout.getvalue(),
            f"\nTuned keepX values:\n\n{expected_printed_features}\n\n",
        )
        outputs = use.render()
        selection = outputs["feature_selection_nutrimouse"].execute()
        data = selection.view(PLSFeatureSelectionDirFmt)
        error_rates = data.error_rate.view(Metadata).to_dataframe()
        selected_features = data.selected_features.view(Metadata).to_dataframe()
        assert_frame_equal(
            error_rates,
            self.expected_error_rates,
            rtol=1e-8,
            atol=1e-10,
        )
        assert_frame_equal(selected_features, self.expected_selected_features)

        report = outputs["feature_selection_report"].execute()
        with (report._archiver.data_dir / "subfigures" / "index.json").open() as fh:
            report_index = json.load(fh)
        self.assertEqual(
            list(report_index),
            ["Error-rate heatmaps", "Error rates", "Selected features"],
        )
        self.assertEqual(
            list(report_index["Error-rate heatmaps"]["children"]),
            ["Component 1", "Component 2"],
        )
