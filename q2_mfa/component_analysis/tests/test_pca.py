# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import os
import warnings

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import prince
from rachis.core.exceptions import RachisWarning
from rachis.plugin.testing import TestPluginBase

from q2_mfa.component_analysis.pca import (
    create_result_object,
    drop_columns_with_missing_values,
    drop_zero_variance_columns,
    pca,
    resolve_random_state,
)
from q2_mfa.component_analysis.types import ComponentAnalysisResult


class TestPCA(TestPluginBase):
    package = "q2_mfa.component_analysis.tests"

    def setUp(self):
        super().setUp()
        self.table = pd.read_csv(
            self.get_data_path("pca/random_data.txt"), sep="\t", index_col=0
        )
        self.missing_columns_table = pd.read_csv(
            self.get_data_path("pca/missing_columns.tsv"), sep="\t", index_col=0
        )
        self.zero_variance_columns_table = pd.read_csv(
            self.get_data_path("pca/zero_variance_columns.tsv"),
            sep="\t",
            index_col=0,
        )

    def test_pca_outputs_match_prince_regression_fixtures(self):
        expected_dir = self.get_data_path("pca/prince-regression")
        results = pca(self.table, engine="scipy", filter_zero_variance=False)

        observed_vectors = {
            "eigenvalues": results.eigenvalues,
            "percentage_of_variance": results.percentage_of_variance,
            "cumulative_percentage_of_variance": (
                results.cumulative_percentage_of_variance
            ),
        }
        observed_tables = {
            "sample_coordinates": results.sample_coordinates.to_numpy(),
            "feature_coordinates": results.feature_coordinates.to_numpy(),
            "sample_cosine_similarities": (
                results.sample_cosine_similarities.to_numpy()
            ),
            "sample_contributions": results.sample_contributions.to_numpy(),
            "feature_correlations": results.feature_correlations.to_numpy(),
            "feature_contributions": results.feature_contributions.to_numpy(),
            "feature_cosine_similarities": (
                results.feature_cosine_similarities.to_numpy()
            ),
        }

        self.assertIsInstance(results, ComponentAnalysisResult)
        self.assertTrue(results.is_pca)
        for output_name, observed in observed_vectors.items():
            expected = np.loadtxt(os.path.join(expected_dir, f"{output_name}.tsv"))
            npt.assert_allclose(observed, expected)
        for output_name, observed in observed_tables.items():
            expected = pd.read_csv(
                os.path.join(expected_dir, f"{output_name}.tsv"),
                sep="\t",
                header=None,
            ).to_numpy()
            npt.assert_allclose(observed, expected)

    def test_create_result_object(self):
        prince_result = prince.PCA(engine="scipy").fit(self.table)

        observed = create_result_object(prince_result, self.table)

        self.assertIsInstance(observed, ComponentAnalysisResult)
        self.assertTrue(observed.is_pca)

    def test_drop_columns_with_missing_values_filters_and_warns(self):
        with self.assertWarns(RachisWarning) as warning:
            observed = drop_columns_with_missing_values(self.missing_columns_table)

        self.assertEqual(
            str(warning.warning),
            "Dropped columns with missing values: F1, F3",
        )
        self.assertEqual(list(observed.columns), ["F0", "F2"])

    def test_drop_columns_with_missing_values_keeps_complete_table(self):
        with warnings.catch_warnings(record=True) as observed_warnings:
            observed = drop_columns_with_missing_values(self.table)

        self.assertEqual(observed_warnings, [])
        pdt.assert_frame_equal(observed, self.table)

    def test_drop_zero_variance_columns_filters_and_warns(self):
        with self.assertWarns(RachisWarning) as warning:
            observed = drop_zero_variance_columns(self.zero_variance_columns_table)

        self.assertEqual(
            str(warning.warning),
            "Dropped columns with zero variance: F0, F3",
        )
        self.assertEqual(list(observed.columns), ["F1", "F2"])

    def test_drop_zero_variance_columns_keeps_varying_table(self):
        with warnings.catch_warnings(record=True) as observed_warnings:
            observed = drop_zero_variance_columns(self.table)

        self.assertEqual(observed_warnings, [])
        pdt.assert_frame_equal(observed, self.table)

    def test_pca_filters_zero_variance_columns(self):
        with self.assertWarns(RachisWarning) as warning:
            results = pca(self.zero_variance_columns_table, engine="scipy")

        self.assertEqual(
            str(warning.warning),
            "Dropped columns with zero variance: F0, F3",
        )
        self.assertEqual(list(results.feature_coordinates.index), ["F1", "F2"])

    def test_resolve_random_state_generates_seed_for_sklearn(self):
        random_state = resolve_random_state(None, "sklearn")

        self.assertIsInstance(random_state, int)
        self.assertGreaterEqual(random_state, 0)
        self.assertLess(random_state, 2**32)

    def test_resolve_random_state_uses_value_for_sklearn(self):
        random_state = resolve_random_state(14, "sklearn")

        self.assertEqual(random_state, 14)

    def test_resolve_random_state_uses_none_for_non_sklearn(self):
        random_state = resolve_random_state(None, "scipy")

        self.assertIsNone(random_state)
