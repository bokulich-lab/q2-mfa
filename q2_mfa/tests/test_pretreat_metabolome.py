# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import time
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.testing as pdt
from numpy.testing import assert_allclose
from rachis import CategoricalMetadataColumn
from rachis.core.exceptions import RachisWarning

from q2_mfa.pretreat_metabolome import (
    impute_table,
    normalize_pqn,
    normalize_tic,
    pretreat_metabolome,
    remove_missing_features,
    resolve_capture_holder,
    scale_table,
    transform_table,
)


class TestPretreatMetabolome(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = pd.DataFrame(
            {
                "F1": [0.0, 1.0, 2.0],
                "F2": [3.0, 0.0, 4.0],
                "F3": [5.0, 6.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )

        cls.table_with_missing_feature = cls.table.assign(F4=np.nan)

        cls.negative_table = pd.DataFrame(
            {
                "F1": [0.0, -1.0, 2.0],
                "F2": [3.0, 4.0, 5.0],
            },
            index=["S1", "S2", "S3"],
        )

        cls.no_zero_table = pd.DataFrame(
            {
                "F1": [1.0, 2.0, 3.0],
                "F2": [4.0, 5.0, 6.0],
                "F3": [7.0, 8.0, 9.0],
            },
            index=["S1", "S2", "S3"],
        )

        cls.zero_sum_table = pd.DataFrame(
            {
                "F1": [0.0, 0.0],
                "F2": [0.0, 0.0],
            },
            index=["S1", "S2"],
        )

        cls.nan_sum_table = pd.DataFrame(
            {
                "F1": [np.nan, 1.0],
                "F2": [np.nan, 2.0],
            },
            index=["S1", "S2"],
        )

        cls.zero_factor_table = pd.DataFrame(
            {
                "F1": [0.0, 10.0, 10.0],
                "F2": [0.0, 20.0, 20.0],
            },
            index=["S0", "S1", "S2"],
        )

        cls.no_variance_table = pd.DataFrame(
            {
                "F1": [7.0, 7.0, 7.0],
                "F2": [1.0, 2.0, 3.0],
                "F3": [2.0, 2.0, 5.0],
            },
            index=["S1", "S2", "S3"],
        )

        cls.ref_metadata = CategoricalMetadataColumn(
            pd.Series(
                {"S1": "control", "S2": "control", "S3": "treatment"},
                name="group",
            ).rename_axis("id")
        )

    def test_centering(self):
        out = pretreat_metabolome(
            self.table,
            sample_normalization=None,
            transform=None,
            scale="center",
        )

        expected = self.table - self.table.mean(axis=0)
        pdt.assert_frame_equal(out, expected)

    def test_tic_sample_normalization(self):
        out = pretreat_metabolome(
            self.table,
            sample_normalization="tic",
            transform=None,
            scale=None,
        )

        expected = self.table.div(self.table.sum(axis=1), axis=0)
        pdt.assert_frame_equal(out, expected)

    def test_preserves_shape_index_columns(self):
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pretreat_metabolome(
                self.table,
                sample_normalization="pqn",
                impute="knn",
                transform="log",
                pseudocount=1e-6,
                scale="auto",
            )
        duration = time.perf_counter() - start
        print(f"KNN pretreatment R script duration: {duration:.3f}s")

        self.assert_table_shape_index_columns(out, self.table)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RachisWarning)
        self.assertEqual(
            str(caught[0].message),
            "Pseudocount was provided, but no zero values were found; "
            "the pseudocount was not applied.",
        )

    def test_autoscale(self):
        out = scale_table(self.table, scale="auto")

        X = self.table
        sd = X.std(axis=0, ddof=0)
        expected = (X - X.mean(axis=0)) / sd

        self.assert_table_shape_index_columns(out, self.table)
        pdt.assert_frame_equal(out, expected)

    def test_pareto_scaling(self):
        out = scale_table(self.table, scale="pareto")

        X = self.table
        sd = X.std(axis=0, ddof=0)
        expected = (X - X.mean(axis=0)) / np.sqrt(sd)

        pdt.assert_frame_equal(out, expected)

    def test_range_scaling(self):
        out = scale_table(self.table, scale="range")

        X = self.table
        rng = X.max(axis=0) - X.min(axis=0)
        expected = (X - X.mean(axis=0)) / rng

        pdt.assert_frame_equal(out, expected)

    def test_autoscale_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Autoscaling not possible: at least one feature has zero variance\.",
        ):
            scale_table(self.no_variance_table, scale="auto")

    def test_pareto_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Pareto scaling not possible: at least one feature has zero variance\.",
        ):
            scale_table(self.no_variance_table, scale="pareto")

    def test_range_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Range scaling not possible: at least one feature has zero range\.",
        ):
            scale_table(self.no_variance_table, scale="range")

    def test_transform_log_fixed_pseudocount(self):
        out = transform_table(
            self.table,
            transform="log",
            pseudocount=0.5,
        )

        expected = np.log(self.table + 0.5)
        self.assert_table_shape_index_columns(out, self.table)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_log10_fixed_pseudocount(self):
        out = transform_table(
            self.table,
            transform="log10",
            pseudocount=0.5,
        )

        expected = np.log10(self.table + 0.5)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_sqrt_fixed_pseudocount(self):
        out = transform_table(
            self.table,
            transform="sqrt",
        )

        expected = np.sqrt(self.table)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_log_none_pseudocount(self):
        out = transform_table(
            self.table,
            transform="log",
            pseudocount=None,
        )

        expected = np.log(self.table + 0.5)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_log_does_not_apply_pseudocount_without_zeros(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = transform_table(
                self.no_zero_table,
                transform="log",
                pseudocount=0.5,
            )

        expected = np.log(self.no_zero_table)
        assert_allclose(out.to_numpy(), expected)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RachisWarning)
        self.assertEqual(
            str(caught[0].message),
            "Pseudocount was provided, but no zero values were found; "
            "the pseudocount was not applied.",
        )

    def test_log_raises_on_negative_values(self):
        with self.assertRaisesRegex(
            ValueError, "Transformation requires all values to be non negative."
        ):
            transform_table(
                self.negative_table,
                transform="log",
                pseudocount=1e-6,
            )

    def test_log_raises_when_pseudocount_cannot_be_inferred(self):
        with self.assertRaisesRegex(
            ValueError,
            "Log transformation requires at least one positive value "
            "to infer a pseudocount.",
        ):
            transform_table(
                self.zero_sum_table,
                transform="log",
                pseudocount=None,
            )

    def test_transform_sqrt_exact(self):
        out = transform_table(
            self.table,
            transform="sqrt",
        )

        expected = np.sqrt(self.table.to_numpy())
        assert_allclose(out.to_numpy(), expected, rtol=0, atol=1e-12)

    def test_tic_normalization(self):
        out = normalize_tic(self.table)

        expected = self.table.div(self.table.sum(axis=1), axis=0)
        self.assert_table_shape_index_columns(out, self.table)
        pdt.assert_frame_equal(out, expected)

    def test_tic_raises_on_negative_values(self):
        with self.assertRaisesRegex(ValueError, r"non-negative intensities"):
            normalize_tic(self.negative_table)

    def test_tic_raises_on_zero_sample_sum(self):
        with self.assertRaisesRegex(ValueError, r"zero sample sum"):
            normalize_tic(self.zero_sum_table)

    def test_tic_raises_on_all_nan_sample_sum(self):
        with self.assertRaisesRegex(ValueError, r"zero sample sum"):
            normalize_tic(self.nan_sum_table)

    def test_pqn_raises_on_negative_values(self):
        with self.assertRaisesRegex(ValueError, r"non-negative intensities"):
            normalize_pqn(self.negative_table, method="median")

    def test_pqn_raises_on_nan_factor_all_zeros(self):
        with self.assertRaisesRegex(ValueError, r"could not compute dilution factor"):
            normalize_pqn(self.zero_sum_table, method="median")

    def test_pqn_raises_on_non_positive_factor(self):
        with self.assertRaisesRegex(ValueError, r"non-positive dilution factor"):
            normalize_pqn(self.zero_factor_table, method="median")

    def test_pqn_median(self):
        out = normalize_pqn(self.table, method="median")

        expected = pd.DataFrame(
            {
                "F1": [0.0, 1.0, 1.5],
                "F2": [3.0, 0.0, 3.0],
                "F3": [5.0, 6.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )
        self.assert_table_shape_index_columns(out, self.table)
        pdt.assert_frame_equal(out, expected)

    def test_pqn_mean(self):
        out = normalize_pqn(self.table, method="mean")

        expected = pd.DataFrame(
            {
                "F1": [0.0, 1.0, 7 / 6],
                "F2": [7 / 3, 0.0, 7 / 3],
                "F3": [35 / 9, 6.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )
        pdt.assert_frame_equal(out, expected)

    def test_pqn_metadata(self):
        out = normalize_pqn(
            self.table,
            method="median",
            ref_samples=self.ref_metadata,
            ref_label="control",
        )

        expected = pd.DataFrame(
            {
                "F1": [0.0, 11 / 12, 3 / 4],
                "F2": [33 / 10, 0.0, 3 / 2],
                "F3": [11 / 2, 11 / 2, 0.0],
            },
            index=["S1", "S2", "S3"],
        )

        pdt.assert_frame_equal(out, expected)

    def test_pqn_metadata_error(self):
        with self.assertRaisesRegex(
            ValueError, r"Reference label 'qc' not found in metadata column"
        ):
            normalize_pqn(
                self.table,
                method="median",
                ref_samples=self.ref_metadata,
                ref_label="qc",
            )

    def test_pqn_metadata_requires_label(self):
        with self.assertRaisesRegex(
            ValueError,
            r"PQN reference metadata and reference label must be provided together",
        ):
            normalize_pqn(
                self.table,
                method="median",
                ref_samples=self.ref_metadata,
                ref_label=None,
            )

    def test_knn_impute(self):
        out = impute_table(self.table, impute="knn", knn_neighbors=2)

        self.assert_table_shape_index_columns(out, self.table)
        self.assertFalse(out.isna().to_numpy().any())
        self.assertFalse((out.to_numpy() == 0.0).any())

    def test_qrilc_impute(self):
        out = impute_table(self.table, impute="qrilc")

        self.assert_table_shape_index_columns(out, self.table)
        self.assertFalse(out.isna().to_numpy().any())
        self.assertFalse((out.to_numpy() == 0.0).any())

    def test_miss_forest_impute(self):
        with self.assertWarnsRegex(
            RachisWarning,
            "Removed 1 features with no observed values.",
        ):
            out = impute_table(
                self.table_with_missing_feature,
                impute="miss_forest",
                mf_ntree=2,
                mf_random_state=42,
            )

        self.assert_table_shape_index_columns(out, self.table)
        self.assertFalse(out.isna().to_numpy().any())
        self.assertFalse((out.to_numpy() == 0.0).any())

    def test_remove_missing_features(self):
        with self.assertWarnsRegex(
            RachisWarning,
            "Removed 1 features with no observed values.",
        ):
            out = remove_missing_features(self.table_with_missing_feature)

        pdt.assert_frame_equal(out, self.table)

    @patch("q2_mfa.pretreat_metabolome.run_r_table_script")
    def test_qrilc_imputes_in_log2_space_and_returns_original_scale(self, mock_run):
        expected_log2_table = np.log2(self.table.replace(0, np.nan))
        mock_run.return_value = expected_log2_table.T

        out = impute_table(self.table, impute="qrilc")

        mock_run.assert_called_once()
        pdt.assert_frame_equal(
            mock_run.call_args.kwargs["table"], expected_log2_table.T
        )
        pdt.assert_frame_equal(self.table.replace(0, np.nan), out)

    def test_pretreat_metabolome_miss_forest_imputation(self):
        out = pretreat_metabolome(
            self.table,
            sample_normalization=None,
            transform=None,
            scale=None,
            impute="miss_forest",
        )

        self.assert_table_shape_index_columns(out, self.table)
        self.assertFalse(out.isna().to_numpy().any())
        self.assertFalse((out.to_numpy() == 0.0).any())

    def test_resolve_capture_holder_returns_none(self):
        self.assertIsNone(resolve_capture_holder(None, random=False))

    def test_resolve_capture_holder_preserves_explicit_seed(self):
        self.assertEqual(resolve_capture_holder(42, random=True), 42)

    def assert_table_shape_index_columns(self, observed, expected):
        self.assertIsInstance(observed, pd.DataFrame)
        self.assertEqual(observed.shape, expected.shape)
        self.assertListEqual(list(observed.index), list(expected.index))
        self.assertListEqual(list(observed.columns), list(expected.columns))


if __name__ == "__main__":
    unittest.main()
