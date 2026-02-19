# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt
from numpy.testing import assert_allclose
from pandas._testing import assert_frame_equal
from rachis import CategoricalMetadataColumn
from rachis.plugin.testing import TestPluginBase

from q2_mfa.pretreat_metabolome import (
    impute_table,
    normalize_pqn,
    pretreat_metabolome,
    scale_table,
    transform_table,
)


class TestPretreatMetabolome(TestPluginBase):
    package = "q2_mfa"

    def setUp(self):
        super().setUp()
        self.df_nonneg = pd.DataFrame(
            {
                "F1": [0.0, 1.0, 2.0],
                "F2": [3.0, 0.0, 4.0],
                "F3": [5.0, 6.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )

    def test_centering(self):
        out = pretreat_metabolome(
            self.df_nonneg,
            sample_normalization=None,
            transform=None,
            center=True,
            scale=None,
        )

        expected = self.df_nonneg - self.df_nonneg.mean(axis=0)
        assert_frame_equal(out, expected)

    def test_preserves_shape_index_columns(self):
        out = pretreat_metabolome(
            self.df_nonneg,
            sample_normalization="pqn",
            impute="knn",
            transform="log",
            pseudocount=1e-6,
            center=False,
            scale="auto",
        )
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(out.shape, self.df_nonneg.shape)
        self.assertListEqual(list(out.index), list(self.df_nonneg.index))
        self.assertListEqual(list(out.columns), list(self.df_nonneg.columns))


class TestScaleTable(TestPluginBase):
    package = "q2_mfa"

    def setUp(self):
        super().setUp()

        self.df_nonneg = pd.DataFrame(
            {
                "F1": [0.0, 1.0, 2.0],
                "F2": [3.0, 0.0, 4.0],
                "F3": [5.0, 6.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )

        self.df_no_variance = pd.DataFrame(
            {
                "F1": [7.0, 7.0, 7.0],
                "F2": [1.0, 2.0, 3.0],
                "F3": [2.0, 2.0, 5.0],
            },
            index=["S1", "S2", "S3"],
        )

    def test_autoscale(self):
        out = scale_table(self.df_nonneg, scale="auto")

        X = self.df_nonneg
        sd = X.std(axis=0, ddof=0)
        expected = (X - X.mean(axis=0)) / sd

        assert_frame_equal(out, expected)

    def test_pareto_scaling(self):
        out = scale_table(self.df_nonneg, scale="pareto")

        X = self.df_nonneg
        sd = X.std(axis=0, ddof=0)
        expected = (X - X.mean(axis=0)) / np.sqrt(sd)

        assert_frame_equal(out, expected)

    def test_range_scaling(self):
        out = scale_table(self.df_nonneg, scale="range")

        # Expected manual computation
        X = self.df_nonneg
        rng = X.max(axis=0) - X.min(axis=0)
        expected = (X - X.mean(axis=0)) / rng

        assert_frame_equal(out, expected)

    def test_autoscale_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Autoscaling not possible: at least one feature has zero variance\.",
        ):
            scale_table(self.df_no_variance, scale="auto")

    def test_pareto_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Pareto scaling not possible: at least one feature has zero variance\.",
        ):
            scale_table(self.df_no_variance, scale="pareto")

    def test_range_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Range scaling not possible: at least one feature has zero range\.",
        ):
            scale_table(self.df_no_variance, scale="range")


class TestTransformTable(TestPluginBase):
    package = "q2_mfa"

    def setUp(self):
        super().setUp()
        self.df_has_negative = pd.DataFrame(
            {
                "F1": [0.0, -1.0, 2.0],
            },
            index=["S1", "S2", "S3"],
        )
        self.df_small = pd.DataFrame(
            {
                "F1": [0.0, 1.0],
                "F2": [3.0, 7.0],
            },
            index=["S1", "S2"],
        )

    def test_transform_log_fixed_pseudocount(self):
        out = transform_table(
            self.df_small,
            transform="log",
            pseudocount=0.5,
        )

        expected = np.log(self.df_small + 0.5)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_log10_fixed_pseudocount(self):
        out = transform_table(
            self.df_small,
            transform="log10",
            pseudocount=0.5,
        )

        expected = np.log10(self.df_small + 0.5)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_sqrt_fixed_pseudocount(self):
        out = transform_table(
            self.df_small,
            transform="sqrt",
        )

        expected = np.sqrt(self.df_small)
        assert_allclose(out.to_numpy(), expected)

    def test_transform_log_none_pseudocount(self):
        out = transform_table(
            self.df_small,
            transform="log",
            pseudocount=None,
        )

        expected = np.log(self.df_small + 1.0)
        assert_allclose(out.to_numpy(), expected)

    def test_log_raises_on_negative_values(self):
        with self.assertRaisesRegex(
            ValueError, "Transformation requires all values to be non negative."
        ):
            transform_table(
                self.df_has_negative,
                transform="log",
                pseudocount=1e-6,
            )

    def test_transform_sqrt_exact(self):
        out = transform_table(
            self.df_small,
            transform="sqrt",
        )

        expected = np.sqrt(self.df_small.to_numpy())
        assert_allclose(out.to_numpy(), expected, rtol=0, atol=1e-12)


class TestPQN(TestPluginBase):
    package = "q2_mfa"

    def setUp(self):
        super().setUp()
        self.df = pd.DataFrame(
            {
                "F1": [8.0, 4.0, 2.0],
                "F2": [16.0, 8.0, 3.0],
                "F3": [24.0, 12.0, 6.0],
            },
            index=["S1", "S2", "S3"],
        )

        self.df_negative = pd.DataFrame(
            {"F1": [1, -1], "F2": [2, 3]},
            index=["S1", "S2"],
        )

        self.df_all_zeros = pd.DataFrame(
            {"F1": [0, 0], "F2": [0, 0]},
            index=["S1", "S2"],
        )

        self.df_one_zero_sample = pd.DataFrame(
            {"F1": [0, 10, 10], "F2": [0, 20, 20]},
            index=["S0", "S1", "S2"],
        )

        self.ref_metadata = CategoricalMetadataColumn(
            pd.Series(
                {"S1": "control", "S2": "control", "S3": "treatment"},
                name="group",
            ).rename_axis("id")
        )

    def test_raises_on_negative_values(self):
        with self.assertRaisesRegex(ValueError, r"non-negative intensities"):
            normalize_pqn(self.df_negative, method="median")

    def test_raises_on_nan_factor_all_zeros(self):
        with self.assertRaisesRegex(ValueError, r"could not compute dilution factor"):
            normalize_pqn(self.df_all_zeros, method="median")

    def test_raises_on_non_positive_factor(self):
        with self.assertRaisesRegex(ValueError, r"non-positive dilution factor"):
            normalize_pqn(self.df_one_zero_sample, method="median")

    def test_pqn_median(self):
        out = normalize_pqn(self.df, method="median")

        exp = pd.DataFrame(
            {
                "F1": [4.0, 4.0, 4.0],
                "F2": [8.0, 8.0, 6.0],
                "F3": [12.0, 12.0, 12.0],
            },
            index=["S1", "S2", "S3"],
        )
        pdt.assert_frame_equal(out, exp)

    def test_pqn_mean(self):
        out = normalize_pqn(self.df, method="mean")

        exp = pd.DataFrame(
            {
                "F1": [14 / 3, 14 / 3, 14 / 3],
                "F2": [28 / 3, 28 / 3, 7.0],
                "F3": [14.0, 14.0, 14.0],
            },
            index=["S1", "S2", "S3"],
        )
        pdt.assert_frame_equal(out, exp)

    def test_pqn_metadata(self):
        out = normalize_pqn(
            self.df,
            method="median",
            ref_samples=self.ref_metadata,
            ref_label="control",
        )
        # Reference should be computed only from S1 and S2
        ref_expected = self.df.loc[["S1", "S2"]].median(axis=0)

        # Manually compute expected PQN normalization using the filtered reference
        ratios = self.df.div(ref_expected, axis=1)
        dilution_factors = ratios.median(axis=1)
        expected = self.df.div(dilution_factors, axis=0)

        pdt.assert_frame_equal(out, expected)

    def test_pqn_metadata_error(self):
        with self.assertRaisesRegex(
            ValueError, r"Reference label 'qc' not found in metadata column"
        ):
            normalize_pqn(
                self.df,
                method="median",
                ref_samples=self.ref_metadata,
                ref_label="qc",
            )


class TestImputeTable(TestPluginBase):
    package = "q2_mfa"

    def setUp(self):
        super().setUp()
        self.df = pd.DataFrame(
            {
                "F1": [0.0, 1.0, 2.0],
                "F2": [3.0, 0.0, 4.0],
            },
            index=["S1", "S2", "S3"],
        )

    def test_impute_preserves_shape_index_columns(self):
        out = impute_table(self.df, impute="knn", knn_neighbors=2)

        self.assertEqual(out.shape, self.df.shape)
        self.assertFalse((out.to_numpy() == 0.0).any())
        self.assertListEqual(list(out.index), list(self.df.index))
        self.assertListEqual(list(out.columns), list(self.df.columns))

    def test_impute_replaces_zeros(self):
        out = impute_table(self.df, impute="rf", rf_n_estimators=10, rf_random_state=42)

        self.assertEqual(out.shape, self.df.shape)
        self.assertFalse((out.to_numpy() == 0.0).any())
        self.assertListEqual(list(out.index), list(self.df.index))
        self.assertListEqual(list(out.columns), list(self.df.columns))


if __name__ == "__main__":
    unittest.main()
