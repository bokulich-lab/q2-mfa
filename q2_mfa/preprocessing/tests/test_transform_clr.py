# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from rachis.plugin.testing import TestPluginBase
from skbio.stats.composition import clr, multi_replace

from q2_mfa.preprocessing import transform_clr


class TestTransformCLR(TestPluginBase):
    package = "q2_mfa.preprocessing.tests"

    def setUp(self):
        super().setUp()

        self.df = pd.DataFrame(
            [[0, 1, 2], [3, 4, 5]],
            index=["sample-1", "sample-2"],
            columns=["feature-1", "feature-2", "feature-3"],
        )

    def test_transform_clr_default_pseudocount_is_one(self):
        df_adjusted = self.df + 1
        log_df = np.log(df_adjusted)
        exp = log_df.sub(log_df.mean(axis=1), axis=0)

        obs = transform_clr(
            self.df,
            replacement_method="pseudocount",
        )
        pd.testing.assert_frame_equal(obs, exp)
        self.assertEqual(obs.shape, self.df.shape)
        pd.testing.assert_index_equal(obs.index, self.df.index)
        pd.testing.assert_index_equal(obs.columns, self.df.columns)

    def test_transform_clr_multiplicative_fixed_delta(self):
        replaced = multi_replace(self.df, delta=0.5)
        exp = pd.DataFrame(
            clr(replaced),
            index=self.df.index,
            columns=self.df.columns,
        )

        obs = transform_clr(
            self.df,
            delta=0.5,
        )
        pd.testing.assert_frame_equal(obs, exp)
        self.assertEqual(obs.shape, self.df.shape)
        pd.testing.assert_index_equal(obs.index, self.df.index)
        pd.testing.assert_index_equal(obs.columns, self.df.columns)

    def test_multiplicative_all_zero_sample_raises(self):
        df = pd.DataFrame(
            [[0, 0, 0], [2, 3, 4]],
            index=["sample-1", "sample-2"],
            columns=["feature-1", "feature-2", "feature-3"],
        )

        with self.assertRaisesRegex(ValueError, "all 0"):
            transform_clr(df)

    def test_transform_clr_sparse_data(self):
        # Create sparse random data with 90% zeros
        np.random.seed(42)
        data = np.random.randint(0, 1001, size=(100, 50))
        zero_mask = np.random.choice([True, False], size=data.shape, p=[0.9, 0.1])
        zero_mask[:, 0] = False
        data[zero_mask] = 0
        df_sparse = pd.DataFrame(
            data,
            index=[f"sample-{i}" for i in range(100)],
            columns=[f"feature-{j}" for j in range(50)],
        )

        obs = transform_clr(df_sparse)

        self.assertEqual(obs.shape, df_sparse.shape)
        pd.testing.assert_index_equal(obs.index, df_sparse.index)
        pd.testing.assert_index_equal(obs.columns, df_sparse.columns)
