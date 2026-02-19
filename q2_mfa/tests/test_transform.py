# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from rachis.plugin.testing import TestPluginBase

from q2_mfa.transform import transform_clr


class TestTransformCLR(TestPluginBase):
    package = "q2_mfa.tests"

    def test_transform_clr_data_adaptive_uses_min_nonzero(self):
        df = pd.DataFrame(
            [[0, 1], [2, 3]],
        )

        exp = pd.DataFrame(
            [[-0.346574, 0.346574], [-0.143841, 0.143841]],
        )

        obs = transform_clr(df, pseudocount=None)
        pd.testing.assert_frame_equal(obs, exp)
        assert np.allclose(obs.sum(axis=1).to_numpy(), 0.0)

    def test_transform_clr_fixed_pseudocount(self):
        df = pd.DataFrame(
            [[0, 1], [2, 3]],
        )

        exp = pd.DataFrame(
            [[-0.549306, 0.549306], [-0.168236, 0.168236]],
        )

        obs = transform_clr(df, pseudocount=0.5)
        pd.testing.assert_frame_equal(obs, exp)
        assert np.allclose(obs.sum(axis=1).to_numpy(), 0.0)
