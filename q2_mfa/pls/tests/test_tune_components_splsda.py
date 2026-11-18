# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
from pathlib import Path

import pandas as pd
from rachis.plugin.testing import TestPluginBase

from q2_mfa.pls.examples import _nutrimouse_table, nutrimouse_metadata_factory
from q2_mfa.pls.tune_components_splsda import tune_components_splsda


class TestTuneComponentsSPLSDA(TestPluginBase):
    package = "q2_mfa.pls.tests"

    @classmethod
    def setUpClass(cls):
        """Loads the lipid predictors and diet labels for real-R testing."""
        super().setUpClass()
        cls.lipids = _nutrimouse_table("lipid").view(pd.DataFrame)
        cls.diet = nutrimouse_metadata_factory().get_column("diet")

    def test_one_component_with_leave_one_out(self):
        """Reports component one and missing repeat SDs for leave-one-out."""
        # The public tuning action requires at least two candidate components.
        data = tune_components_splsda(
            table=self.lipids,
            y=self.diet,
            ncomp=1,
            validation="loo",
            nrepeat=1,
        )
        rates = dict(data.error_rate.iter_views(pd.DataFrame))[Path("error_rate.jsonl")]
        choices = dict(data.choice_matrix.iter_views(pd.DataFrame))[
            Path("choice_matrix.jsonl")
        ]
        self.assertEqual(rates["component"].tolist(), [1] * 6)
        self.assertTrue(rates["sd"].isna().all())
        self.assertEqual(choices.drop(columns="id").to_numpy().tolist(), [[1] * 3] * 2)
