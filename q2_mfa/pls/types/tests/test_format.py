# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
# ----------------------------------------------------------------------------
from pathlib import Path

from rachis.plugin.testing import TestPluginBase

from q2_mfa.pls import (
    AssociationMatrixDirFmt,
    PLSComponentSelectionDirFmt,
    PLSFeatureSelectionDirFmt,
)


class TestPLSFormats(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    @classmethod
    def setUpClass(cls):
        helper = cls()
        cls.tune_components_fixture_dir = Path(
            helper.get_data_path("pls-tune-components")
        )
        cls.association_matrix_fixture_dir = Path(
            helper.get_data_path("association-matrix")
        )

    def test_tune_components_complete_fixture_validates(self):
        PLSComponentSelectionDirFmt(
            self.tune_components_fixture_dir, mode="r"
        ).validate()

    def test_association_matrix_complete_fixture_validates(self):
        AssociationMatrixDirFmt(
            self.association_matrix_fixture_dir, mode="r"
        ).validate()


class TestPLSFeatureSelectionDirFmt(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    @classmethod
    def setUpClass(cls):
        helper = cls()
        cls.fixture_dir = Path(helper.get_data_path("pls-feature-selection/complete"))

    def test_complete_fixture_validates(self):
        PLSFeatureSelectionDirFmt(self.fixture_dir, mode="r").validate()
