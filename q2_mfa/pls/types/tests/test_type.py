# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
# ----------------------------------------------------------------------------
from rachis.plugin.testing import TestPluginBase

from q2_mfa.pls import (
    AssociationMatrix,
    AssociationMatrixDirFmt,
    PLSComponentSelection,
    PLSComponentSelectionDirFmt,
    PLSFit,
    PLSFitDirFmt,
)


class TestPLSTypes(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    def test_tune_components_semantic_type_registration(self):
        self.assertRegisteredSemanticType(PLSComponentSelection)

    def test_tune_components_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            PLSComponentSelection, PLSComponentSelectionDirFmt
        )

    def test_association_matrix_semantic_type_registration(self):
        self.assertRegisteredSemanticType(AssociationMatrix)

    def test_association_matrix_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            AssociationMatrix, AssociationMatrixDirFmt
        )


class TestPLSFitType(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    def test_semantic_type_registration(self):
        self.assertRegisteredSemanticType(PLSFit)

    def test_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(PLSFit, PLSFitDirFmt)
