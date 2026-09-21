# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
# ----------------------------------------------------------------------------
from rachis.plugin.testing import TestPluginBase

from q2_mfa.pls import PLSTuneComponents, PLSTuneComponentsDirFmt


class TestPLSTuneComponentsType(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    def test_semantic_type_registration(self):
        self.assertRegisteredSemanticType(PLSTuneComponents)

    def test_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            PLSTuneComponents, PLSTuneComponentsDirFmt
        )
