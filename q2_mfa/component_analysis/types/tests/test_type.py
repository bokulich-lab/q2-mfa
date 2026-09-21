# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from rachis.plugin.testing import TestPluginBase

from q2_mfa.component_analysis import ComponentAnalysis, ComponentAnalysisDirFmt


class TestMFATypes(TestPluginBase):
    package = "q2_mfa.component_analysis.types.tests"

    def test_component_analysis_semantic_type_registration(self):
        self.assertRegisteredSemanticType(ComponentAnalysis)

    def test_component_analysis_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            ComponentAnalysis, ComponentAnalysisDirFmt
        )
