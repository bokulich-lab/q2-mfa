# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
# ----------------------------------------------------------------------------
from pathlib import Path

from rachis.plugin.testing import TestPluginBase

from q2_mfa.pls import PLSTuneComponentsDirFmt


class TestPLSTuneComponentsDirFmt(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    @classmethod
    def setUpClass(cls):
        helper = cls()
        cls.fixture_dir = Path(helper.get_data_path("pls-tune-components/complete"))

    def test_complete_fixture_validates(self):
        PLSTuneComponentsDirFmt(self.fixture_dir, mode="r").validate()
