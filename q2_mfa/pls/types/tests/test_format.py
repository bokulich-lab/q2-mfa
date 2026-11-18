# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
# ----------------------------------------------------------------------------
from pathlib import Path

from rachis import Artifact, Metadata
from rachis.plugin.testing import TestPluginBase

from q2_mfa.pls import (
    AssociationMatrixDirFmt,
    PLSComponentSelection,
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
        artifact = Artifact.import_data(
            PLSComponentSelection, str(self.tune_components_fixture_dir)
        )
        data = artifact.view(PLSComponentSelectionDirFmt)
        data.validate()
        self.assertEqual(
            {path.name for path, _ in data.error_rate.iter_views(Metadata)},
            {"error_rate_weighted.jsonl", "error_rate_majority.jsonl"},
        )
        self.assertEqual(
            {path.name for path, _ in data.choice_matrix.iter_views(Metadata)},
            {"choice_matrix_weighted.jsonl", "choice_matrix_majority.jsonl"},
        )

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


class TestSingleTablePLSComponentSelectionDirFmt(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    def test_complete_fixture_validates(self):
        """Validates the error rates and component choices in the new format."""
        artifact = Artifact.import_data(
            PLSComponentSelection,
            self.get_data_path("pls-tune-components-single-table"),
        )
        data = artifact.view(PLSComponentSelectionDirFmt)
        data.validate()
        self.assertEqual(
            {path.name for path, _ in data.error_rate.iter_views(Metadata)},
            {"error_rate.jsonl"},
        )
        self.assertEqual(
            {path.name for path, _ in data.choice_matrix.iter_views(Metadata)},
            {"choice_matrix.jsonl"},
        )

    def test_collection_paths(self):
        """Keeps existing vote filenames and supports unsuffixed filenames."""
        data = PLSComponentSelectionDirFmt()
        for vote in (None, "weighted", "majority"):
            suffix = f"_{vote}" if vote else ""
            with self.subTest(vote=vote):
                self.assertEqual(
                    data.error_rate.path_maker(vote=vote).name,
                    f"error_rate{suffix}.jsonl",
                )
                self.assertEqual(
                    data.choice_matrix.path_maker(vote=vote).name,
                    f"choice_matrix{suffix}.jsonl",
                )
