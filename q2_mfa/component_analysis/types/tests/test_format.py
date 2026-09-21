# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from q2_types.tabular import TableJSONLFileFormat
from q2_types.tabular._deferred_setup._transformers import table_jsonl_to_df
from rachis.plugin.testing import TestPluginBase
from skbio import OrdinationResults

from q2_mfa.component_analysis import ComponentAnalysisDirFmt, ComponentAnalysisResult
from q2_mfa.component_analysis.types._transformer import _TABLE_SPECS, _TableSpec

_FEATURE_TABLE_ATTRS = (
    "feature_coordinates",
    "feature_correlations",
    "feature_contributions",
    "feature_cosine_similarities",
)


class TestComponentAnalysisFormatRegression(TestPluginBase):
    package = "q2_mfa.component_analysis.types.tests"

    @classmethod
    def setUpClass(cls):
        """Loads shared immutable Prince and JSONL fixture paths."""
        helper = cls()
        cls.pca_tables = Path(
            helper.get_data_path("component-analysis/prince-tables/pca")
        )
        cls.mfa_tables = Path(
            helper.get_data_path("component-analysis/prince-tables/mfa")
        )
        cls.pca_jsonl = Path(helper.get_data_path("component-analysis/jsonl/pca"))
        cls.mfa_jsonl = Path(helper.get_data_path("component-analysis/jsonl/mfa"))
        cls.pca_result = _load_component_analysis(cls.pca_tables)
        cls.mfa_result = _load_component_analysis(cls.mfa_tables)

    def test_pca_component_analysis_to_dirfmt_transformer(self):
        """Tests PCA ComponentAnalysisResult serialization against JSONL fixtures."""
        to_fmt = self.get_transformer(ComponentAnalysisResult, ComponentAnalysisDirFmt)

        observed = to_fmt(self.pca_result)
        observed.validate()

        self._assert_jsonl_dirs_equal(observed.path, self.pca_jsonl)

    def test_mfa_component_analysis_to_dirfmt_transformer(self):
        """Tests MFA ComponentAnalysisResult serialization against JSONL fixtures."""
        to_fmt = self.get_transformer(ComponentAnalysisResult, ComponentAnalysisDirFmt)

        observed = to_fmt(self.mfa_result)
        observed.validate()

        self._assert_jsonl_dirs_equal(observed.path, self.mfa_jsonl)

    def test_pca_jsonl_dirfmt_to_component_analysis_transformer(self):
        """Tests PCA JSONL directory-format deserialization."""
        to_result = self.get_transformer(
            ComponentAnalysisDirFmt, ComponentAnalysisResult
        )
        fmt = ComponentAnalysisDirFmt(self.pca_jsonl, mode="r")

        observed = to_result(fmt)

        self._assert_component_analysis_equal(observed, self.pca_result)
        self.assertTrue(observed.is_pca)

    def test_mfa_jsonl_dirfmt_to_component_analysis_transformer(self):
        """Tests MFA JSONL directory-format deserialization."""
        to_result = self.get_transformer(
            ComponentAnalysisDirFmt, ComponentAnalysisResult
        )
        fmt = ComponentAnalysisDirFmt(self.mfa_jsonl, mode="r")

        observed = to_result(fmt)

        self._assert_component_analysis_equal(observed, self.mfa_result)
        self.assertTrue(observed.is_mfa)

    def test_pca_jsonl_dirfmt_validates(self):
        """Tests that the PCA JSONL fixture validates as a directory format."""
        fmt = ComponentAnalysisDirFmt(self.pca_jsonl, mode="r")

        fmt.validate()

    def test_mfa_jsonl_dirfmt_validates(self):
        """Tests that the MFA JSONL fixture validates as a directory format."""
        fmt = ComponentAnalysisDirFmt(self.mfa_jsonl, mode="r")

        fmt.validate()

    def test_component_analysis_to_skbio_ordination(self):
        """Tests ordination conversion from an MFA directory-format fixture."""
        to_ordination = self.get_transformer(ComponentAnalysisDirFmt, OrdinationResults)
        fmt = ComponentAnalysisDirFmt(self.mfa_jsonl, mode="r")

        observed = to_ordination(fmt)

        self.assertEqual(observed.short_method_name, "MFA")
        self.assertEqual(observed.long_method_name, "Multiple Factor Analysis")
        self.assertIsInstance(observed.eigvals, np.ndarray)
        np.testing.assert_allclose(
            observed.eigvals,
            self.mfa_result.eigenvalues,
        )
        self.assertIsInstance(observed.proportion_explained, np.ndarray)
        np.testing.assert_allclose(
            observed.proportion_explained,
            self.mfa_result.percentage_of_variance / 100,
        )
        assert_frame_equal(
            observed.samples,
            self.mfa_result.sample_coordinates,
            check_exact=False,
        )
        assert_frame_equal(
            observed.features,
            self.mfa_result.feature_coordinates,
            check_exact=False,
        )

    def test_pca_component_analysis_to_skbio_ordination_names(self):
        """Tests PCA method names in ordination conversion."""
        to_ordination = self.get_transformer(ComponentAnalysisDirFmt, OrdinationResults)
        fmt = ComponentAnalysisDirFmt(self.pca_jsonl, mode="r")

        observed = to_ordination(fmt)

        self.assertEqual(observed.short_method_name, "PCA")
        self.assertEqual(observed.long_method_name, "Principal Component Analysis")

    def test_component_analysis_to_dirfmt_missing_required_table(self):
        """Tests that missing required tables fail with a clear error."""
        result = _load_component_analysis(self.pca_tables)
        result.sample_cosine_similarities = None
        to_fmt = self.get_transformer(ComponentAnalysisResult, ComponentAnalysisDirFmt)

        with self.assertRaisesRegex(
            ValueError, "Missing required table: sample_cosine_similarities."
        ):
            to_fmt(result)

    def _assert_component_analysis_equal(
        self,
        observed: ComponentAnalysisResult,
        expected: ComponentAnalysisResult,
    ) -> None:
        """
        Asserts equality for every ComponentAnalysisResult output table.

        Args:
            observed (ComponentAnalysisResult): The observed result.
            expected (ComponentAnalysisResult): The expected result.

        Returns:
            None
        """
        for spec in _TABLE_SPECS:
            observed_table = getattr(observed, spec.attr)
            expected_table = getattr(expected, spec.attr)
            if expected_table is None:
                self.assertIsNone(observed_table)
            elif isinstance(expected_table, np.ndarray):
                np.testing.assert_allclose(
                    observed_table,
                    expected_table,
                )
            else:
                assert_frame_equal(
                    observed_table,
                    expected_table,
                    check_exact=False,
                )

    def _assert_jsonl_dirs_equal(self, observed_dir: Path, expected_dir: Path) -> None:
        """
        Asserts that two ComponentAnalysis JSONL directories are equivalent.

        Args:
            observed_dir (Path): The observed JSONL directory.
            expected_dir (Path): The expected JSONL directory.

        Returns:
            None
        """
        expected_names = {path.name for path in expected_dir.glob("*.jsonl")}
        observed_names = {path.name for path in observed_dir.glob("*.jsonl")}
        self.assertEqual(observed_names, expected_names)
        for expected_file in sorted(expected_dir.glob("*.jsonl")):
            observed_file = observed_dir / expected_file.name
            self.assertTrue(observed_file.exists())
            observed = table_jsonl_to_df(
                TableJSONLFileFormat(str(observed_file), mode="r")
            )
            expected = table_jsonl_to_df(
                TableJSONLFileFormat(str(expected_file), mode="r")
            )
            self.assertEqual(
                observed.attrs["description"],
                expected.attrs["description"],
            )
            assert_frame_equal(
                observed.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_exact=False,
            )


def _load_component_analysis(directory: Path) -> ComponentAnalysisResult:
    """
    Loads a ComponentAnalysisResult object from Prince-table fixture TSVs.

    Args:
        directory (Path): The Prince-table fixture directory.

    Returns:
        ComponentAnalysisResult: The loaded component-analysis result.
    """
    kwargs = {}
    for spec in _TABLE_SPECS:
        suffix = ".npy" if spec.kind == "vector" else ".tsv"
        path = directory / f"{spec.attr}{suffix}"
        if not path.exists():
            kwargs[spec.attr] = None
        elif spec.kind == "vector":
            kwargs[spec.attr] = np.load(path)
        elif spec.kind == "wide":
            kwargs[spec.attr] = _read_wide_table(path, spec.index)
        elif spec.kind == "indexed_rows":
            kwargs[spec.attr] = _read_indexed_rows_table(path, spec)
        elif spec.kind == "multi_columns":
            kwargs[spec.attr] = _read_multi_columns_table(path)
        else:
            raise ValueError(f"Unknown table kind: {spec.kind}.")
    return ComponentAnalysisResult(**kwargs)


def _read_wide_table(path: Path, index: str) -> pd.DataFrame:
    """
    Reads a simple wide Prince-table fixture.

    Args:
        path (Path): The wide TSV fixture path.
        index (str): The storage index name from the table spec.

    Returns:
        pd.DataFrame: The loaded wide table.
    """
    table = pd.read_csv(path, sep="\t", index_col=0)
    table.columns = pd.Index(table.columns.astype(int), name="component")
    table.index.name = None if index == "sample_id" else index
    return table


def _read_indexed_rows_table(path: Path, spec: _TableSpec) -> pd.DataFrame:
    """
    Reads an indexed-row table fixture.

    Args:
        path (Path): The indexed-row table TSV fixture path.
        spec (_TableSpec): The table serialization specification.

    Returns:
        pd.DataFrame: The loaded indexed-row table.
    """
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    row_index = list(spec.row_index)
    restored_index = list(spec.restored_index or spec.row_index)
    if set(restored_index).issubset(columns):
        table = pd.read_csv(path, sep="\t", index_col=restored_index)
        table.index = pd.MultiIndex.from_tuples(
            table.index,
            names=restored_index,
        )
    elif set(row_index).issubset(columns):
        table = pd.read_csv(path, sep="\t", index_col=row_index)
        table.index = pd.MultiIndex.from_tuples(
            table.index,
            names=restored_index,
        )
    else:
        table = _read_wide_table(path, spec.index)
        return table
    table.columns = pd.Index(table.columns.astype(int), name="component")
    return table


def _read_multi_columns_table(path: Path) -> pd.DataFrame:
    """
    Reads a Prince partial-row coordinate fixture.

    Args:
        path (Path): The partial-row coordinate TSV path.

    Returns:
        pd.DataFrame: The loaded partial-row coordinate table.
    """
    table = pd.read_csv(path, sep="\t", header=[0, 1], index_col=0)
    table.index.name = None
    table.columns = pd.MultiIndex.from_tuples(
        [(group, int(component)) for group, component in table.columns],
        names=[None, None],
    )
    return table
