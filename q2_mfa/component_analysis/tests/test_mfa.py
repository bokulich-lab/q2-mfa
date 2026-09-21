# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import os
import warnings

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import prince
from rachis import Metadata
from rachis.core.exceptions import RachisWarning
from rachis.plugin.testing import TestPluginBase

from q2_mfa.component_analysis.mfa import (
    _build_prince_input,
    _metadata_to_grouped_tables,
    _parse_metadata_groups,
    _validate_metadata_group_column_types,
    mfa,
)
from q2_mfa.component_analysis.pca import create_result_object
from q2_mfa.component_analysis.types import ComponentAnalysisResult


class TestMFA(TestPluginBase):
    package = "q2_mfa.component_analysis.tests"

    @classmethod
    def setUpClass(cls):
        instance = cls()
        cls.table_a = instance._load_table("mfa/mfa_table_a.tsv")
        cls.table_b = instance._load_table("mfa/mfa_table_b.tsv")
        cls.mismatched = instance._load_table("mfa/mfa_mismatched.tsv")
        cls.disjoint = instance._load_table("mfa/mfa_disjoint.tsv")
        cls.duplicate_a = instance._load_table("mfa/mfa_duplicate_a.tsv")
        cls.duplicate_b = instance._load_table("mfa/mfa_duplicate_b.tsv")
        cls.filter_table_a = instance._load_table("mfa/mfa_filter_table_a.tsv")
        cls.filter_table_b = instance._load_table("mfa/mfa_filter_table_b.tsv")
        cls.filter_drop_group = instance._load_table("mfa/mfa_filter_drop_group.tsv")
        cls.sample_metadata = Metadata.load(
            instance.get_data_path("mfa/mfa_metadata.tsv")
        )
        cls.mixed_type_metadata = Metadata.load(
            instance.get_data_path("mfa/mfa_mixed_type_metadata.tsv")
        )

    def _load_table(self, filename):
        return pd.read_csv(self.get_data_path(filename), sep="\t", index_col=0)

    def test_parse_metadata_groups_default(self):
        observed = _parse_metadata_groups(None, ["age", "bmi"])

        self.assertEqual(observed, {"metadata": ["age", "bmi"]})

    def test_parse_metadata_groups_string_group_name(self):
        observed = _parse_metadata_groups("clinical", ["age", "bmi"])

        self.assertEqual(observed, {"clinical": ["age", "bmi"]})

    def test_parse_metadata_groups_error_empty_string(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            _parse_metadata_groups(" ", ["age", "bmi"])

    def test_parse_metadata_groups_mapping_string_columns(self):
        observed = _parse_metadata_groups(
            {"clinical": "age, bmi", "body": "score"},
            ["age", "bmi", "score"],
        )

        self.assertEqual(observed, {"clinical": ["age", "bmi"], "body": ["score"]})

    def test_parse_metadata_groups_error_empty_columns(self):
        with self.assertRaisesRegex(ValueError, "must contain at least one column"):
            _parse_metadata_groups({"clinical": " , "}, ["age", "bmi"])

    def test_metadata_to_grouped_tables_none(self):
        self.assertEqual(_metadata_to_grouped_tables(None, None), {})

    def test_metadata_to_grouped_tables_error_groups_without_metadata(self):
        with self.assertRaisesRegex(ValueError, "requires sample_metadata"):
            _metadata_to_grouped_tables(None, {"clinical": "age"})

    def test_metadata_to_grouped_tables_selects_groups(self):
        observed = _metadata_to_grouped_tables(
            self.sample_metadata,
            {"clinical": "age,score", "body": "bmi"},
        )

        self.assertEqual(list(observed), ["clinical", "body"])
        pdt.assert_frame_equal(
            observed["clinical"],
            self.sample_metadata.to_dataframe().loc[:, ["age", "score"]],
        )
        pdt.assert_frame_equal(
            observed["body"],
            self.sample_metadata.to_dataframe().loc[:, ["bmi"]],
        )

    def test_validate_metadata_group_column_types_allows_single_type_groups(self):
        observed = _validate_metadata_group_column_types(
            self.mixed_type_metadata,
            {"clinical": ["age", "score"], "treatment": ["group"]},
        )

        self.assertIsNone(observed)

    def test_validate_metadata_group_column_types_error_mixed_types(self):
        with self.assertRaisesRegex(
            ValueError,
            "Metadata group 'clinical' contains multiple column types",
        ):
            _validate_metadata_group_column_types(
                self.mixed_type_metadata,
                {"clinical": ["age", "group"]},
            )

    def test_metadata_to_grouped_tables_error_unknown_column(self):
        with self.assertRaisesRegex(ValueError, "not present in the metadata: nope"):
            _metadata_to_grouped_tables(self.sample_metadata, {"clinical": "nope"})

    def test_metadata_to_grouped_tables_error_duplicate_column(self):
        with self.assertRaisesRegex(
            ValueError, "cannot be assigned to multiple groups: age"
        ):
            _metadata_to_grouped_tables(
                self.sample_metadata,
                {"clinical": "age", "body": "age"},
            )

    def test_build_prince_input_tables(self):
        tables = {"metabolome": self.table_a, "microbiome": self.table_b}
        table = _build_prince_input(tables)

        self.assertEqual(list(table.index), ["sample-1", "sample-2", "sample-3"])
        self.assertIsInstance(table.columns, pd.MultiIndex)
        self.assertEqual(
            list(table.columns),
            [
                ("metabolome", "feature-a1"),
                ("metabolome", "feature-a2"),
                ("microbiome", "feature-b1"),
                ("microbiome", "feature-b2"),
            ],
        )

    def test_build_prince_input_tables_metadata(self):
        tables = {"metabolome": self.table_a}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = _build_prince_input(
                tables,
                self.sample_metadata,
                {"clinical": "age", "body": "bmi"},
            )

        self.assertEqual(list(table.index), ["sample-1", "sample-2", "sample-3"])
        self.assertIsInstance(table.columns, pd.MultiIndex)
        self.assertEqual(
            list(table.columns),
            [
                ("metabolome", "feature-a1"),
                ("metabolome", "feature-a2"),
                ("clinical", "age"),
                ("body", "bmi"),
            ],
        )

    def test_build_prince_input_metadata(self):
        table = _build_prince_input(
            sample_metadata=self.sample_metadata,
            metadata_groups={"clinical": "age", "body": "bmi"},
        )

        self.assertEqual(
            list(table.index),
            ["sample-1", "sample-2", "sample-3", "sample-4"],
        )
        self.assertIsInstance(table.columns, pd.MultiIndex)
        self.assertEqual(
            list(table.columns),
            [("clinical", "age"), ("body", "bmi")],
        )

    def test_build_prince_input_drops_samples_with_warning(self):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            table = _build_prince_input(
                {"metabolome": self.table_a, "other": self.mismatched}
            )

        self.assertEqual(len(observed), 2)
        self.assertTrue(
            all(issubclass(warning.category, RachisWarning) for warning in observed)
        )
        self.assertEqual(
            str(observed[0].message),
            "Dropping samples from group 'metabolome' that are not shared "
            "across all tables:\nsample-2",
        )
        self.assertEqual(
            str(observed[1].message),
            "Dropping samples from group 'other' that are not shared across all "
            "tables:\nsample-x",
        )
        self.assertEqual(list(table.index), ["sample-1", "sample-3"])

    def test_build_prince_input_filters_features_and_drops_empty_group(self):
        tables = {
            "metabolome": self.filter_table_a,
            "microbiome": self.filter_table_b,
            "empty": self.filter_drop_group,
        }

        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            table = _build_prince_input(tables)

        observed_messages = [str(warning.message) for warning in observed]
        self.assertTrue(
            all(issubclass(warning.category, RachisWarning) for warning in observed)
        )
        self.assertIn(
            "Dropped columns with missing values: ('metabolome', 'feature-missing')",
            observed_messages,
        )
        self.assertIn(
            "Dropped columns with zero variance: ('metabolome', 'feature-constant')",
            observed_messages,
        )
        self.assertIn(
            "Dropped columns with missing values: ('empty', 'feature-drop-missing')",
            observed_messages,
        )
        self.assertIn(
            "Dropped columns with zero variance: ('empty', 'feature-drop-constant')",
            observed_messages,
        )
        self.assertIn(
            "Dropped MFA group 'empty' because all features were removed during "
            "missing value filtering or zero-variance filtering.",
            observed_messages,
        )
        self.assertEqual(
            list(table.columns),
            [
                ("metabolome", "feature-a1"),
                ("metabolome", "feature-a2"),
                ("microbiome", "feature-b1"),
                ("microbiome", "feature-b2"),
            ],
        )

    def test_build_prince_input_error_no_groups(self):
        with self.assertRaisesRegex(ValueError, "at least two groups"):
            _build_prince_input()

    def test_build_prince_input_error_one_table_group(self):
        with self.assertRaisesRegex(ValueError, "at least two groups"):
            _build_prince_input({"metabolome": self.table_a})

    def test_build_prince_input_error_whitespace_feature_group_name(self):
        with self.assertRaisesRegex(
            ValueError, "MFA group names cannot be empty strings"
        ):
            _build_prince_input({" ": self.table_a, "other": self.table_b})

    def test_build_prince_input_error_whitespace_metadata_group_name(self):
        with self.assertRaisesRegex(
            ValueError, "MFA group names cannot be empty strings"
        ):
            _build_prince_input(
                {"metabolome": self.table_a},
                self.sample_metadata,
                {" ": "age"},
            )

    def test_build_prince_input_error_no_shared_samples(self):
        with self.assertRaisesRegex(ValueError, "do not share any sample IDs"):
            _build_prince_input({"metabolome": self.table_a, "other": self.disjoint})

    def test_build_prince_input_uses_two_metadata_groups(self):
        table = _build_prince_input(
            sample_metadata=self.sample_metadata,
            metadata_groups={"clinical": "age,score", "body": "bmi"},
        )

        self.assertIn(("clinical", "age"), table.columns)
        self.assertIn(("clinical", "score"), table.columns)
        self.assertIn(("body", "bmi"), table.columns)
        self.assertEqual(
            list(table.index), ["sample-1", "sample-2", "sample-3", "sample-4"]
        )

    def test_build_prince_input_error_string_metadata_group_is_one_group(self):
        with self.assertRaisesRegex(ValueError, "at least two groups"):
            _build_prince_input(
                sample_metadata=self.sample_metadata,
                metadata_groups="clinical",
            )

    def test_build_prince_input_error_unknown_metadata_column(self):
        with self.assertRaisesRegex(ValueError, "not present in the metadata: nope"):
            _build_prince_input(
                {"metabolome": self.table_a},
                self.sample_metadata,
                {"clinical": "nope"},
            )

    def test_build_prince_input_error_duplicate_metadata_column(self):
        with self.assertRaisesRegex(
            ValueError, "cannot be assigned to multiple groups: age"
        ):
            _build_prince_input(
                {"metabolome": self.table_a},
                self.sample_metadata,
                {"clinical": "age", "body": "age"},
            )

    def test_build_prince_input_error_metadata_groups_without_metadata(self):
        with self.assertRaisesRegex(
            ValueError, "metadata_groups requires sample_metadata"
        ):
            _build_prince_input(
                {"metabolome": self.table_a},
                metadata_groups={"clinical": "age"},
            )

    def test_build_prince_input_error_group_duplicates(self):
        with self.assertRaisesRegex(
            ValueError, "cannot duplicate feature table group names: metabolome"
        ):
            _build_prince_input(
                {"metabolome": self.table_a},
                self.sample_metadata,
                {"metabolome": "age"},
            )

    def test_mfa_outputs_match_prince_regression_fixtures(self):
        tables = {"metabolome": self.table_a, "microbiome": self.table_b}
        expected_dir = self.get_data_path("mfa/prince-regression")
        results = mfa(tables=tables, engine="scipy")

        observed_vectors = {
            "eigenvalues": results.eigenvalues,
            "percentage_of_variance": results.percentage_of_variance,
            "cumulative_percentage_of_variance": (
                results.cumulative_percentage_of_variance
            ),
        }
        observed_tables = {
            "sample_coordinates": results.sample_coordinates.to_numpy(),
            "feature_coordinates": results.feature_coordinates.to_numpy(),
            "sample_cosine_similarities": (
                results.sample_cosine_similarities.to_numpy()
            ),
            "sample_contributions": results.sample_contributions.to_numpy(),
            "feature_correlations": results.feature_correlations.to_numpy(),
            "feature_contributions": results.feature_contributions.to_numpy(),
            "feature_cosine_similarities": (
                results.feature_cosine_similarities.to_numpy()
            ),
            "partial_sample_coordinates": (
                results.partial_sample_coordinates.to_numpy()
            ),
            "group_coordinates": results.group_coordinates.to_numpy(),
            "group_contributions": results.group_contributions.to_numpy(),
            "group_cosine_similarities": (results.group_cosine_similarities.to_numpy()),
            "partial_correlations": results.partial_correlations.to_numpy(),
            "partial_contributions": results.partial_contributions.to_numpy(),
        }

        self.assertIsInstance(results, ComponentAnalysisResult)
        self.assertTrue(results.is_mfa)
        for output_name, observed in observed_vectors.items():
            expected = np.loadtxt(os.path.join(expected_dir, f"{output_name}.tsv"))
            npt.assert_allclose(observed, expected)
        for output_name, observed in observed_tables.items():
            expected = pd.read_csv(
                os.path.join(expected_dir, f"{output_name}.tsv"),
                sep="\t",
                header=None,
            ).to_numpy()
            npt.assert_allclose(observed, expected)

    def test_create_component_analysis_object_from_mfa(self):
        tables = {"metabolome": self.table_a, "microbiome": self.table_b}
        table = _build_prince_input(tables)
        prince_result = prince.MFA(engine="scipy").fit(table)

        observed = create_result_object(prince_result, table)

        self.assertIsInstance(observed, ComponentAnalysisResult)
        self.assertTrue(observed.is_mfa)

    def test_mfa_filters_missing_and_zero_variance_features(self):
        tables = {
            "metabolome": self.filter_table_a,
            "microbiome": self.filter_table_b,
        }

        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            results = mfa(tables=tables, engine="scipy")

        observed_messages = [str(warning.message) for warning in observed]
        self.assertIn(
            "Dropped columns with missing values: ('metabolome', 'feature-missing')",
            observed_messages,
        )
        self.assertIn(
            "Dropped columns with zero variance: ('metabolome', 'feature-constant')",
            observed_messages,
        )

        self.assertEqual(
            list(results.feature_coordinates.index),
            [
                ("metabolome", "feature-a1"),
                ("metabolome", "feature-a2"),
                ("microbiome", "feature-b1"),
                ("microbiome", "feature-b2"),
            ],
        )
