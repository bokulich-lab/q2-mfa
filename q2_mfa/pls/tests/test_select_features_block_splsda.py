# -----------------------------------------------------------------------------
# Copyright (c) 2026, Rachis Developers.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# -----------------------------------------------------------------------------

import json
from unittest.mock import MagicMock, call, patch

import pandas as pd
from pandas.testing import assert_frame_equal
from q2_types.feature_table import FeatureTable, Unconstrained
from q2_types.tabular import TableJSONLFileFormat
from q2templates.reports import matryoshka_template
from rachis import Artifact, Metadata, ResultCollection
from rachis.plugin.testing import TestPluginBase
from rpy2.robjects import conversion, default_converter, pandas2ri, r
from rpy2.robjects.conversion import localconverter

from q2_mfa.pls import AssociationMatrix, PLSFeatureSelectionDirFmt
from q2_mfa.pls.descriptions import (
    feature_selection_descriptions,
    feature_selection_report_descriptions,
)
from q2_mfa.pls.select_features_block_splsda import (
    _parse_keepx_collection,
    _r_feature_selection_error_rate_to_dataframe,
    _select_features_block_splsda_visualiser,
    _selected_features_table,
    _serialize_feature_selection,
)
from q2_mfa.pls.utils import base


class TestSelectFeaturesBlockSPLSDA(TestPluginBase):
    package = "q2_mfa.pls.tests"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        instance = cls()

        def read_table(name):
            return pd.read_csv(
                instance.get_data_path(f"tune-components/{name}"),
                sep="\t",
                index_col="id",
            )

        cls.tables = ResultCollection(
            {
                name: Artifact.import_data(
                    FeatureTable[Unconstrained],
                    read_table(f"{name}.tsv"),
                )
                for name in ("block-a", "block-b")
            }
        )
        cls.response = Metadata(read_table("response.tsv")).get_column("group")
        cls.design_matrix = Artifact.import_data(
            AssociationMatrix,
            pd.read_csv(
                instance.get_data_path("utils/design-valid.tsv"),
                sep="\t",
                index_col="id",
            ),
        )

    def test_parse_keepx_collection_requires_expected_blocks(self):
        """Rejects keepX grids whose block names differ from the input blocks."""
        with self.assertRaisesRegex(ValueError, "must exactly match input block names"):
            _parse_keepx_collection({"block-a": "1,2"}, ("block-a", "block-b"), ncomp=2)

    def test_parse_keepx_collection_parses_named_candidate_grids(self):
        """Converts each named comma-separated candidate grid to integers."""
        observed = _parse_keepx_collection(
            {
                "block-a": "1, 2, 3",
                "block-b": "4,5,6",
                "block-c": "7,8,9",
            },
            ("block-a", "block-b", "block-c"),
            ncomp=3,
        )

        self.assertEqual(
            observed,
            {
                "block-a": [1, 2, 3],
                "block-b": [4, 5, 6],
                "block-c": [7, 8, 9],
            },
        )

    def test_parse_keepx_collection_requires_positive_integers(self):
        """Rejects non-positive and non-integer keepX candidate values."""
        for values in ("0,2", "one,2"):
            with self.subTest(values=values):
                with self.assertRaisesRegex(
                    ValueError,
                    "comma-separated list of positive integers",
                ):
                    _parse_keepx_collection(
                        {"block-a": values, "block-b": "1,2"},
                        ("block-a", "block-b"),
                        ncomp=2,
                    )

    def test_parse_keepx_collection_requires_equal_prior_selection_lengths(self):
        """Rejects prior selections with unequal numbers of components."""
        with self.assertRaisesRegex(
            ValueError,
            "^All 'already-tested-x' lists must have the same length\\. "
            "Each entry specifies how many features to retain for one component\\.$",
        ):
            _parse_keepx_collection(
                {"block-a": "1,2", "block-b": "3"},
                ("block-a", "block-b"),
                ncomp=3,
                already_tested=True,
            )

    def test_parse_keepx_collection_requires_remaining_component_to_tune(self):
        """Rejects prior selections that fill every requested component."""
        with self.assertRaisesRegex(
            ValueError,
            "^'ncomp' must be greater than the length of each 'already-tested-x' "
            "list so that at least one component remains to tune\\.$",
        ):
            _parse_keepx_collection(
                {"block-a": "1,2", "block-b": "3,4"},
                ("block-a", "block-b"),
                ncomp=2,
                already_tested=True,
            )

    @patch("q2_mfa.pls.select_features_block_splsda._call_without_messages")
    def test_private_feature_selection_calls_mixomics_and_serializes(
        self, call_without_messages
    ):
        """Passes converted inputs to mixOmics and serializes its result."""
        tune_result = r(
            "list("
            "error.rate = structure(matrix(c(0.2, 0.3, 0.1, 0.25), nrow = 4, "
            "dimnames = list(c('1_1', '2_1', '1_2', '2_2'), 'comp1'))), "
            "error.rate.sd = structure(matrix(c(0.01, 0.02, 0.03, 0.04), "
            "nrow = 4, dimnames = list(c('1_1', '2_1', '1_2', '2_2'), "
            "'comp1'))), choice.keepX = list(`block-a` = 1, `block-b` = 2))"
        )
        call_without_messages.return_value = tune_result
        select_features = self.plugin.methods["_select_features_block_splsda"]

        (observed,) = select_features(
            tables=self.tables,
            y=self.response,
            keepx={"block-a": "1,2", "block-b": "1,2"},
            design_weight=0.1,
            ncomp=1,
            seed=1,
            threads=1,
        )

        from q2_mfa.pls.select_features_block_splsda import mixomics

        call_without_messages.assert_called_once()
        tune_function, r_blocks, _ = call_without_messages.call_args.args
        self.assertIs(tune_function, mixomics.tune_block_splsda)
        with localconverter(default_converter + pandas2ri.converter):
            for name, table in self.tables.items():
                converted = conversion.get_conversion().rpy2py(
                    base.as_data_frame(r_blocks.rx2(name))
                )
                assert_frame_equal(converted, table.view(pd.DataFrame))
        tune_parameters = call_without_messages.call_args.kwargs
        self.assertEqual(tune_parameters["ncomp"], 1)
        self.assertEqual(tune_parameters["seed"], 1)
        self.assertEqual(
            [list(values) for values in tune_parameters["test.keepX"]],
            [[1, 2], [1, 2]],
        )
        observed.view(PLSFeatureSelectionDirFmt).validate()

    @patch("q2_mfa.pls.select_features_block_splsda.Metadata")
    def test_feature_selection_visualiser_creates_component_heatmaps(self, metadata):
        """Calls the heatmap action once per component for two blocks."""
        error_rates = MagicMock()
        selected_features = MagicMock()
        error_rate_data = MagicMock()
        error_rate_data.columns = [
            "keepx_block-a",
            "keepx_block-b",
            "component",
            "mean",
            "sd",
        ]
        components = MagicMock()
        components.unique.return_value = [1, 2]
        error_rate_data.__getitem__.return_value = components
        error_rates.to_dataframe.return_value = error_rate_data
        feature_selection = MagicMock()
        feature_selection.view.return_value.error_rate.view.return_value = error_rates
        feature_selection.view.return_value.selected_features.view.return_value = (
            selected_features
        )
        error_rate_table = MagicMock()
        selected_features_table = MagicMock()
        tabulate = MagicMock(
            side_effect=[(error_rate_table,), (selected_features_table,)]
        )
        component_heatmaps = [MagicMock(), MagicMock()]
        component_metadata = [MagicMock(), MagicMock()]
        metadata.side_effect = component_metadata
        heatmap = MagicMock(side_effect=[(plot,) for plot in component_heatmaps])
        heatmap_report = MagicMock()
        combined_report = MagicMock()
        context = MagicMock()
        context.get_action.side_effect = [tabulate, heatmap]
        context.make_report.side_effect = [heatmap_report, combined_report]

        observed = _select_features_block_splsda_visualiser(
            ctx=context,
            feature_selection=feature_selection,
        )

        self.assertEqual(
            context.get_action.call_args_list,
            [call("metadata", "tabulate"), call("vizard", "heatmap")],
        )
        self.assertEqual(
            tabulate.call_args_list,
            [call(input=error_rates), call(input=selected_features)],
        )
        self.assertEqual(
            heatmap.call_args_list,
            [
                call(
                    metadata=component_metadata[0],
                    x_measure="keepx_block-a",
                    y_measure="keepx_block-b",
                    gradient_measure="mean",
                    title="block sPLS-DA (DIABLO) keepX error rates: component 1",
                ),
                call(
                    metadata=component_metadata[1],
                    x_measure="keepx_block-a",
                    y_measure="keepx_block-b",
                    gradient_measure="mean",
                    title="block sPLS-DA (DIABLO) keepX error rates: component 2",
                ),
            ],
        )
        self.assertEqual(context.make_report.call_count, 2)
        self.assertEqual(
            context.make_report.call_args_list[0],
            call(
                matryoshka_template,
                {
                    "Component 1": component_heatmaps[0],
                    "Component 2": component_heatmaps[1],
                },
                descriptions={
                    "Component 1": feature_selection_report_descriptions["heatmaps"],
                    "Component 2": feature_selection_report_descriptions["heatmaps"],
                },
            ),
        )
        self.assertIs(observed, combined_report)

    def test_feature_selection_visualiser_returns_tables_for_three_blocks(self):
        """Calls only the tabulate action when there are more than two blocks."""
        error_rates = MagicMock()
        selected_features = MagicMock()
        error_rate_data = MagicMock()
        error_rate_data.columns = [
            "keepx_block-a",
            "keepx_block-b",
            "keepx_block-c",
            "component",
            "mean",
            "sd",
        ]
        error_rates.to_dataframe.return_value = error_rate_data
        feature_selection = MagicMock()
        feature_selection.view.return_value.error_rate.view.return_value = error_rates
        feature_selection.view.return_value.selected_features.view.return_value = (
            selected_features
        )
        error_rate_table = MagicMock()
        selected_features_table = MagicMock()
        tabulate = MagicMock(
            side_effect=[(error_rate_table,), (selected_features_table,)]
        )
        report = MagicMock()
        context = MagicMock()
        context.get_action.return_value = tabulate
        context.make_report.return_value = report

        observed = _select_features_block_splsda_visualiser(
            ctx=context,
            feature_selection=feature_selection,
        )

        self.assertEqual(
            context.get_action.call_args_list, [call("metadata", "tabulate")]
        )
        self.assertEqual(
            tabulate.call_args_list,
            [call(input=error_rates), call(input=selected_features)],
        )
        context.make_report.assert_called_once_with(
            matryoshka_template,
            {
                "Error rates": error_rate_table,
                "Selected features": selected_features_table,
            },
            descriptions={
                "Error rates": feature_selection_report_descriptions["error_rates"],
                "Selected features": feature_selection_report_descriptions[
                    "selected_features"
                ],
            },
        )
        self.assertIs(observed, report)

    def test_error_rate_table_converts_mixomics_statistics(self):
        """Reshapes multiple components and aligns reordered statistics by label."""
        tune_result = r["source"](
            self.get_data_path("feature-selection/error-rate-statistics.R")
        ).rx2("value")

        observed = _r_feature_selection_error_rate_to_dataframe(
            tune_result, ("block-a", "block-b")
        )
        expected = (
            pd.read_csv(
                self.get_data_path("feature-selection/serializer-error-rate.tsv"),
                sep="\t",
            )
            .sort_values(["keepx_block-a", "keepx_block-b"], kind="stable")
            .reset_index(drop=True)
            .astype({"component": "Int64"})
        )

        assert_frame_equal(observed, expected)

    def test_selected_features_table_converts_mixomics_keepx(self):
        """Converts mixOmics selected keepX values to a feature table."""
        tune_result = r(
            "list(choice.keepX = list(" "metabolome = c(5, 7), microbiome = c(3, 4)))"
        )

        observed = _selected_features_table(tune_result)
        expected = pd.DataFrame(
            {"metabolome": [5, 7], "microbiome": [3, 4]},
            index=pd.Index([1, 2], dtype="Int64", name="component"),
        )

        assert_frame_equal(observed, expected)

    def test_serializer_writes_described_jsonl_files(self):
        """Writes both feature-selection outputs with their JSONL descriptions."""
        error_rates = pd.read_csv(
            self.get_data_path("feature-selection/serializer-error-rate.tsv"),
            sep="\t",
        )
        selected_features = pd.read_csv(
            self.get_data_path("feature-selection/serializer-selected-features.tsv"),
            sep="\t",
            index_col="component",
        )
        expected_error_rates = TableJSONLFileFormat(
            self.get_data_path(
                "feature-selection/serializer-expected-error-rate.jsonl"
            ),
            mode="r",
        ).view(pd.DataFrame)
        expected_selected_features = TableJSONLFileFormat(
            self.get_data_path(
                "feature-selection/serializer-expected-selected-features.jsonl"
            ),
            mode="r",
        ).view(pd.DataFrame)

        output = _serialize_feature_selection(error_rates, selected_features)
        error_rate_format = output.error_rate
        selected_features_format = output.selected_features

        for serialized, expected, description in (
            (
                error_rate_format,
                expected_error_rates,
                feature_selection_descriptions["error_rate"],
            ),
            (
                selected_features_format,
                expected_selected_features,
                feature_selection_descriptions["selected_features"],
            ),
        ):
            with serialized.path_maker().open() as fh:
                header = json.loads(next(fh))
            self.assertEqual(header["description"], description)
            assert_frame_equal(
                serialized.view(pd.DataFrame), expected, check_dtype=False
            )

    def test_feature_selection_preserves_prior_keepx_with_association_matrix(self):
        """Runs real R tuning with an association matrix and prior selections."""
        (selection,) = self.plugin.methods["_select_features_block_splsda"](
            tables=self.tables,
            y=self.response,
            design_matrix=self.design_matrix,
            ncomp=2,
            keepx={"block-a": "1,2", "block-b": "1,2"},
            already_tested_x={"block-a": "1", "block-b": "2"},
            folds=3,
            nrepeat=3,
            seed=1,
            threads=1,
        )
        selected = (
            selection.view(PLSFeatureSelectionDirFmt)
            .selected_features.view(Metadata)
            .to_dataframe()
            .set_index("component")
        )
        self.assertEqual(selected.index.tolist(), [1, 2])
        self.assertEqual(selected.loc[1].tolist(), [1, 2])
        for count in selected.loc[2]:
            self.assertIn(count, [1, 2])
