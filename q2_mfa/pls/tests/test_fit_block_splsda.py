# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, Mock

import pandas as pd
from q2_types.tabular import TableJSONLFileFormat
from rachis import Artifact
from rachis.plugin import CaptureHolder, Collection, Str
from rachis.plugin.testing import TestPluginBase
from rpy2.robjects import r

from q2_mfa.pls import PLSFeatureSelection, PLSFeatureSelectionDirFmt, PLSFitDirFmt
from q2_mfa.pls.examples import nutrimouse_metadata_factory, nutrimouse_tables_factory
from q2_mfa.pls.fit_block_splsda import (
    _block_keepx_from_feature_selection,
    _block_matrices_to_dataframe,
    _consensus_variates,
    _feature_component_correlations,
    _feature_similarity_to_dataframe,
    _fit_block_splsda_visualiser,
    _r_auc_to_dataframe,
    _r_ave_to_dataframe,
    _r_feature_stability_to_dataframe,
    _r_named_frames,
    _resolve_fit_keepx,
    fit_block_splsda,
)
from q2_mfa.pls.select_features_block_splsda import _serialize_feature_selection
from q2_mfa.plugin_setup import plugin


class TestFitBlockSPLSDAPipeline(TestPluginBase):
    package = "q2_mfa.pls.tests"

    @classmethod
    def setUpClass(cls):
        """Fits the bundled nutrimouse dataset through the real R implementation."""
        super().setUpClass()
        cls.tables = nutrimouse_tables_factory()
        cls.response = nutrimouse_metadata_factory().get_column("diet")
        (cls.fitted,) = plugin.methods["_fit_block_splsda"](
            tables=cls.tables,
            y=cls.response,
            design_weight=0.9,
            ncomp=2,
            keepx={"gene": "5,5", "lipid": "9,5"},
            folds=2,
            nrepeat=3,
            seed=42,
        )

    def test_fit_pipeline_calls_actions(self):
        """Forwards artifacts and parameters through the pipeline context."""
        fit = MagicMock(return_value=(self.fitted,))
        visualise = MagicMock(return_value=(Mock(),))
        ctx = MagicMock()
        ctx.get_action.side_effect = [fit, visualise]
        fit_block_splsda(
            ctx,
            tables=self.tables,
            y=self.response,
            design_weight=0.9,
            ncomp=2,
            keepx={"gene": "5,5", "lipid": "9,5"},
            seed=42,
        )
        self.assertIs(fit.call_args.kwargs["tables"], self.tables)
        self.assertIs(fit.call_args.kwargs["y"], self.response)
        self.assertEqual(fit.call_args.kwargs["keepx"], {"gene": "5,5", "lipid": "9,5"})
        self.assertEqual(fit.call_args.kwargs["seed"], 42)
        visualise.assert_called_once_with(pls_results=self.fitted, y=self.response)

    def test_visualiser_passes_fitted_coordinates(self):
        """Checks real fitted coordinates and target labels sent to vizard."""
        ctx = MagicMock()
        scatterplot = MagicMock(return_value=(Mock(),))
        correlation_plot = MagicMock(return_value=(Mock(),))
        tabulate = MagicMock(return_value=(Mock(),))
        ctx.get_action.side_effect = [scatterplot, correlation_plot, tabulate]
        _fit_block_splsda_visualiser(ctx, self.fitted, self.response)
        fitted = self.fitted.view(PLSFitDirFmt)
        scores = fitted.scores.view(pd.DataFrame).set_index("sample")
        self.assertEqual(scatterplot.call_count, 4)
        for call, block in zip(
            scatterplot.call_args_list,
            ("gene", "lipid", "average", "weighted_average"),
        ):
            expected = scores.loc[scores["block"] == block, ["comp1", "comp2"]]
            expected["Target"] = self.response.to_series().reindex(expected.index)
            pd.testing.assert_frame_equal(
                call.kwargs["metadata"].to_dataframe(),
                expected.rename_axis("sample-id"),
                check_index_type=False,
            )
            self.assertEqual(call.kwargs["x_measure"], "comp1")
            self.assertEqual(call.kwargs["y_measure"], "comp2")
            self.assertEqual(call.kwargs["color_by"], "Target")
        expected = fitted.feature_component_correlation.view(pd.DataFrame)
        observed = correlation_plot.call_args.kwargs["metadata"].to_dataframe()
        pd.testing.assert_frame_equal(
            observed,
            expected.set_index("id"),
            check_dtype=False,
            check_index_type=False,
        )
        self.assertEqual(correlation_plot.call_args.kwargs["x_measure"], "comp1")
        self.assertEqual(correlation_plot.call_args.kwargs["y_measure"], "comp2")
        self.assertEqual(correlation_plot.call_args.kwargs["color_by"], "block")
        tabulate.assert_not_called()

    def test_visualiser_single_component_uses_tables(self):
        """Tabulates scores when the fit has one component."""
        (artifact,) = self.plugin.methods["_fit_block_splsda"](
            tables=self.tables,
            y=self.response,
            design_weight=0.9,
            ncomp=1,
            keepx={"gene": "5", "lipid": "9"},
            folds=2,
            nrepeat=3,
            seed=42,
        )
        ctx = MagicMock()
        scatterplot = MagicMock()
        correlation_plot = MagicMock()
        tabulate = MagicMock(return_value=(Mock(),))
        ctx.get_action.side_effect = [scatterplot, correlation_plot, tabulate]
        _fit_block_splsda_visualiser(ctx, artifact, self.response)
        self.assertEqual(tabulate.call_count, 5)
        for call in tabulate.call_args_list:
            self.assertIn("comp1", call.kwargs["input"].columns)
            self.assertNotIn("comp2", call.kwargs["input"].columns)
        scatterplot.assert_not_called()
        correlation_plot.assert_not_called()

    def test_visualiser_renders_report(self):
        """Renders the real vizard scatterplots into a nested report."""
        (report,) = self.plugin.pipelines["_fit_block_splsda_visualiser"](
            pls_results=self.fitted, y=self.response
        )
        with (report._archiver.data_dir / "subfigures" / "index.json").open() as fh:
            index = json.load(fh)
        self.assertEqual(list(index), ["Sample scores", "Feature correlations"])
        self.assertEqual(len(index["Sample scores"]["children"]), 4)
        self.assertTrue((report._archiver.data_dir / "index.html").is_file())


class TestFitBlockSPLSDA(TestPluginBase):
    package = "q2_mfa.pls.tests"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        instance = cls()
        cls.named_matrices = r["source"](
            instance.get_data_path("fit/named-frames-input.R")
        ).rx2("value")
        cls.wide_tables = {
            block: pd.read_csv(
                instance.get_data_path(f"fit/named-frames-{block}.tsv"),
                sep="\t",
                index_col=0,
            ).rename_axis(None)
            for block in ("block_b", "block_a")
        }
        cls.ave_values = r["source"](instance.get_data_path("fit/ave-input.R")).rx2(
            "value"
        )
        cls.ave_expected = pd.read_csv(
            instance.get_data_path("fit/ave-expected.tsv"), sep="\t"
        )
        cls.auc_values = r["source"](instance.get_data_path("fit/auc-input.R")).rx2(
            "value"
        )
        cls.auc_expected = pd.read_csv(
            instance.get_data_path("fit/auc-expected.tsv"), sep="\t"
        )
        cls.feature_stability_values = r["source"](
            instance.get_data_path("fit/feature-stability-input.R")
        ).rx2("value")
        cls.feature_stability_expected = pd.read_csv(
            instance.get_data_path("fit/feature-stability-expected.tsv"), sep="\t"
        )
        error_rates = (
            TableJSONLFileFormat(
                instance.get_data_path(
                    "feature-selection/serializer-expected-error-rate.jsonl"
                ),
                mode="r",
            )
            .view(pd.DataFrame)
            .drop(columns="id")
        )
        selected = (
            TableJSONLFileFormat(
                instance.get_data_path(
                    "feature-selection/serializer-expected-selected-features.jsonl"
                ),
                mode="r",
            )
            .view(pd.DataFrame)
            .drop(columns="id")
            .set_index("component")
        )
        cls.selection = Artifact.import_data(
            PLSFeatureSelection, _serialize_feature_selection(error_rates, selected)
        )
        cls.nutrimouse_expected = {
            name: pd.read_csv(
                instance.get_data_path(f"fit/nutrimouse/{name}.tsv"),
                sep="\t",
                comment="#",
                dtype={"id": str, "sample": str},
            )
            for name in (
                "loadings",
                "scores",
                "weights",
                "feature-component-correlation",
                "feature-similarity",
                "feature-similarity-comp1",
            )
        }
        cls.nutrimouse_matrices = {}
        for name, identifier in (
            ("loadings", "feature"),
            ("scores", "sample"),
            ("feature-component-correlation", "feature"),
        ):
            cls.nutrimouse_matrices[name] = {
                block: table.set_index(identifier).drop(columns=["id", "block"])
                for block, table in cls.nutrimouse_expected[name].groupby(
                    "block", sort=False
                )
            }
        cls.nutrimouse_tables = nutrimouse_tables_factory()
        cls.nutrimouse_weights = (
            cls.nutrimouse_expected["weights"].set_index("block").drop(columns="id")
        )

    def test_r_named_frames(self):
        observed = _r_named_frames(self.named_matrices)
        self.assertEqual(list(observed), list(self.wide_tables))
        for block, expected in self.wide_tables.items():
            with self.subTest(block=block):
                pd.testing.assert_frame_equal(observed[block], expected)

    def test_r_ave_to_dataframe(self):
        pd.testing.assert_frame_equal(
            _r_ave_to_dataframe(self.ave_values), self.ave_expected
        )

    def test_r_auc_to_dataframe(self):
        pd.testing.assert_frame_equal(
            _r_auc_to_dataframe(self.auc_values), self.auc_expected
        )

    def test_r_feature_stability_to_dataframe(self):
        pd.testing.assert_frame_equal(
            _r_feature_stability_to_dataframe(self.feature_stability_values),
            self.feature_stability_expected,
        )

    def test_block_keepx_from_feature_selection(self):
        self.assertEqual(
            _block_keepx_from_feature_selection(
                self.selection.view(PLSFeatureSelectionDirFmt), 2
            ),
            {"block-a": "1,2", "block-b": "2,1"},
        )

    def test_resolve_fit_keepx(self):
        manual = {"block-a": "1, 2", "block-b": "2,1"}
        observed = _resolve_fit_keepx(manual, None, ["block-b", "block-a"], 2)
        self.assertEqual(
            list(observed.items()),
            [("block-b", [2, 1]), ("block-a", [1, 2])],
        )

    def test_resolve_fit_keepx_from_artifact(self):
        observed = _resolve_fit_keepx(
            None,
            self.selection.view(PLSFeatureSelectionDirFmt),
            ["block-a", "block-b"],
            2,
        )
        self.assertEqual(observed, {"block-a": [1, 2], "block-b": [2, 1]})

    def test_resolve_fit_keepx_requires_exactly_one_source(self):
        manual = {"block-a": "1,2", "block-b": "2,1"}
        selection = self.selection.view(PLSFeatureSelectionDirFmt)
        for captured in (False, True):
            for values, artifact in ((None, None), (manual, selection)):
                with self.subTest(captured=captured, manual=values is not None):
                    keepx = (
                        CaptureHolder(
                            "keepx", values, Collection[Str], Mock(parameters={})
                        )
                        if captured
                        else values
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        r"^Provide exactly one of 'keepx' or 'feature-selection'\.$",
                    ):
                        _resolve_fit_keepx(keepx, artifact, ["block-a", "block-b"], 2)

    def test_resolve_fit_keepx_requires_one_count_per_component(self):
        for counts in ("1", "1,2,3"):
            for other_counts in (counts, "1,2"):
                with self.subTest(counts=counts, other_counts=other_counts):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"^Each keepx block must provide exactly 2 feature counts, "
                        r"one for each fitted component\.$",
                    ):
                        _resolve_fit_keepx(
                            {"block-a": counts, "block-b": other_counts},
                            None,
                            ["block-a", "block-b"],
                            2,
                        )

    def test_resolve_fit_keepx_validates_block_names(self):
        with self.assertRaisesRegex(
            ValueError, r"^keepx keys must exactly match input block names\.$"
        ):
            _resolve_fit_keepx({"block-a": "1,2"}, None, ["block-a", "block-b"], 2)

    def test_selected_counts_follow_requested_block_order(self):
        observed = _resolve_fit_keepx(
            None,
            self.selection.view(PLSFeatureSelectionDirFmt),
            ["block-b", "block-a"],
            2,
        )
        self.assertEqual(
            list(observed.items()), [("block-b", [2, 1]), ("block-a", [1, 2])]
        )

    def test_selection_requires_matching_components(self):
        with self.assertRaisesRegex(
            ValueError,
            "The feature-selection artifact must contain one row per fitted "
            "component.",
        ):
            _resolve_fit_keepx(
                None,
                self.selection.view(PLSFeatureSelectionDirFmt),
                ["block-a", "block-b"],
                3,
            )

    def test_feature_component_correlations_match_plot_var(self):
        correlations = _feature_component_correlations(
            {
                block: table.view(pd.DataFrame)
                for block, table in self.nutrimouse_tables.items()
            },
            self.nutrimouse_matrices["scores"],
            self.nutrimouse_matrices["loadings"],
        )
        pd.testing.assert_frame_equal(
            _block_matrices_to_dataframe(correlations, identifier="feature"),
            self.nutrimouse_expected["feature-component-correlation"].drop(
                columns="id"
            ),
            check_dtype=False,
        )

    def test_consensus_variates_match_plot_indiv(self):
        expected = self.nutrimouse_matrices["scores"]
        variates = {
            block: scores
            for block, scores in expected.items()
            if block not in ("average", "weighted_average")
        }
        variates["lipid"] = variates["lipid"].iloc[::-1, ::-1]
        consensus = _consensus_variates(variates, self.nutrimouse_weights.iloc[:, ::-1])
        self.assertEqual(list(consensus), ["average", "weighted_average"])
        for method, scores in consensus.items():
            with self.subTest(method=method):
                pd.testing.assert_frame_equal(
                    scores.sort_index(),
                    expected[method].sort_index(),
                    check_dtype=False,
                )

    def test_feature_similarity_matches_circos_plot(self):
        correlations = self.nutrimouse_matrices["feature-component-correlation"]
        for component_count, key in (
            (1, "feature-similarity-comp1"),
            (2, "feature-similarity"),
        ):
            with self.subTest(components=component_count):
                expected = self.nutrimouse_expected[key]
                selected = set(expected["feature_1"]) | set(expected["feature_2"])
                observed = _feature_similarity_to_dataframe(
                    {
                        block: table.loc[
                            table.index.isin(selected),
                            table.columns[:component_count],
                        ]
                        for block, table in correlations.items()
                    }
                )
                pd.testing.assert_frame_equal(
                    observed,
                    expected.drop(columns="id"),
                    check_dtype=False,
                )
