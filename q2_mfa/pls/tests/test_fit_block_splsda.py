# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from unittest.mock import Mock

import pandas as pd
from q2_types.tabular import TableJSONLFileFormat
from rachis import Artifact
from rachis.plugin import CaptureHolder, Collection, Str
from rachis.plugin.testing import TestPluginBase
from rpy2.robjects import r

from q2_mfa.pls import PLSFeatureSelection, PLSFeatureSelectionDirFmt
from q2_mfa.pls.examples import nutrimouse_factory
from q2_mfa.pls.fit_block_splsda import (
    _consensus_variates,
    _feature_component_correlations,
    _feature_similarity_to_dataframe,
    _keepx_from_feature_selection,
    _r_auc_to_dataframe,
    _r_ave_to_dataframe,
    _r_feature_stability_to_dataframe,
    _r_named_frames,
    _r_prop_expl_var_to_dataframe,
    _resolve_fit_keepx,
)
from q2_mfa.pls.select_features_block_splsda import _serialize_feature_selection


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
        cls.prop_expl_var_values = r["source"](
            instance.get_data_path("fit/prop-expl-var-input.R")
        ).rx2("value")
        cls.prop_expl_var_expected = pd.read_csv(
            instance.get_data_path("fit/prop-expl-var-expected.tsv"), sep="\t"
        )
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
        ):
            cls.nutrimouse_matrices[name] = {
                block: table.set_index(identifier).drop(columns=["id", "block"])
                for block, table in cls.nutrimouse_expected[name].groupby(
                    "block", sort=False
                )
            }
        cls.nutrimouse_tables, _ = nutrimouse_factory()
        cls.nutrimouse_weights = (
            cls.nutrimouse_expected["weights"].set_index("block").drop(columns="id")
        )

    def test_r_named_frames(self):
        observed = _r_named_frames(self.named_matrices)
        self.assertEqual(list(observed), list(self.wide_tables))
        for block, expected in self.wide_tables.items():
            with self.subTest(block=block):
                pd.testing.assert_frame_equal(observed[block], expected)

    def test_r_prop_expl_var_to_dataframe(self):
        """Preserves block order and aligns variance values by component name."""
        pd.testing.assert_frame_equal(
            _r_prop_expl_var_to_dataframe(self.prop_expl_var_values),
            self.prop_expl_var_expected,
        )

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

    def test_keepx_from_feature_selection(self):
        self.assertEqual(
            _keepx_from_feature_selection(
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
            correlations,
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
        correlations = self.nutrimouse_expected["feature-component-correlation"].drop(
            columns="id"
        )
        for component_count, key in (
            (1, "feature-similarity-comp1"),
            (2, "feature-similarity"),
        ):
            with self.subTest(components=component_count):
                expected = self.nutrimouse_expected[key]
                selected = set(expected["feature_1"]) | set(expected["feature_2"])
                observed = _feature_similarity_to_dataframe(
                    correlations.loc[correlations["feature"].isin(selected)].iloc[
                        :, : 2 + component_count
                    ]
                )
                pd.testing.assert_frame_equal(
                    observed,
                    expected.drop(columns="id"),
                    check_dtype=False,
                )
