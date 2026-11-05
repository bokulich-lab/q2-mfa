# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, call, patch

import pandas as pd
from pandas.testing import assert_frame_equal
from q2_types.feature_table import FeatureTable, Unconstrained
from rachis import Artifact, Metadata, ResultCollection
from rachis.plugin.testing import TestPluginBase
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import IntVector, ListVector, StrVector, r
from rpy2.robjects.packages import importr

from q2_mfa.pls import AssociationMatrix, PLSComponentSelectionDirFmt
from q2_mfa.pls.descriptions import jsonl_descriptions
from q2_mfa.pls.tune_components_block_splsda import (
    _error_rate_metadata,
    _print_component_choice,
    _r_choice_matrix_to_df,
    _serialize_tune_components,
    _tune_components_block_splsda,
    _tune_components_block_visualisation,
    mixomics,
    tune_components_block_splsda,
)

base = importr("base")


class TestTuneComponentsBlockSPLSDA(TestPluginBase):
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
                "block-a": Artifact.import_data(
                    FeatureTable[Unconstrained], read_table("block-a.tsv")
                ),
                "block-b": Artifact.import_data(
                    FeatureTable[Unconstrained], read_table("block-b.tsv")
                ),
            }
        )
        cls.response = Metadata(read_table("response.tsv")).get_column("group")
        cls.expected_weighted_error_rates = read_table("weighted-error-rates.tsv")
        cls.expected_weighted_error_rates["component"] = (
            cls.expected_weighted_error_rates["component"].astype(float)
        )
        cls.expected_majority_error_rates = read_table("majority-error-rates.tsv")
        cls.expected_majority_error_rates["component"] = (
            cls.expected_majority_error_rates["component"].astype(float)
        )
        cls.expected_weighted_choices = read_table("weighted-choices.tsv").astype(float)
        cls.expected_majority_choices = read_table("majority-choices.tsv").astype(float)

    def test_parameters_reach_mixomics(self):
        """Checks non-default arguments after R translation and S3 dispatch."""
        for name in ("block.plsda", "perf.sgccda"):
            self.addCleanup(r, f'untrace("{name}", where=asNamespace("mixOmics"))')
        r("""
            trace("block.plsda", where=asNamespace("mixOmics"), print=FALSE,
                  tracer=quote(stopifnot(
                ncomp == 3, !scale, tol == 1e-4, max.iter == 200,
                near.zero.var, design == 0.4
            )))
            trace("perf.sgccda", where=asNamespace("mixOmics"), print=FALSE,
                  tracer=quote({
                stopifnot(
                    inherits(object, "sgccda"), dist == "all",
                    validation == "loo", folds == 4, nrepeat == 4,
                    signif.threshold == 0.2, seed == 17,
                    BiocParallel::bpworkers(BPPARAM) == 2,
                    length(list(...)) == 0
                )
                stop("mixOmics parameters verified")
            }))
        """)
        # Refresh rpy2's cached handle to the traced R function.
        with (
            patch.object(mixomics, "block_plsda", importr("mixOmics").block_plsda),
            self.assertRaisesRegex(RRuntimeError, "mixOmics parameters verified"),
        ):
            _tune_components_block_splsda(
                tables={
                    name: table.view(pd.DataFrame)
                    for name, table in self.tables.items()
                },
                y=self.response,
                design_weight=0.4,
                ncomp=3,
                scale=False,
                tol=1e-4,
                max_iter=200,
                near_zero_var=True,
                validation="loo",
                folds=4,
                nrepeat=4,
                signif_threshold=0.2,
                seed=17,
                threads=2,
            )

    def test_tune_components_pipeline_matches_expected_values(self):
        tune_components = self.plugin.pipelines["tune_components_block_splsda"]
        design_matrix = Artifact.import_data(
            AssociationMatrix,
            pd.DataFrame(
                [[0.0, 0.1], [0.1, 0.0]],
                index=pd.Index(["block-b", "block-a"], name="id"),
                columns=["block-b", "block-a"],
            ),
        )

        results = tune_components(
            tables=self.tables,
            y=self.response,
            design_matrix=design_matrix,
            ncomp=2,
            validation="Mfold",
            folds=3,
            nrepeat=3,
            seed=1,
            threads=1,
        )

        tuning = results.tune_components
        tuning_data = tuning.view(PLSComponentSelectionDirFmt)
        actual_tables = (
            (
                tuning_data.error_rate_weighted.view(Metadata).to_dataframe(),
                self.expected_weighted_error_rates,
                True,
            ),
            (
                tuning_data.error_rate_majority.view(Metadata).to_dataframe(),
                self.expected_majority_error_rates,
                True,
            ),
            (
                tuning_data.choice_matrix_weighted.view(Metadata).to_dataframe(),
                self.expected_weighted_choices,
                False,
            ),
            (
                tuning_data.choice_matrix_majority.view(Metadata).to_dataframe(),
                self.expected_majority_choices,
                False,
            ),
        )

        for actual, expected, has_error_rates in actual_tables:
            if has_error_rates:
                assert_frame_equal(actual, expected, rtol=1e-8, atol=1e-10)
            else:
                assert_frame_equal(actual, expected, check_exact=True)

        with (
            results.visualization._archiver.data_dir / "subfigures" / "index.json"
        ).open() as fh:
            report_index = json.load(fh)
        self.assertEqual(list(report_index), ["Error rates", "Component choices"])
        self.assertEqual(
            list(report_index["Error rates"]["children"]),
            ["Weighted vote", "Majority vote"],
        )
        self.assertEqual(
            list(report_index["Component choices"]["children"]),
            ["Weighted vote", "Majority vote"],
        )

    def test_tune_components_pipeline_calls_its_actions(self):
        tuning = MagicMock()
        tune_components = MagicMock(return_value=(tuning,))
        visualisation = MagicMock(return_value=(MagicMock(),))
        mock_context = MagicMock()
        mock_context.get_action.side_effect = [
            tune_components,
            visualisation,
        ]

        tune_components_block_splsda(
            ctx=mock_context,
            tables=self.tables,
            y=self.response,
            design_weight=0.1,
        )

        self.assertEqual(
            mock_context.get_action.call_args_list,
            [
                call("mfa", "_tune_components_block_splsda"),
                call("mfa", "_tune_components_block_visualisation"),
            ],
        )
        tune_components.assert_called_once_with(
            tables=self.tables,
            y=self.response,
            design_matrix=None,
            design_weight=0.1,
            ncomp=2,
            scale=True,
            tol=1e-6,
            max_iter=100,
            near_zero_var=False,
            validation="Mfold",
            folds=10,
            nrepeat=3,
            signif_threshold=0.01,
            seed=None,
            threads=1,
        )
        visualisation.assert_called_once_with(tune_components=tuning)

    @patch("q2_mfa.pls.tune_components_block_splsda._error_rate_metadata")
    def test_tune_components_visualisation_pipeline_calls_its_actions(
        self, mock_error_rate_metadata
    ):
        tuning = MagicMock()
        weighted_error_rates = MagicMock()
        majority_error_rates = MagicMock()
        mock_error_rate_metadata.side_effect = [
            weighted_error_rates,
            majority_error_rates,
        ]
        lineplot = MagicMock(return_value=(MagicMock(),))
        tabulate = MagicMock(return_value=(MagicMock(),))
        mock_context = MagicMock()
        mock_context.get_action.side_effect = [lineplot, tabulate]

        _tune_components_block_visualisation(ctx=mock_context, tune_components=tuning)

        self.assertEqual(
            mock_context.get_action.call_args_list,
            [call("vizard", "lineplot"), call("metadata", "tabulate")],
        )
        self.assertEqual(mock_error_rate_metadata.call_count, 2)
        lineplot.assert_has_calls(
            [
                call(
                    metadata=weighted_error_rates,
                    x_measure="component",
                    y_measure="mean",
                    replicate_method="none",
                    group_by="error_rate",
                    color_palette="category20",
                    title="sPLS-DA weighted-vote error rates",
                ),
                call(
                    metadata=majority_error_rates,
                    x_measure="component",
                    y_measure="mean",
                    replicate_method="none",
                    group_by="error_rate",
                    color_palette="category20",
                    title="sPLS-DA majority-vote error rates",
                ),
            ]
        )
        tabulate.assert_has_calls(
            [
                call(
                    input=tuning.view.return_value.choice_matrix_weighted.view(Metadata)
                ),
                call(
                    input=tuning.view.return_value.choice_matrix_majority.view(Metadata)
                ),
            ]
        )
        self.assertEqual(mock_context.make_report.call_count, 3)

    def test_error_rate_metadata_filters_and_labels_overall_error_rates(self):
        actual = _error_rate_metadata(
            Metadata(self.expected_weighted_error_rates)
        ).to_dataframe()

        self.assertEqual(
            actual.index.tolist(),
            [
                "_5",
                "_6",
                "_7",
                "_8",
                "_13",
                "_14",
                "_15",
                "_16",
                "_21",
                "_22",
                "_23",
                "_24",
            ],
        )
        self.assertEqual(
            actual["error_rate"].tolist(),
            [
                "max.dist: Overall.ER",
                "max.dist: Overall.ER",
                "max.dist: Overall.BER",
                "max.dist: Overall.BER",
                "centroids.dist: Overall.ER",
                "centroids.dist: Overall.ER",
                "centroids.dist: Overall.BER",
                "centroids.dist: Overall.BER",
                "mahalanobis.dist: Overall.ER",
                "mahalanobis.dist: Overall.ER",
                "mahalanobis.dist: Overall.BER",
                "mahalanobis.dist: Overall.BER",
            ],
        )

    def test_r_choice_matrix_to_df_converts_r_matrix_labels_and_values(self):
        choice_matrix = base.matrix(
            IntVector([2, 1, 4, 3]),
            nrow=2,
            dimnames=base.list(
                StrVector(["Overall.ER", "Overall.BER"]),
                StrVector(["max.dist", "centroids.dist"]),
            ),
        )
        perf_result = ListVector(
            {"choice.ncomp": ListVector({"WeightedVote": choice_matrix})}
        )

        actual = _r_choice_matrix_to_df(perf_result, "WeightedVote")

        expected = pd.DataFrame(
            {"max.dist": [2, 1], "centroids.dist": [4, 3]},
            index=pd.Index(["Overall.ER", "Overall.BER"], name="id"),
        ).astype("Int64")
        assert_frame_equal(actual, expected)

    @patch("builtins.print")
    def test_print_component_choice_prints_labelled_table(self, mock_print):
        choice_table = pd.DataFrame(
            {"max.dist": [1]}, index=pd.Index(["Overall.BER"], name="id")
        )

        _print_component_choice(choice_table, "WeightedVote")

        self.assertEqual(
            mock_print.call_args_list,
            [
                call("WeightedVote component-choice matrix:\n", flush=True),
                call(f"{choice_table.to_string()}\n", flush=True),
            ],
        )

    def test_serialize_tune_components_writes_jsonl(self):
        directory_format = _serialize_tune_components(
            self.expected_weighted_error_rates.reset_index(drop=True),
            self.expected_majority_error_rates.reset_index(drop=True),
            self.expected_weighted_choices,
            self.expected_majority_choices,
        )

        directory_format.validate()
        serialized_tables = (
            (
                directory_format.error_rate_weighted,
                self.expected_weighted_error_rates.reset_index(),
                jsonl_descriptions["error_rate_weighted"],
            ),
            (
                directory_format.error_rate_majority,
                self.expected_majority_error_rates.reset_index(),
                jsonl_descriptions["error_rate_majority"],
            ),
            (
                directory_format.choice_matrix_weighted,
                self.expected_weighted_choices.reset_index(),
                jsonl_descriptions["choice_matrix_weighted"],
            ),
            (
                directory_format.choice_matrix_majority,
                self.expected_majority_choices.reset_index(),
                jsonl_descriptions["choice_matrix_majority"],
            ),
        )

        for serialized, expected, description in serialized_tables:
            with serialized.path_maker().open() as fh:
                header = json.loads(next(fh))
            self.assertEqual(header["index"], [])
            self.assertEqual(header["description"], description)
            assert_frame_equal(
                serialized.view(pd.DataFrame), expected, check_dtype=False
            )
