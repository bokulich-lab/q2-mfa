# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
from pandas.testing import assert_frame_equal
from q2_types.feature_table import FeatureTable, Unconstrained
from rachis import Artifact, Metadata, ResultCollection
from rachis.plugin.testing import TestPluginBase
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import r
from rpy2.robjects.packages import importr

from q2_mfa.pls import PLSComponentSelection, PLSComponentSelectionDirFmt
from q2_mfa.pls.descriptions import report_descriptions
from q2_mfa.pls.tune_components_block_splsda import (
    _error_rate_metadata,
    _print_component_choice,
    _r_choice_matrix_to_df,
    _tune_components_block_splsda,
    _tune_components_block_visualisation,
    mixomics,
)


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
                dtype={"id": str},
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
        cls.expected_weighted_error_rates = read_table(
            "nutrimouse-weighted-error-rates.tsv"
        )
        cls.expected_weighted_error_rates["component"] = (
            cls.expected_weighted_error_rates["component"].astype(float)
        )

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

    def test_error_rate_metadata_filters_and_labels_overall_error_rates(self):
        actual = _error_rate_metadata(
            Metadata(self.expected_weighted_error_rates)
        ).to_dataframe()

        self.assertEqual(
            actual.index.tolist(),
            [
                "11",
                "12",
                "13",
                "14",
                "25",
                "26",
                "27",
                "28",
                "39",
                "40",
                "41",
                "42",
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
        """Preserves component choice labels and integer values."""
        perf_result = r["source"](
            self.get_data_path("tune-components/perf-statistics.R")
        ).rx2("value")

        actual = _r_choice_matrix_to_df(perf_result, "WeightedVote")

        expected = pd.DataFrame(
            {"max.dist": [2, 1], "centroids.dist": [4, 3]},
            index=pd.Index(["Overall.ER", "Overall.BER"], name="id"),
        ).astype("int64")
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


class TestComponentTuningVisualisation(TestPluginBase):
    package = "q2_mfa.pls.types.tests"

    @classmethod
    def setUpClass(cls):
        """Imports immutable artifacts for both component-tuning layouts."""
        super().setUpClass()
        helper = cls()
        cls.tuning_artifacts = {
            "single": Artifact.import_data(
                PLSComponentSelection,
                helper.get_data_path("pls-tune-components-single-table"),
            ),
            "block": Artifact.import_data(
                PLSComponentSelection, helper.get_data_path("pls-tune-components")
            ),
        }

    def test_visualisation_reads_single_table_and_block_artifacts(self):
        """Passes real artifact tables to the appropriate downstream actions."""
        layouts = {
            "single": [("", "sPLS-DA error rates")],
            "block": [
                ("_weighted", "sPLS-DA weighted-vote error rates"),
                ("_majority", "sPLS-DA majority-vote error rates"),
            ],
        }
        for layout, variants in layouts.items():
            with self.subTest(layout=layout):
                artifact = self.tuning_artifacts[layout]
                data = artifact.view(PLSComponentSelectionDirFmt)
                expected_rates = dict(data.error_rate.iter_views(pd.DataFrame))
                expected_rate_metadata = dict(data.error_rate.iter_views(Metadata))
                expected_choices = dict(data.choice_matrix.iter_views(Metadata))
                lineplot = MagicMock(return_value=(MagicMock(),))
                tabulate = MagicMock(return_value=(MagicMock(),))
                context = MagicMock()
                context.get_action.side_effect = [lineplot, tabulate]

                _tune_components_block_visualisation(context, artifact)

                self.assertEqual(lineplot.call_count, len(variants))
                self.assertEqual(tabulate.call_count, 2 * len(variants))
                for index, (suffix, title) in enumerate(variants):
                    raw = expected_rates[Path(f"error_rate{suffix}.jsonl")]
                    expected = raw.set_index("id")
                    expected = expected.loc[
                        expected["class"].isin({"Overall.ER", "Overall.BER"})
                    ]
                    actual = lineplot.call_args_list[index].kwargs
                    actual_rates = actual["metadata"].to_dataframe()
                    self.assertEqual(
                        actual_rates.index.tolist(), expected.index.tolist()
                    )
                    self.assertEqual(
                        actual_rates["error_rate"].tolist(),
                        (expected["distance"] + ": " + expected["class"]).tolist(),
                    )
                    self.assertEqual(
                        actual_rates["mean"].tolist(), expected["mean"].tolist()
                    )
                    self.assertEqual(actual["title"], title)
                    self.assertEqual(actual["x_measure"], "component")
                    self.assertEqual(actual["y_measure"], "mean")
                    rate_table = (
                        tabulate.call_args_list[2 * index]
                        .kwargs["input"]
                        .to_dataframe()
                    )
                    assert_frame_equal(
                        rate_table,
                        expected_rate_metadata[
                            Path(f"error_rate{suffix}.jsonl")
                        ].to_dataframe(),
                    )
                    choices = (
                        tabulate.call_args_list[2 * index + 1]
                        .kwargs["input"]
                        .to_dataframe()
                    )
                    expected = expected_choices[
                        Path(f"choice_matrix{suffix}.jsonl")
                    ].to_dataframe()
                    assert_frame_equal(choices, expected)
                expected_report_count = 2 if layout == "single" else 5
                self.assertEqual(context.make_report.call_count, expected_report_count)
                for index, (suffix, _) in enumerate(variants):
                    report_call = context.make_report.call_args_list[index]
                    self.assertEqual(list(report_call.args[1]), ["Plot", "Table"])
                    description = report_descriptions[f"error_rate{suffix}"]
                    self.assertEqual(
                        report_call.kwargs["descriptions"],
                        {"Plot": description, "Table": description},
                    )
                sections = context.make_report.call_args.args[1]
                self.assertEqual(list(sections), ["Error rates", "Component choices"])
                if layout == "single":
                    descriptions = context.make_report.call_args.kwargs["descriptions"]
                    for description in descriptions.values():
                        self.assertNotIn("vote", description.lower())
