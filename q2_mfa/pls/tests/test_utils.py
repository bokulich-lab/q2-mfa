# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import warnings
from unittest.mock import Mock, patch

import pandas as pd
from rachis import Metadata
from rachis.core.exceptions import RachisWarning
from rachis.plugin import CaptureHolder, Int
from rachis.plugin.testing import TestPluginBase
from rpy2.robjects import FloatVector, ListVector, StrVector

from q2_mfa.pls import PLSComponentSelectionDirFmt, utils


class TestPLSUtils(TestPluginBase):
    package = "q2_mfa.pls.tests"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        instance = cls()

        def read_table(name):
            return pd.read_csv(
                instance.get_data_path(f"utils/{name}"), sep="\t", index_col="id"
            )

        cls.blocks = {
            "block-a": read_table("block-a.tsv"),
            "block-b": read_table("block-b.tsv"),
        }
        cls.response = Metadata(read_table("response.tsv")).get_column("group")
        cls.design = read_table("design-valid.tsv")
        cls.tuned_ncomp = PLSComponentSelectionDirFmt(
            instance.get_data_path("utils/tuned-ncomp"), mode="r"
        )
        cls.tuned_ncomp_missing_selection = PLSComponentSelectionDirFmt(
            instance.get_data_path("utils/tuned-ncomp-missing-selection"), mode="r"
        )

    def test_align_samples_drops_unshared_and_unlabelled_samples(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tables, target = utils._align_samples(
                self.blocks, self.response.to_series()
            )

        self.assertEqual([warning.category for warning in caught], [RachisWarning] * 3)
        self.assertEqual(
            [str(warning.message) for warning in caught],
            [
                "Dropping samples from block 'block-a' that are not shared "
                "across all blocks:\ns1, s6",
                "Dropping samples from block 'block-b' that are not shared "
                "across all blocks:\ns4, s6",
                "Dropping samples from block 'y' that are not shared across all "
                "blocks:\ns5",
            ],
        )
        self.assertEqual(list(tables), ["block-a", "block-b"])
        for table in tables.values():
            self.assertEqual(table.index.tolist(), ["s2", "s3"])
        self.assertEqual(target.index.tolist(), ["s2", "s3"])

    def test_align_samples_rejects_fewer_than_two_tables(self):
        with self.assertRaisesRegex(
            ValueError, "At least two named feature tables are required."
        ):
            utils._align_samples(
                {"block-a": self.blocks["block-a"]}, self.response.to_series()
            )

    def test_align_samples_rejects_inputs_without_shared_samples(self):
        tables = {
            "block-a": self.blocks["block-a"].rename(
                index=lambda sample: f"a-{sample}"
            ),
            "block-b": self.blocks["block-b"].rename(
                index=lambda sample: f"b-{sample}"
            ),
        }
        with self.assertRaisesRegex(
            ValueError,
            "The input feature tables and response share no sample IDs.",
        ):
            utils._align_samples(tables, self.response.to_series())

    def test_resolve_design_rejects_missing_design_options(self):
        with self.assertRaisesRegex(
            ValueError, "Provide exactly one of 'design-matrix' or 'design-weight'."
        ):
            utils._resolve_design(None, None, list(self.blocks))

    def test_resolve_design_rejects_both_design_options(self):
        with self.assertRaisesRegex(
            ValueError, "Provide exactly one of 'design-matrix' or 'design-weight'."
        ):
            utils._resolve_design(self.design, 0.1, list(self.blocks))

    def test_resolve_design_returns_scalar_weight(self):
        self.assertEqual(utils._resolve_design(None, 0.1, list(self.blocks)), 0.1)

    def test_resolve_design_rejects_mismatched_block_names(self):
        invalid = self.design.rename(index={"block-a": "other"})
        with self.assertRaisesRegex(
            ValueError,
            "The row and column names in 'design-matrix' must exactly match ",
        ):
            utils._resolve_design(invalid, None, list(self.blocks))

    def test_resolve_design_reorders_valid_matrix_to_block_order(self):
        observed = utils._resolve_design(self.design, None, list(self.blocks))
        expected = pd.DataFrame(
            [[0.0, 0.2], [0.2, 0.0]],
            index=pd.Index(["block-a", "block-b"], name="id"),
            columns=["block-a", "block-b"],
        )
        pd.testing.assert_frame_equal(observed, expected)

    def test_build_bpparam_uses_serial_backend_for_one_thread(self):
        with patch.object(utils, "biocparallel") as biocparallel:
            utils._build_bpparam(1)
        biocparallel.SerialParam.assert_called_once_with()
        biocparallel.MulticoreParam.assert_not_called()

    def test_build_bpparam_uses_multicore_workers(self):
        with patch.object(utils, "biocparallel") as biocparallel:
            utils._build_bpparam(3)
        biocparallel.MulticoreParam.assert_called_once_with(workers=3)
        biocparallel.SerialParam.assert_not_called()

    def test_build_bpparam_delegates_worker_count_for_zero_threads(self):
        with patch.object(utils, "biocparallel") as biocparallel:
            utils._build_bpparam(0)
        biocparallel.MulticoreParam.assert_called_once_with()
        biocparallel.SerialParam.assert_not_called()

    def test_to_r_inputs_preserves_block_and_sample_labels(self):
        sample_ids = ["s2", "s3"]
        blocks = {name: table.loc[sample_ids] for name, table in self.blocks.items()}
        target = self.response.to_series().loc[sample_ids]

        r_blocks, r_target, r_design, r_keep_x, r_already_tested_x = utils._to_r_inputs(
            blocks,
            target,
            self.design,
            {"block-a": [1, 2], "block-b": [3, 4]},
            {"block-a": [1], "block-b": [3]},
        )

        self.assertEqual(list(r_blocks.names), ["block-a", "block-b"])
        self.assertEqual(list(utils.base.rownames(r_blocks.rx2("block-a"))), sample_ids)
        self.assertEqual(list(r_target.names), sample_ids)
        self.assertEqual(list(utils.base.as_character(r_target)), ["case", "control"])
        self.assertEqual(list(utils.base.rownames(r_design)), ["block-b", "block-a"])
        self.assertEqual(list(utils.base.colnames(r_design)), ["block-b", "block-a"])
        self.assertEqual(list(r_keep_x.names), ["block-a", "block-b"])
        self.assertEqual(list(r_keep_x.rx2("block-a")), [1, 2])
        self.assertEqual(list(r_keep_x.rx2("block-b")), [3, 4])
        self.assertEqual(list(r_already_tested_x.names), ["block-a", "block-b"])
        self.assertEqual(list(r_already_tested_x.rx2("block-a")), [1])
        self.assertEqual(list(r_already_tested_x.rx2("block-b")), [3])

    def test_vote_error_rates_combines_mean_and_standard_deviation(self):
        dimnames = utils.base.list(
            StrVector(["Overall.BER", "Overall.ER"]),
            StrVector(["comp1", "comp2"]),
        )
        mean_rates = utils.base.matrix(
            FloatVector([0.1, 0.3, 0.2, 0.4]), nrow=2, dimnames=dimnames
        )
        standard_deviations = utils.base.matrix(
            FloatVector([0.01, 0.03, 0.02, 0.04]), nrow=2, dimnames=dimnames
        )
        perf_result = ListVector(
            {
                "WeightedVote.error.rate": ListVector({"max.dist": mean_rates}),
                "WeightedVote.error.rate.sd": ListVector(
                    {"max.dist": standard_deviations}
                ),
            }
        )
        observed = utils._r_perf_error_rate_to_df(perf_result, "WeightedVote")

        expected = pd.DataFrame(
            {
                "distance": ["max.dist"] * 4,
                "class": ["Overall.BER", "Overall.BER", "Overall.ER", "Overall.ER"],
                "component": [1, 2, 1, 2],
                "mean": [0.1, 0.2, 0.3, 0.4],
                "sd": [0.01, 0.02, 0.03, 0.04],
            }
        )
        expected["component"] = expected["component"].astype("Int64")
        pd.testing.assert_frame_equal(observed, expected)

    def test_ncomp_from_tuned_ncomp_rejects_missing_selected_value(self):
        """Rejects a tuning choice matrix with no selected component value."""

        with self.assertRaisesRegex(
            ValueError,
            r"^The component-tuning artifact has no selected number of components "
            r"for the requested vote, measure, and distance\.$",
        ):
            utils._ncomp_from_tuned_ncomp(
                self.tuned_ncomp_missing_selection,
                weighted=True,
                measure="BER",
                distance="max.dist",
            )

    def test_ncomp_from_tuned_ncomp_selects_requested_vote(self):
        """Reads the selected component from the requested vote matrix."""
        for weighted, expected in ((True, 2), (False, 3)):
            with self.subTest(weighted=weighted):
                observed = utils._ncomp_from_tuned_ncomp(
                    self.tuned_ncomp,
                    weighted=weighted,
                    measure="BER",
                    distance="max.dist",
                )
                self.assertEqual(observed, expected)

    def test_resolve_ncomp_returns_manual_value(self):
        """Uses a manually provided component count without reading tuning data."""
        ncomp = CaptureHolder("ncomp", 4, Int, Mock(parameters={}))

        observed = utils._resolve_ncomp(
            ncomp,
            tuned_ncomp=None,
            weighted=True,
            measure="BER",
            distance="max.dist",
        )

        self.assertEqual(observed, 4)

    def test_resolve_ncomp_captures_value_from_tuned_ncomp(self):
        """Derives and records the component count when no manual value exists."""
        provenance = Mock(parameters={})
        ncomp = CaptureHolder("ncomp", None, Int, provenance)

        observed = utils._resolve_ncomp(
            ncomp,
            tuned_ncomp=self.tuned_ncomp,
            weighted=True,
            measure="BER",
            distance="max.dist",
        )

        self.assertEqual(observed, 2)
        self.assertTrue(ncomp.is_set)
        self.assertEqual(provenance.parameters, {"ncomp": 2})

    def test_resolve_ncomp_rejects_both_sources(self):
        """Rejects a manual count paired with a component-tuning artifact."""
        ncomp = CaptureHolder("ncomp", 4, Int, Mock(parameters={}))

        with self.assertRaisesRegex(
            ValueError,
            r"^Provide exactly one of ncomp or a tuned-ncomp artifact\.$",
        ):
            utils._resolve_ncomp(
                ncomp,
                tuned_ncomp=self.tuned_ncomp,
                weighted=True,
                measure="BER",
                distance="max.dist",
            )

    def test_resolve_ncomp_rejects_neither_source(self):
        """Rejects an unset manual count without a component-tuning artifact."""
        ncomp = CaptureHolder("ncomp", None, Int, Mock(parameters={}))

        with self.assertRaisesRegex(
            ValueError,
            r"^Provide exactly one of ncomp or a tuned-ncomp artifact\.$",
        ):
            utils._resolve_ncomp(
                ncomp,
                tuned_ncomp=None,
                weighted=True,
                measure="BER",
                distance="max.dist",
            )

    def test_to_r_inputs_without_keepx_returns_optional_nulls(self):
        """Keeps the five-result interface when no feature grids are supplied."""
        sample_ids = ["s2", "s3"]
        blocks = {name: table.loc[sample_ids] for name, table in self.blocks.items()}
        inputs = utils._to_r_inputs(
            blocks, self.response.to_series().loc[sample_ids], 0.1
        )
        self.assertEqual(len(inputs), 5)
        self.assertEqual(inputs[2], 0.1)
        self.assertIsNone(inputs[3])
        self.assertIsNone(inputs[4])
