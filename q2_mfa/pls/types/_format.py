# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import rachis.plugin.model as model
from q2_types.distance_matrix import LSMatFormat
from q2_types.tabular import TableJSONLFileFormat


class PLSComponentSelectionDirFmt(model.DirectoryFormat):
    """Stores DIABLO component-selection diagnostics."""

    error_rate_weighted = model.File(
        "error_rate_weighted.jsonl",
        format=TableJSONLFileFormat,
    )
    error_rate_majority = model.File(
        "error_rate_majority.jsonl",
        format=TableJSONLFileFormat,
    )
    choice_matrix_weighted = model.File(
        "choice_matrix_weighted.jsonl",
        format=TableJSONLFileFormat,
    )
    choice_matrix_majority = model.File(
        "choice_matrix_majority.jsonl",
        format=TableJSONLFileFormat,
    )


AssociationMatrixDirFmt = model.SingleFileDirectoryFormat(
    "AssociationMatrixDirFmt", "association-matrix.tsv", LSMatFormat
)


class PLSFeatureSelectionDirFmt(model.DirectoryFormat):
    """Stores DIABLO feature-selection diagnostics and selected keepX values."""

    error_rate = model.File("error_rate.jsonl", format=TableJSONLFileFormat)
    selected_features = model.File(
        "selected_features.jsonl", format=TableJSONLFileFormat
    )


class PLSFitDirFmt(model.DirectoryFormat):
    """Stores PLS and DIABLO tables with wide scores, loadings, and correlations."""

    loadings = model.File("loadings.jsonl", format=TableJSONLFileFormat)
    scores = model.File("scores.jsonl", format=TableJSONLFileFormat)
    weights = model.File("weights.jsonl", format=TableJSONLFileFormat)
    prop_expl_var = model.File("prop_expl_var.jsonl", format=TableJSONLFileFormat)
    ave = model.File("ave.jsonl", format=TableJSONLFileFormat)
    feature_stability = model.File(
        "feature_stability.jsonl", format=TableJSONLFileFormat
    )
    auc = model.File("auc.jsonl", format=TableJSONLFileFormat)
    error_rate_weighted = model.File(
        "error_rate_weighted.jsonl", format=TableJSONLFileFormat
    )
    error_rate_majority = model.File(
        "error_rate_majority.jsonl", format=TableJSONLFileFormat
    )
    feature_component_correlation = model.File(
        "feature_component_correlation.jsonl", format=TableJSONLFileFormat
    )
    feature_similarity = model.File(
        "feature_similarity.jsonl", format=TableJSONLFileFormat
    )
