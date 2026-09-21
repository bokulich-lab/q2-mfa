# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import rachis.plugin.model as model
from q2_types.tabular import TableJSONLFileFormat


class ComponentAnalysisDirFmt(model.DirectoryFormat):
    eigenvalues = model.File("eigenvalues.jsonl", format=TableJSONLFileFormat)
    percentage_of_variance = model.File(
        "percentage_of_variance.jsonl", format=TableJSONLFileFormat
    )
    cumulative_percentage_of_variance = model.File(
        "cumulative_percentage_of_variance.jsonl", format=TableJSONLFileFormat
    )
    sample_coordinates = model.File(
        "sample_coordinates.jsonl", format=TableJSONLFileFormat
    )
    sample_cosine_similarities = model.File(
        "sample_cosine_similarities.jsonl", format=TableJSONLFileFormat
    )
    sample_contributions = model.File(
        "sample_contributions.jsonl", format=TableJSONLFileFormat
    )
    feature_coordinates = model.File(
        "feature_coordinates.jsonl", format=TableJSONLFileFormat
    )
    feature_correlations = model.File(
        "feature_correlations.jsonl", format=TableJSONLFileFormat
    )
    feature_contributions = model.File(
        "feature_contributions.jsonl", format=TableJSONLFileFormat
    )
    feature_cosine_similarities = model.File(
        "feature_cosine_similarities.jsonl", format=TableJSONLFileFormat
    )
    group_coordinates = model.File(
        "group_coordinates.jsonl", format=TableJSONLFileFormat, optional=True
    )
    group_contributions = model.File(
        "group_contributions.jsonl", format=TableJSONLFileFormat, optional=True
    )
    group_cosine_similarities = model.File(
        "group_cosine_similarities.jsonl",
        format=TableJSONLFileFormat,
        optional=True,
    )
    partial_sample_coordinates = model.File(
        "partial_sample_coordinates.jsonl",
        format=TableJSONLFileFormat,
        optional=True,
    )
    partial_correlations = model.File(
        "partial_correlations.jsonl", format=TableJSONLFileFormat, optional=True
    )
    partial_contributions = model.File(
        "partial_contributions.jsonl", format=TableJSONLFileFormat, optional=True
    )
