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
    error_rate = model.FileCollection(
        r"error_rate(?:_(?:weighted|majority))?\.jsonl$",
        format=TableJSONLFileFormat,
    )
    choice_matrix = model.FileCollection(
        r"choice_matrix(?:_(?:weighted|majority))?\.jsonl$",
        format=TableJSONLFileFormat,
    )

    @error_rate.set_path_maker
    def error_rate_path_maker(self, vote=None):
        suffix = f"_{vote}" if vote else ""
        return f"error_rate{suffix}.jsonl"

    @choice_matrix.set_path_maker
    def choice_matrix_path_maker(self, vote=None):
        suffix = f"_{vote}" if vote else ""
        return f"choice_matrix{suffix}.jsonl"


AssociationMatrixDirFmt = model.SingleFileDirectoryFormat(
    "AssociationMatrixDirFmt", "association-matrix.tsv", LSMatFormat
)


class PLSFeatureSelectionDirFmt(model.DirectoryFormat):
    error_rate = model.File("error_rate.jsonl", format=TableJSONLFileFormat)
    selected_features = model.File(
        "selected_features.jsonl", format=TableJSONLFileFormat
    )
