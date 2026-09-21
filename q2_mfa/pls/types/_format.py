# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import rachis.plugin.model as model
from q2_types.tabular import TableJSONLFileFormat


class PLSTuneComponentsDirFmt(model.DirectoryFormat):
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
