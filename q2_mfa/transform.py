# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd


def transform_clr(
    table: pd.DataFrame,
    pseudocount: float = None,
) -> pd.DataFrame:
    """
    Adds a pseudocount to a feature table and applies a centered log-ratio (CLR)
    transformation.  The pseudocount can be provided explicitly or computed as the
    minimum non-zero value in the table.

    Args:
        table (pd.DataFrame): feature table with samples as rows and features as
            columns containing non-negative values
        pseudocount (float): value added to all entries prior to transformation. If
            it is set to None, the pseudocount is computed as the minimum non-zero
            value.

    Output:
        pd.DataFrame: CLR-transformed feature table where each sample sums to zero
            and values are real-valued log-ratios
    """

    if not pseudocount:
        pseudocount = table[table > 0].min().min()

    table = np.log(table + pseudocount)
    table = table.sub(table.mean(axis=1), axis=0)
    return table
