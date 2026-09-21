# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
from skbio.stats.composition import clr, multi_replace


def transform_clr(
    table: pd.DataFrame,
    pseudocount: float = 1.0,
    replacement_method: str = "multiplicative",
    delta: float = None,
) -> pd.DataFrame:
    """
    Replace zeros and apply a centered log-ratio (CLR) transformation.

    Args:
        table (pd.DataFrame): feature table with samples as rows and features as
            columns containing non-negative values
        pseudocount (float): Value used to replace zeros before CLR. If not
            provided, a pseudocount of 1 is used. This parameter is used only
            when ``replacement_method='pseudocount'``.
        replacement_method (str): Zero-handling strategy to use before CLR.
            ``'multiplicative'`` applies multiplicative replacement and is the
            default. ``'pseudocount'`` adds the pseudocount to the table.
        delta (float): Replacement value used for multiplicative replacement.
            This parameter is only used when
            ``replacement_method='multiplicative'``. If not provided, the
            default behavior of ``skbio.stats.composition.multi_replace`` is
            used.

    Output:
        pd.DataFrame: CLR-transformed feature table where each sample sums to zero
            and values are real-valued log-ratios

    Raises:
        ValueError: If ``replacement_method='multiplicative'`` and at least one
            sample contains only zeros. In that case, the feature table should
            be filtered before applying CLR.
    """
    if replacement_method == "multiplicative":
        # Raise error if one sample is complete 0
        if (table == 0).all(axis=1).any():
            raise ValueError(
                "At least one sample is all 0, please filter the feature table first."
            )
        transformed = multi_replace(table, delta=delta)
    else:
        transformed = table + pseudocount

    transformed = clr(transformed)

    return pd.DataFrame(transformed, index=table.index, columns=table.columns)
