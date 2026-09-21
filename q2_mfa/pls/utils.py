# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from q2_types.feature_table import FeatureTable, Unconstrained
from q2_types.metadata import ImmutableMetadata
from rachis import Metadata, ResultCollection
from rpy2.robjects import (
    ListVector,
    StrVector,
    conversion,
    default_converter,
    pandas2ri,
    r,
)
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import DataFrame as RDataFrame
from rpy2.robjects.vectors import FactorVector


def _align_samples_metadata(ctx, tables, metadata_column):
    """
    Aligns feature tables and a metadata column to shared samples.

    Requires at least two feature tables, removes missing metadata values, and
    restricts every remaining input to the sample IDs they share.

    - Removes missing metadata values.
    - Identifies IDs shared by the remaining column and every feature table
    - Filters the metadata column to those IDs.
    - Filters and reorders every feature table to the metadata sample order.

    Args:
        ctx (Context): Pipeline execution context used to retrieve actions.
        tables (ResultCollection): Named unconstrained feature tables.
        metadata_column (MetadataColumn): Metadata column to align.

    Returns:
        tuple[ResultCollection, Metadata]: A tuple containing:
            - ResultCollection: Tables filtered to the shared sample IDs.
            - Metadata: Aligned metadata containing the supplied column.

    Raises:
        ValueError: If fewer than two feature tables are supplied or no samples
            are shared by the feature tables and metadata column.
    """
    if len(tables) < 2:
        raise ValueError("At least two feature tables are required.")

    filter_ids = ctx.get_action("feature_table", "filter_ids")
    metadata_column = metadata_column.drop_missing_values()
    shared_ids = metadata_column.get_ids()
    for table in tables.values():
        shared_ids.intersection_update(table.view(pd.DataFrame).index)
    if not shared_ids:
        raise ValueError("The feature tables and metadata column have no shared IDs.")
    shared_ids = [str(sample_id) for sample_id in shared_ids]
    metadata_column = metadata_column.filter_ids(shared_ids)
    aligned_metadata = Metadata(metadata_column.to_dataframe())
    sample_ids = aligned_metadata.to_dataframe().index.tolist()

    aligned_tables = {}
    for name, table in tables.items():
        (filtered_table,) = filter_ids(
            table=table,
            axis="sample",
            ids=shared_ids,
        )
        aligned_tables[name] = ctx.make_artifact(
            FeatureTable[Unconstrained],
            filtered_table.view(pd.DataFrame).loc[sample_ids],
        )
    return (
        ResultCollection(aligned_tables),
        ctx.make_artifact(ImmutableMetadata, aligned_metadata),
    )


def _resolve_design(
    design_matrix, design_weight: float | None, block_names: list[str]
) -> pd.DataFrame | float:
    """
    Validates an explicit block design matrix or returns a scalar weight.

    Requires exactly one design specification. Explicit matrices must contain
    the input block names on both axes, be numeric and symmetric, and have a
    zero diagonal; scalar weights are returned unchanged.

    Args:
        design_matrix (Metadata, optional): Explicit block-relationship matrix.
        design_weight (float, optional): Shared off-diagonal relationship
            weight.
        block_names (list[str]): Ordered names of the feature-table blocks.

    Returns:
        pd.DataFrame | float: Validated and ordered design matrix, or the
            supplied scalar design weight.
    """
    if (design_matrix is None) == (design_weight is None):
        raise ValueError("Provide exactly one of 'design-matrix' or 'design-weight'.")
    if design_weight is not None:
        return design_weight

    matrix = design_matrix.to_dataframe().copy()
    expected = set(block_names)
    if set(matrix.index) != expected or set(matrix.columns) != expected:
        raise ValueError(
            "The row and column names in 'design-matrix' must exactly match "
            "the input feature-table collection names."
        )
    matrix = matrix.loc[block_names, block_names]
    try:
        matrix = matrix.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Values in 'design-matrix' must be numeric.") from error
    if not np.allclose(matrix.to_numpy(), matrix.to_numpy().T):
        raise ValueError("'design-matrix' must be symmetrical.")
    if not np.allclose(np.diag(matrix.to_numpy()), 0):
        raise ValueError("The diagonal of 'design-matrix' must contain only zeroes.")
    return matrix


def _build_bpparam(threads: int, seed: int):
    """
    Creates a BiocParallel backend.

    Uses a serial backend for one worker and a multicore backend otherwise. A
    thread count of zero leaves worker selection to BiocParallel.

    Args:
        threads (int): Requested worker count, where zero selects the
            BiocParallel default.
        seed (int): Random seed supplied to the backend.

    Returns:
        rpy2.robjects.RObject: Configured BiocParallel parameter object.
    """
    r("suppressPackageStartupMessages(library(BiocParallel))")
    if threads == 1:
        return r["SerialParam"](RNGseed=seed)
    kwargs = {"RNGseed": seed}
    if threads > 1:
        kwargs["workers"] = threads
    return r["MulticoreParam"](**kwargs)


def _to_r_inputs(
    blocks: dict[str, pd.DataFrame],
    target: pd.Series,
    design: pd.DataFrame | float,
) -> tuple[ListVector, FactorVector, RDataFrame | float]:
    """
    Converts aligned Python inputs to labelled R objects.

    Converts each block and an explicit design matrix through the pandas R
    converter, and creates a named R factor for the categorical response.

    Args:
        blocks (dict[str, pd.DataFrame]): Aligned named feature tables.
        target (pd.Series): Aligned categorical response values.
        design (pd.DataFrame | float): Explicit design matrix or scalar design
            weight.

    Returns:
        tuple[rpy2.robjects.ListVector, rpy2.robjects.vectors.FactorVector,
        rpy2.robjects.RObject]: Named R blocks, response factor, and design.
    """
    with localconverter(default_converter + pandas2ri.converter):
        converter = conversion.get_conversion()
        r_blocks = ListVector(
            {name: converter.py2rpy(table) for name, table in blocks.items()}
        )
        r_design = converter.py2rpy(design)
    r_target = r["factor"](StrVector(target.astype(str).tolist()))
    r_target.names = StrVector(target.index.astype(str).tolist())
    return r_blocks, r_target, r_design


def _r_vote_error_rate_to_dataframe(perf_result: ListVector, vote: str) -> pd.DataFrame:
    """
    Converts mixOmics vote error-rate means and standard deviations to rows.

    Combines the requested vote's error-rate and standard-deviation matrices
    into one table with distance measure, class, component, mean, and standard
    deviation columns.

    Args:
        perf_result (rpy2.robjects.vectors.ListVector): Result returned by
            mixOmics ``perf()``.
        vote (str): Vote type to extract, such as ``WeightedVote`` or
            ``MajorityVote``.

    Returns:
        pd.DataFrame: Error-rate records for the requested vote.

    Raises:
        ValueError: If mixOmics did not provide either the mean or standard
            deviation error-rate matrix for the requested vote.
    """
    columns = ["distance", "class", "component", "mean", "sd"]
    records = {}
    for statistic, error_rates in (
        ("mean", perf_result.rx2(f"{vote}.error.rate")),
        ("sd", perf_result.rx2(f"{vote}.error.rate.sd")),
    ):
        if type(error_rates).__name__ == "NULLType":
            raise ValueError(
                f"mixOmics did not provide {statistic} error rates for {vote}."
            )
        for distance, rates in zip(error_rates.names, error_rates):
            values = np.asarray(rates, dtype=float)
            class_names = [str(name) for name in r["rownames"](rates)]
            for class_name, class_values in zip(class_names, values):
                for component, value in enumerate(class_values, start=1):
                    key = (str(distance), class_name, component)
                    record = records.setdefault(
                        key,
                        {
                            "distance": key[0],
                            "class": key[1],
                            "component": key[2],
                        },
                    )
                    record[statistic] = value

    error_rates = pd.DataFrame(records.values(), columns=columns)
    error_rates["component"] = error_rates["component"].astype("Int64")
    return error_rates
