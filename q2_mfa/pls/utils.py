# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import warnings

import numpy as np
import pandas as pd
from rachis.core.exceptions import RachisWarning
from rachis.plugin import CaptureHolder
from rpy2.robjects import (
    IntVector,
    ListVector,
    conversion,
    default_converter,
    pandas2ri,
)
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame as RDataFrame
from rpy2.robjects.vectors import FactorVector

from q2_mfa.pls.types._format import PLSComponentSelectionDirFmt

base = importr("base")
biocparallel = importr("BiocParallel")


def _align_samples(
    tables: dict[str, pd.DataFrame], target: pd.Series
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """
    Aligns feature tables and a categorical response to shared samples.

    Drops missing response values, retains only sample IDs shared by every
    block and the response, and warns for each block or response row that is
    excluded.

    Args:
        tables (dict[str, pd.DataFrame]): Named feature tables indexed by
            sample ID.
        target (pd.Series): Categorical response values indexed by sample ID.

    Returns:
        tuple[dict[str, pd.DataFrame], pd.Series]: A tuple containing:
            - dict[str, pd.DataFrame]: Aligned named feature tables.
            - pd.Series: Aligned response values.

    Raises:
        ValueError: If fewer than two feature tables are supplied or no samples
            are shared by the feature tables and response.
    """
    if len(tables) < 2:
        raise ValueError("At least two named feature tables are required.")
    target = target.dropna()
    blocks = {**tables, "y": target.to_frame()}

    shared = next(iter(blocks.values())).index
    for table in list(blocks.values())[1:]:
        shared = shared.intersection(table.index)
    if shared.empty:
        raise ValueError("The input feature tables and response share no sample IDs.")

    aligned = {}
    for name, table in blocks.items():
        dropped = table.index.difference(shared)
        if not dropped.empty:
            warnings.warn(
                f"Dropping samples from block '{name}' that are not shared "
                f"across all blocks:\n"
                f"{', '.join(map(str, dropped))}",
                RachisWarning,
                stacklevel=2,
            )
        aligned[str(name)] = table.loc[shared].copy()
    target = aligned.pop("y").iloc[:, 0]
    return aligned, target


def _resolve_design(
    design_matrix: pd.DataFrame | None,
    design_weight: float | None,
    block_names: list[str],
) -> pd.DataFrame | float:
    if (design_matrix is None) == (design_weight is None):
        raise ValueError("Provide exactly one of 'design-matrix' or 'design-weight'.")
    if design_weight is not None:
        return design_weight

    expected = set(block_names)
    if set(design_matrix.index) != expected or set(design_matrix.columns) != expected:
        raise ValueError(
            "The row and column names in 'design-matrix' must exactly match "
            "the input feature-table collection names."
        )
    # reorder the matrix by block order
    design_matrix = design_matrix.loc[block_names, block_names]
    return design_matrix


def _build_bpparam(threads: int):
    """
    Creates a BiocParallel backend.

    Uses a serial backend for one worker and a multicore backend otherwise. A
    thread count of zero leaves worker selection to BiocParallel.

    Args:
        threads (int): Requested worker count, where zero selects the
            BiocParallel default.
    Returns:
        rpy2.robjects.RObject: Configured BiocParallel parameter object.
    """
    if threads == 1:
        return biocparallel.SerialParam()
    if threads > 1:
        return biocparallel.MulticoreParam(workers=threads)
    return biocparallel.MulticoreParam()


def _to_r_inputs(
    blocks: dict[str, pd.DataFrame],
    target: pd.Series,
    design: pd.DataFrame | float,
    keep_x: dict[str, list[int]] | None = None,
    already_tested_x: dict[str, list[int]] | None = None,
) -> tuple[
    ListVector, FactorVector, RDataFrame | float, ListVector | None, ListVector | None
]:
    """
    Converts aligned Python inputs to labelled R objects.

    Converts each block and an explicit design matrix through the pandas R
    converter, and creates a named R factor for the categorical response.

    Args:
        blocks (dict[str, pd.DataFrame]): Aligned named feature tables.
        target (pd.Series): Aligned categorical response values.
        design (pd.DataFrame | float): Explicit design matrix or scalar design
            weight.
        keep_x (dict, optional): Candidate keepX counts by block.
        already_tested_x (dict, optional): Selected keepX counts for prior
            components.

    Returns:
        tuple[rpy2.robjects.ListVector, rpy2.robjects.vectors.FactorVector,
        rpy2.robjects.RObject, ListVector | None, ListVector | None]: Named R
        blocks, response factor, design, and optional keepX and prior selections.
    """
    with localconverter(default_converter + pandas2ri.converter):
        converter = conversion.get_conversion()
        r_blocks = ListVector(
            {name: converter.py2rpy(table) for name, table in blocks.items()}
        )
        r_design = converter.py2rpy(design)
        r_target = converter.py2rpy(target)
    r_target = base.factor(r_target)
    r_keep_x, r_already_tested_x = (
        (
            ListVector({name: IntVector(values) for name, values in counts.items()})
            if counts is not None
            else None
        )
        for counts in (keep_x, already_tested_x)
    )
    return r_blocks, r_target, r_design, r_keep_x, r_already_tested_x


def _r_perf_error_rate_to_df(perf_result: ListVector, vote: str) -> pd.DataFrame:
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
    """
    mean_error_rates = perf_result.rx2(f"{vote}.error.rate")
    sd_error_rates = perf_result.rx2(f"{vote}.error.rate.sd")
    tables = []

    for distance, mean_rates, sd_rates in zip(
        mean_error_rates.names, mean_error_rates, sd_error_rates
    ):
        means = np.asarray(mean_rates, dtype=float)
        standard_deviations = np.asarray(sd_rates, dtype=float)
        class_names = [str(name) for name in base.rownames(mean_rates)]
        component_count = means.shape[1]
        tables.append(
            pd.DataFrame(
                {
                    "distance": str(distance),
                    "class": np.repeat(class_names, component_count),
                    "component": np.tile(
                        np.arange(1, component_count + 1), len(class_names)
                    ),
                    "mean": means.ravel(),
                    "sd": standard_deviations.ravel(),
                }
            )
        )

    error_rates = pd.concat(tables, ignore_index=True)
    error_rates["component"] = error_rates["component"].astype("Int64")
    return error_rates


def _resolve_ncomp(
    ncomp: CaptureHolder[int],
    tuned_ncomp: PLSComponentSelectionDirFmt | None,
    weighted: bool,
    measure: str,
    distance: str,
) -> int:
    """
    Captures a manual component count or derives it from a tuning artifact.

    Uses a manually supplied ``ncomp`` when available; otherwise, reads the
    count selected by the requested voting, measure, and distance criteria from
    the component-tuning artifact. The resolved value is captured for action
    provenance.

    Args:
        ncomp (CaptureHolder[int] | int | None): Manually supplied component
            count or capture holder for the resolved value.
        tuned_ncomp (PLSComponentSelectionDirFmt | None): Component-tuning artifact
            used when no manual component count is supplied.
        weighted (bool): Whether to use the weighted rather than majority vote.
        measure (str): Requested prediction-error measure.
        distance (str): Requested prediction distance.

    Returns:
        int: The selected number of components.

    Raises:
        ValueError: If exactly one of a manual count and tuning artifact is not
            provided, or the artifact has no unique matching selection.
    """
    manual_ncomp = ncomp.is_set
    has_tuned_ncomp = tuned_ncomp is not None
    if manual_ncomp == has_tuned_ncomp:
        raise ValueError("Provide exactly one of ncomp or a tuned-ncomp artifact.")
    return CaptureHolder.get_or_set(
        ncomp,
        lambda: _ncomp_from_tuned_ncomp(tuned_ncomp, weighted, measure, distance),
    )


def _ncomp_from_tuned_ncomp(
    tuned_ncomp: PLSComponentSelectionDirFmt,
    weighted: bool,
    measure: str,
    distance: str,
) -> int:
    """
    Reads the component count chosen for a requested tuning criterion.

    Reads the weighted- or majority-vote choice matrix from the current
    component-tuning artifact.

    Args:
        tuned_ncomp (PLSComponentSelectionDirFmt): Component-tuning artifact
            containing the selected component counts.
        weighted (bool): Whether to use the weighted rather than majority vote.
        measure (str): Requested prediction-error measure.
        distance (str): Requested prediction distance.

    Returns:
        int: The uniquely selected number of components.

    Raises:
        ValueError: If the artifact has no selected component for the requested
            vote, measure, and distance.
    """
    requested_measure = "Overall.BER" if measure == "BER" else "Overall.ER"
    matrix_file = (
        tuned_ncomp.choice_matrix_weighted
        if weighted
        else tuned_ncomp.choice_matrix_majority
    )
    matrix = matrix_file.view(pd.DataFrame)
    selected_ncomp = matrix.loc[matrix["id"] == requested_measure, distance].iloc[0]
    if pd.isna(selected_ncomp):
        raise ValueError(
            "The component-tuning artifact has no selected number of components for "
            "the requested vote, measure, and distance."
        )
    return int(selected_ncomp)
