# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import secrets
import warnings

import pandas as pd
import prince
from rachis.core.exceptions import RachisWarning
from rachis.plugin import CaptureHolder

from q2_mfa.component_analysis import ComponentAnalysisResult


def resolve_random_state(random_state: CaptureHolder[int] | None, engine: str):
    """
    Resolve the random seed used by the selected PCA engine.

    Uses the provided random state when one is supplied for the sklearn engine,
    generates one when needed, and resolves to None for engines that do not use
    a random state.

    Args:
        random_state (CaptureHolder[int]): The optional random seed holder.
        engine (str): The SVD engine used by prince.

    Returns:
        int | None: The resolved random seed for sklearn, or None otherwise.
    """
    if engine == "sklearn":
        return CaptureHolder.get_or_set(random_state, lambda: secrets.randbits(32))
    return CaptureHolder.get_or_set(random_state, lambda: None)


def drop_columns_with_missing_values(table: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns containing missing values from a table.

    Identifies columns with one or more missing values, emits a warning listing
    the removed column names, and returns the table with those columns removed.

    Args:
        table (pd.DataFrame): The input table to filter.

    Returns:
        pd.DataFrame: The table without columns that contain missing values.
    """
    missing_value_columns = table.columns[table.isna().any()]
    if len(missing_value_columns) > 0:
        dropped_columns = ", ".join(map(str, missing_value_columns.to_flat_index()))
        warnings.warn(
            f"Dropped columns with missing values: {dropped_columns}",
            RachisWarning,
            stacklevel=2,
        )
        table = table.drop(columns=missing_value_columns)
    return table


def drop_zero_variance_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with zero variance from a table.

    Identifies columns whose values do not vary, emits a warning listing the
    removed column names, and returns the table with those columns removed.

    Args:
        table (pd.DataFrame): The input table to filter.

    Returns:
        pd.DataFrame: The table without zero-variance columns.
    """
    zero_variance_columns = table.columns[table.nunique(dropna=False) <= 1]
    if len(zero_variance_columns) > 0:
        dropped_columns = ", ".join(map(str, zero_variance_columns.to_flat_index()))
        warnings.warn(
            f"Dropped columns with zero variance: {dropped_columns}",
            RachisWarning,
            stacklevel=2,
        )
        table = table.drop(columns=zero_variance_columns)
    return table


def create_result_object(
    prince_result: prince.PCA | prince.MFA, table: pd.DataFrame
) -> ComponentAnalysisResult:
    """
    Convert a fitted Prince PCA result into a ComponentAnalysisResult object.

    Args:
        prince_result (prince.PCA | prince.MFA): The fitted Prince model.
        table (pd.DataFrame): The input table used to fit the Prince model.

    Returns:
        ComponentAnalysis: The Prince result in component-analysis form.
    """
    mfa_kwargs = {}
    if isinstance(prince_result, prince.MFA):
        mfa_kwargs = {
            "partial_sample_coordinates": (
                prince_result.partial_row_coordinates(table)
            ),
            "group_coordinates": prince_result.group_coordinates_,
            "group_contributions": prince_result.group_contributions_,
            "group_cosine_similarities": prince_result.group_cosine_similarities_,
            "partial_correlations": prince_result.partial_correlations_,
            "partial_contributions": prince_result.partial_contributions_,
        }

    return ComponentAnalysisResult(
        eigenvalues=prince_result.eigenvalues_,
        percentage_of_variance=prince_result.percentage_of_variance_,
        cumulative_percentage_of_variance=(
            prince_result.cumulative_percentage_of_variance_
        ),
        sample_coordinates=prince_result.row_coordinates(table),
        feature_coordinates=prince_result.column_coordinates_,
        sample_cosine_similarities=prince_result.row_cosine_similarities(table),
        sample_contributions=prince_result.row_contributions_,
        feature_correlations=prince_result.column_correlations,
        feature_contributions=prince_result.column_contributions_,
        feature_cosine_similarities=prince_result.column_cosine_similarities_,
        **mfa_kwargs,
    )


def pca(
    table: pd.DataFrame,
    n_components: int = 2,
    rescale_with_mean: bool = True,
    rescale_with_std: bool = True,
    filter_zero_variance: bool = True,
    engine: str = "sklearn",
    n_iter: int = 3,
    random_state: CaptureHolder[int] | None = None,
) -> ComponentAnalysisResult:
    """
    Perform principal component analysis with prince.

    Filters columns with missing values, optionally filters columns with zero
    variance, fits a Prince PCA model, and returns component-analysis output
    tables.

    Args:
        table (pd.DataFrame): The input sample-by-feature table.
        n_components (int): The number of principal components to compute.
        rescale_with_mean (bool): Whether to center each column before SVD.
        rescale_with_std (bool): Whether to scale each column to unit variance
            before SVD.
        filter_zero_variance (bool): Whether to remove zero-variance columns
            before SVD.
        engine (str): The SVD engine used by prince.
        n_iter (int): The number of iterations used by the sklearn randomized
            SVD engine.
        random_state (CaptureHolder[int] | None): The optional random seed
            holder.

    Returns:
        ComponentAnalysisResult: The PCA result in component-analysis form.
    """
    table = drop_columns_with_missing_values(table)
    if filter_zero_variance:
        table = drop_zero_variance_columns(table)

    random_state = resolve_random_state(random_state, engine)

    pca_params = locals()
    pca_params.pop("table")
    pca_params.pop("filter_zero_variance")

    pca_result = prince.PCA(copy=True, check_input=True, **pca_params).fit(table)
    return create_result_object(pca_result, table)
