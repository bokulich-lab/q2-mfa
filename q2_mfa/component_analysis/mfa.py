# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import warnings
from collections.abc import Iterable, Mapping
from functools import reduce

import pandas as pd
import prince
from rachis import Metadata
from rachis.core.exceptions import RachisWarning
from rachis.plugin import CaptureHolder

from q2_mfa.component_analysis.pca import (
    create_result_object,
    drop_columns_with_missing_values,
    drop_zero_variance_columns,
    resolve_random_state,
)
from q2_mfa.component_analysis.types import ComponentAnalysisResult


def _parse_metadata_groups(
    metadata_groups: str | Mapping[str, str] | None,
    metadata_columns: Iterable[str],
) -> dict[str, list[str]]:
    """
    Resolves metadata group specifications into group-to-column mappings.

    Converts optional metadata group input into a dictionary where keys are MFA
    group names and values are lists of metadata column names. If no mapping is
    provided, all metadata columns are assigned to the default ``metadata``
    group. If a string is provided, all metadata columns are assigned to a group
    with that string as its name. If a mapping is provided, its keys are group
    names and its comma-separated string values are parsed as metadata columns.

    Args:
        metadata_groups (str | dict | None): The metadata group specification.
        metadata_columns (Iterable[str]): The available metadata column names.

    Returns:
        dict: Metadata group names mapped to lists of metadata column names.
    """
    if metadata_groups is None:
        return {"metadata": list(metadata_columns)}

    if isinstance(metadata_groups, str):
        metadata_groups = metadata_groups.strip()
        if not metadata_groups:
            raise ValueError("Metadata group mapping cannot be empty.")
        return {metadata_groups: list(metadata_columns)}

    group_mapping = {}
    for group_name, columns in metadata_groups.items():
        columns = [column.strip() for column in columns.split(",") if column.strip()]
        if not columns:
            raise ValueError(
                f"Metadata group '{group_name}' must contain at least one column."
            )
        group_mapping[group_name] = columns
    return group_mapping


def _validate_metadata_group_column_types(
    sample_metadata: Metadata,
    group_mapping: Mapping[str, Iterable[str]],
) -> None:
    """
    Validates that each metadata group contains only one metadata column type.

    Args:
        sample_metadata (Metadata): The sample metadata containing the grouped
            columns.
        group_mapping (dict): Metadata group names mapped to metadata column
            names.

    Raises:
        ValueError: If any metadata group contains multiple metadata column
            types.
    """
    metadata_columns = sample_metadata.columns
    for group_name, columns in group_mapping.items():
        column_types = {
            metadata_columns[column].type
            for column in columns
            if column in metadata_columns
        }
        if len(column_types) > 1:
            raise ValueError(
                f"Metadata group '{group_name}' contains multiple column "
                "types. Groups must contain only numeric or only categorical columns."
            )


def _metadata_to_grouped_tables(
    sample_metadata: Metadata | None,
    metadata_groups: str | Mapping[str, str] | None,
) -> dict[str, pd.DataFrame]:
    """
    Converts sample metadata into per-group DataFrames for MFA.

    Converts a metadata object to a DataFrame and splits it into one
    DataFrame per requested metadata group. The function validates that all
    requested metadata columns exist, and that no metadata column is assigned to
    more than one metadata group.

    Args:
        sample_metadata (Metadata | None): The sample metadata to include in
            the MFA input.
        metadata_groups (str | dict | None): The metadata group specification.

    Returns:
        dict: Metadata group names mapped to DataFrames containing the selected
            metadata columns.
    """
    if sample_metadata is None:
        if metadata_groups is not None:
            raise ValueError("metadata_groups requires sample_metadata.")
        return {}

    metadata = sample_metadata.to_dataframe().copy()
    group_mapping = _parse_metadata_groups(metadata_groups, metadata.columns)
    _validate_metadata_group_column_types(sample_metadata, group_mapping)
    missing_columns = sorted(
        {
            column
            for columns in group_mapping.values()
            for column in columns
            if column not in metadata.columns
        }
    )
    if missing_columns:
        raise ValueError(
            "Metadata group mapping references columns not present in the metadata: "
            f"{', '.join(missing_columns)}"
        )

    duplicated_columns = sorted(
        {
            column
            for columns in group_mapping.values()
            for column in columns
            if sum(column in group for group in group_mapping.values()) > 1
        }
    )
    if duplicated_columns:
        raise ValueError(
            "Metadata columns cannot be assigned to multiple groups: "
            f"{', '.join(duplicated_columns)}"
        )

    grouped_tables = {}
    for group_name, columns in group_mapping.items():
        grouped_tables[group_name] = metadata.loc[:, columns]

    return grouped_tables


def _build_prince_input(
    tables: Mapping[str, pd.DataFrame] | None = None,
    sample_metadata: Metadata | None = None,
    metadata_groups: str | Mapping[str, str] | None = None,
    filter_zero_variance: bool = True,
) -> pd.DataFrame:
    """
    Builds the grouped wide input table expected by ``prince.MFA``.

    Merges feature-table groups and metadata groups, keeps only sample IDs that
    are shared across every group, and stores the group labels in the first
    level of a two-level column MultiIndex. All features with missing values
    and 0 variance (if filter_zero_variance = True) are dropped. If a group is
    empty after filtering it is also dropped and the user is warned.

    Args:
        tables (dict | None): Feature table groups where keys are MFA group
            names and values are DataFrames.
        sample_metadata (Metadata | None): Optional sample metadata to include
            as MFA groups.
        metadata_groups (str | dict | None): Optional metadata group specification.
        filter_zero_variance (bool): Whether to remove zero-variance columns
            before ordination.

    Returns:
        pd.DataFrame: MFA input table with grouped MultiIndex columns.
    """
    tables = {} if tables is None else dict(getattr(tables, "collection", tables))

    metadata_tables = _metadata_to_grouped_tables(sample_metadata, metadata_groups)

    duplicate_groups = sorted(set(tables).intersection(metadata_tables))
    if duplicate_groups:
        raise ValueError(
            "Metadata group names cannot duplicate feature table group names: "
            f"{', '.join(duplicate_groups)}"
        )
    tables.update(metadata_tables)

    for group_name in tables:
        if not group_name.strip():
            raise ValueError("MFA group names cannot be empty strings.")

    table_values = list(tables.values())
    consensus_samples = (
        reduce(
            lambda shared, table: shared.intersection(table.index),
            table_values[1:],
            table_values[0].index,
        )
        if table_values
        else pd.Index([])
    )

    if table_values and consensus_samples.empty:
        raise ValueError("MFA inputs do not share any sample IDs.")

    grouped_tables = []
    for group_name, table in tables.items():
        dropped_samples = table.index.difference(consensus_samples)
        if not dropped_samples.empty:
            warnings.warn(
                f"Dropping samples from group '{group_name}' that are not "
                f"shared across all tables:\n{', '.join(dropped_samples)}",
                RachisWarning,
            )

        table = table.loc[consensus_samples].copy()
        table = pd.concat({group_name: table}, axis=1)
        table = drop_columns_with_missing_values(table)
        if filter_zero_variance:
            table = drop_zero_variance_columns(table)
        if table.empty:
            warnings.warn(
                (
                    f"Dropped MFA group '{group_name}' because all "
                    "features were removed during missing value filtering or "
                    "zero-variance filtering."
                ),
                RachisWarning,
                stacklevel=2,
            )
            continue
        grouped_tables.append(table)

    if len(grouped_tables) < 2:
        raise ValueError(
            "MFA requires at least two groups after filtering. Groups may have "
            "been removed because all features were removed during missing "
            "value filtering or zero-variance filtering."
        )

    return pd.concat(grouped_tables, axis=1)


def mfa(
    tables: pd.DataFrame = None,
    sample_metadata: Metadata | None = None,
    metadata_groups: dict | None = None,
    n_components: int = 2,
    rescale_with_mean: bool = True,
    rescale_with_std: bool = True,
    filter_zero_variance: bool = True,
    engine: str = "sklearn",
    n_iter: int = 3,
    random_state: CaptureHolder[int] | None = None,
) -> ComponentAnalysisResult:
    """
    Runs Multiple Factor Analysis and returns all Prince-derived outputs.

    Combines feature tables and optional sample metadata into Prince's MFA input
    representation, fits ``prince.MFA``, and returns a component-analysis object
    with MFA-specific coordinate, contribution, correlation, and
    cosine-similarity tables.

    Args:
        tables (pd.DataFrame | None): Feature table collection where each
            collection key is treated as an MFA group.
        sample_metadata (Metadata | None): Optional sample metadata to include
            as MFA groups.
        metadata_groups (dict | None): Optional metadata group mapping where
            keys are group names and values are comma-separated metadata column
            names.
        n_components (int): Number of principal components to compute.
        rescale_with_mean (bool): Whether Prince should center features before
            SVD.
        rescale_with_std (bool): Whether Prince should standardize features
            before SVD.
        filter_zero_variance (bool): Whether to remove zero-variance columns
            before SVD.
        engine (str): Prince SVD engine to use.
        n_iter (int): Number of iterations used by the randomized SVD engine.
        random_state (CaptureHolder[int] | None): Random seed capture used for
            reproducible randomized SVD.

    Returns:
        ComponentAnalysisResult: The MFA result in component-analysis form.
    """
    random_state = resolve_random_state(random_state, engine)

    mfa_params = locals()
    mfa_params.pop("tables")
    mfa_params.pop("sample_metadata")
    mfa_params.pop("metadata_groups")
    mfa_params.pop("filter_zero_variance")

    table = _build_prince_input(
        tables,
        sample_metadata,
        metadata_groups,
        filter_zero_variance=filter_zero_variance,
    )

    mfa_result = prince.MFA(**mfa_params).fit(table)
    return create_result_object(mfa_result, table)
