# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from q2_types.tabular import TableJSONLFileFormat
from q2_types.tabular._deferred_setup._transformers import (
    df_to_table_jsonl,
    table_jsonl_to_df,
)
from skbio import OrdinationResults

from q2_mfa.plugin_setup import plugin

from ._format import ComponentAnalysisDirFmt
from ._result import ComponentAnalysisResult


@dataclass(frozen=True)
class _TableSpec:
    attr: str
    kind: str
    description: str
    index: str | None = None
    required: bool = True
    row_index: tuple[str, ...] | None = None
    restored_index: tuple[str, ...] | None = None


_TABLE_SPECS = (
    _TableSpec(
        "eigenvalues",
        "vector",
        "Variance captured by each component. Prince source: `eigenvalues_`.",
    ),
    _TableSpec(
        "percentage_of_variance",
        "vector",
        "Percent of total variance explained by each component. "
        "Prince source: `percentage_of_variance_`.",
    ),
    _TableSpec(
        "cumulative_percentage_of_variance",
        "vector",
        "Cumulative percent variance explained across components. "
        "Prince source: `cumulative_percentage_of_variance_`.",
    ),
    _TableSpec(
        "sample_coordinates",
        "wide",
        "Sample positions on the component axes. "
        "Prince source: `row_coordinates(table)`.",
        index="sample_id",
    ),
    _TableSpec(
        "sample_cosine_similarities",
        "wide",
        "Quality of each sample's representation on each component. "
        "Prince source: `row_cosine_similarities(table)`.",
        index="sample_id",
    ),
    _TableSpec(
        "sample_contributions",
        "wide",
        "Contribution of each sample to each component. "
        "Prince source: `row_contributions_`.",
        index="sample_id",
    ),
    _TableSpec(
        "feature_coordinates",
        "indexed_rows",
        "Feature coordinates, also called feature loadings, on the component "
        "axes. Prince source: `column_coordinates_`.",
        index="variable",
        row_index=("group", "variable"),
    ),
    _TableSpec(
        "feature_correlations",
        "indexed_rows",
        "Correlation of each feature with each component. "
        "Prince source: `column_correlations`.",
        index="variable",
        row_index=("group", "variable"),
    ),
    _TableSpec(
        "feature_contributions",
        "indexed_rows",
        "Contribution of each feature to each component. "
        "Prince source: `column_contributions_`.",
        index="variable",
        row_index=("group", "variable"),
    ),
    _TableSpec(
        "feature_cosine_similarities",
        "indexed_rows",
        "Quality of each feature's representation on each component. "
        "Prince source: `column_cosine_similarities_`.",
        index="variable",
        row_index=("group", "variable"),
    ),
    _TableSpec(
        "group_coordinates",
        "wide",
        "Group positions on the component axes. "
        "Prince source: `group_coordinates_`.",
        index="group",
        required=False,
    ),
    _TableSpec(
        "group_contributions",
        "wide",
        "Contribution of each group to each component. "
        "Prince source: `group_contributions_`.",
        index="group",
        required=False,
    ),
    _TableSpec(
        "group_cosine_similarities",
        "wide",
        "Quality of each group's representation on each component. "
        "Prince source: `group_cosine_similarities_`.",
        index="group",
        required=False,
    ),
    _TableSpec(
        "partial_sample_coordinates",
        "multi_columns",
        "Sample positions by group and partial PCA component. "
        "Prince source: `partial_row_coordinates(table)`.",
        required=False,
    ),
    _TableSpec(
        "partial_correlations",
        "indexed_rows",
        "Correlation between each group's partial PCA component and each "
        "global MFA component. Prince source: `partial_correlations_`.",
        required=False,
        row_index=("group", "partial_component"),
        restored_index=("group", "component"),
    ),
    _TableSpec(
        "partial_contributions",
        "indexed_rows",
        "Contribution of each group's partial PCA component to each global "
        "MFA component. Prince source: `partial_contributions_`.",
        required=False,
        row_index=("group", "partial_component"),
        restored_index=("group", "component"),
    ),
)


@plugin.register_transformer
def _component_analysis_to_dirfmt(
    result: ComponentAnalysisResult,
) -> ComponentAnalysisDirFmt:
    """
    Converts a ComponentAnalysisResult to a ComponentAnalysisDirFmt.

    Args:
        result (ComponentAnalysisResult): The PCA or MFA result object.

    Returns:
        ComponentAnalysisDirFmt: The JSONL-backed directory format.
    """
    ff = ComponentAnalysisDirFmt()
    for spec in _TABLE_SPECS:
        table = getattr(result, spec.attr)
        if table is None:
            if spec.required:
                raise ValueError(f"Missing required table: {spec.attr}.")
            continue
        _write_result_table(ff.path / f"{spec.attr}.jsonl", table, spec)
    return ff


@plugin.register_transformer
def _dirfmt_to_component_analysis(
    ff: ComponentAnalysisDirFmt,
) -> ComponentAnalysisResult:
    """
    Converts ComponentAnalysisDirFmt to a ComponentAnalysisResult object.

    Args:
        ff (ComponentAnalysisDirFmt): The JSONL-backed directory format.

    Returns:
        ComponentAnalysisResult: The reconstructed result.
    """
    kwargs = {}
    for spec in _TABLE_SPECS:
        path = ff.path / f"{spec.attr}.jsonl"
        if not path.exists():
            kwargs[spec.attr] = None
            continue
        kwargs[spec.attr] = _read_result_table(path, spec)
    return ComponentAnalysisResult(**kwargs)


@plugin.register_transformer
def _dirfmt_to_ordination_results(
    ff: ComponentAnalysisDirFmt,
) -> OrdinationResults:
    """
    Converts ComponentAnalysisDirFmt to scikit-bio ordination results.

    Args:
        ff (ComponentAnalysisDirFmt): The JSONL-backed directory format.

    Returns:
        OrdinationResults: The reconstructed ordination result.
    """
    result = _dirfmt_to_component_analysis(ff)
    method = "MFA" if result.is_mfa else "PCA"
    long_name = (
        "Multiple Factor Analysis" if result.is_mfa else "Principal Component Analysis"
    )
    return OrdinationResults(
        short_method_name=method,
        long_method_name=long_name,
        eigvals=result.eigenvalues,
        samples=result.sample_coordinates,
        features=result.feature_coordinates,
        proportion_explained=result.percentage_of_variance / 100,
    )


def _write_result_table(
    path: Path, table: pd.DataFrame | np.ndarray, spec: _TableSpec
) -> None:
    """
    Writes a result table in long TableJSONL form.

    Args:
        path (Path): The JSONL output path.
        table (pd.DataFrame | np.ndarray): The table or vector to write.
        spec (_TableSpec): The table serialization specification.

    Returns:
        None
    """
    if spec.kind == "vector":
        records = _vector_to_long(table)
    elif spec.kind == "wide":
        records = _wide_to_long(table, spec.index)
    elif spec.kind == "indexed_rows":
        records = _indexed_rows_to_long(table, spec)
    elif spec.kind == "multi_columns":
        records = _multi_columns_to_long(table)
    else:  # pragma: no cover
        raise ValueError(f"Unknown table kind: {spec.kind}.")
    records = records.reset_index(drop=True).convert_dtypes()
    records.attrs["description"] = spec.description
    for column in records.columns:
        records[column].attrs["description"] = ""
    jsonl = df_to_table_jsonl(records)
    shutil.copyfile(str(jsonl), path)


def _read_result_table(path: Path, spec: _TableSpec) -> pd.DataFrame | np.ndarray:
    """
    Reads a long TableJSONL result table into Prince-shaped wide form.

    Args:
        path (Path): The JSONL input path.
        spec (_TableSpec): The table serialization specification.

    Returns:
        pd.DataFrame | np.ndarray: The reconstructed wide table or vector.
    """
    table = table_jsonl_to_df(TableJSONLFileFormat(str(path), mode="r"))
    if spec.kind == "vector":
        return _read_vector(table)
    elif spec.kind == "wide":
        return _long_to_wide(table, spec)
    elif spec.kind == "indexed_rows":
        return _long_to_indexed_rows(table, spec)
    elif spec.kind == "multi_columns":
        return _long_to_multi_columns(table)
    else:  # pragma: no cover
        raise ValueError(f"Unknown table kind: {spec.kind}.")


def _wide_to_long(table: pd.DataFrame, index: str) -> pd.DataFrame:
    """
    Converts a simple wide table to long component records.

    Args:
        table (pd.DataFrame): The wide table.
        index (str): The storage column name for the row index.

    Returns:
        pd.DataFrame: Long records for JSONL storage.
    """
    working = table.copy()
    working.index.name = index
    return working.reset_index().melt(
        id_vars=[index], var_name="component", value_name="value"
    )


def _long_to_wide(table: pd.DataFrame, spec: _TableSpec) -> pd.DataFrame:
    """
    Reconstructs a simple wide table from long component records.

    Args:
        table (pd.DataFrame): The long table records.
        spec (_TableSpec): The table serialization specification.

    Returns:
        pd.DataFrame: The reconstructed wide table.
    """
    index = spec.index
    wide_table = table.pivot(index=index, columns="component", values="value")
    wide_table.index.name = None if index == "sample_id" else index
    wide_table.columns.name = "component"
    wide_table.index = _restore_identifier_index(wide_table.index)
    wide_table.columns = _restore_identifier_index(wide_table.columns)
    return wide_table


def _multi_columns_to_long(table: pd.DataFrame) -> pd.DataFrame:
    """
    Converts partial sample coordinates with MultiIndex columns to long records.

    Args:
        table (pd.DataFrame): The partial sample coordinate table.

    Returns:
        pd.DataFrame: Long records for JSONL storage.
    """
    records = table.stack(level=[0, 1], future_stack=True).reset_index()
    records.columns = ["sample_id", "group", "partial_component", "value"]
    return records


def _long_to_multi_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs partial sample coordinates with MultiIndex columns.

    Args:
        table (pd.DataFrame): The long table records.
    Returns:
        pd.DataFrame: The reconstructed partial sample coordinate table.
    """
    partial_table = table.pivot(
        index="sample_id", columns=["group", "partial_component"], values="value"
    )
    partial_table.index.name = None
    partial_table.columns.names = [None, None]
    partial_table.index = _restore_identifier_index(partial_table.index)
    partial_table.columns = _restore_identifier_index(partial_table.columns)
    return partial_table


def _indexed_rows_to_long(table: pd.DataFrame, spec: _TableSpec) -> pd.DataFrame:
    """
    Converts a table with indexed rows to long component records.

    Flat PCA feature tables fall back to wide serialization; MFA feature and
    partial-axis tables serialize each MultiIndex row level as its own field.

    Args:
        table (pd.DataFrame): The wide indexed-row table.
        spec (_TableSpec): The table serialization specification.

    Returns:
        pd.DataFrame: Long records for JSONL storage.
    """
    if not isinstance(table.index, pd.MultiIndex):
        return _wide_to_long(table, spec.index)
    index_columns = list(spec.row_index)
    working = table.copy()
    working.index = working.index.set_names(index_columns)
    return working.reset_index().melt(
        id_vars=index_columns,
        var_name="component",
        value_name="value",
    )


def _long_to_indexed_rows(table: pd.DataFrame, spec: _TableSpec) -> pd.DataFrame:
    """
    Reconstructs an indexed-row table from long component records.

    Flat PCA feature records fall back to wide reconstruction; MFA feature and
    partial-axis records restore configured MultiIndex row levels.

    Args:
        table (pd.DataFrame): The long table records.
        spec (_TableSpec): The table serialization specification.

    Returns:
        pd.DataFrame: The reconstructed indexed-row table.
    """
    index_columns = list(spec.row_index)
    if not set(index_columns).issubset(table.columns):
        return _long_to_wide(table, spec)
    index_names = list(spec.restored_index or spec.row_index)
    indexed_table = table.pivot(
        index=index_columns, columns="component", values="value"
    )
    indexed_table.index = indexed_table.index.set_names(index_names)
    indexed_table.columns.name = "component"
    indexed_table.index = _restore_identifier_index(indexed_table.index)
    indexed_table.columns = _restore_identifier_index(indexed_table.columns)
    return indexed_table


def _restore_identifier_index(index: pd.Index) -> pd.Index:
    """
    Restores string identifier levels to object dtype.

    TableJSONL reads string columns as pandas string-extension dtype, while
    Prince returns sample, feature, and group identifiers as object string
    labels. Numeric component labels are preserved.

    Args:
        index (pd.Index): The index or MultiIndex to normalize.

    Returns:
        pd.Index: The index with string identifier levels restored to object
            dtype.
    """
    if isinstance(index, pd.MultiIndex):
        levels = [
            level.astype(object) if pd.api.types.is_string_dtype(level) else level
            for level in index.levels
        ]
        return index.set_levels(levels)
    if pd.api.types.is_string_dtype(index):
        return index.astype(object)
    return index


def _vector_to_long(vector: np.ndarray) -> pd.DataFrame:
    """
    Converts a component vector to long records.

    Args:
        vector (np.ndarray): The component values to write.

    Returns:
        pd.DataFrame: Long records for JSONL storage.
    """
    values = np.asarray(vector, dtype=float)
    return pd.DataFrame(
        {
            "component": np.arange(len(values), dtype=int),
            "value": values,
        }
    )


def _read_vector(table: pd.DataFrame) -> np.ndarray:
    """
    Reconstructs a component vector from long table records.

    Args:
        table (pd.DataFrame): The long table records read from TableJSONL,
            with ``component`` and ``value`` columns.

    Returns:
        np.ndarray: The reconstructed component values.
    """
    ordered = table.sort_values("component")
    return ordered["value"].to_numpy(dtype=float)
