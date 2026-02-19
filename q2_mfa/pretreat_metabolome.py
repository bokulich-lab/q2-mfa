# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import rachis
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer


def normalize_pqn(
    table: pd.DataFrame,
    method: str = "median",
    ref_samples: rachis.CategoricalMetadataColumn = None,
    ref_label: str = None,
) -> pd.DataFrame:
    """
    Applies Probabilistic Quotient Normalization (PQN) to a metabolomics
    feature table.

    PQN corrects for sample dilution effects by:
        1. Computing a reference spectrum (median or mean across samples).
        2. Computing per-sample quotients to the reference.
        3. Dividing each sample by its median quotient.

    Parameters:
        table (pd.DataFrame):
            Feature table (samples x features) with non-negative values.
        method (str):
            Method to compute reference spectrum. Either "median" or "mean".
        ref_samples (rachis.CategoricalMetadataColumn):
            Categorical metadata column to filter samples for reference calculation.
        ref_label (str):
            Label value in ref_samples to identify which samples to use for reference.

    Returns:
        pd.DataFrame:
            PQN-normalized table.

    Raises:
        ValueError:
            If input contains negative values or normalization fails.
    """

    if (table < 0).any().any():
        raise ValueError("PQN requires non-negative intensities.")

    X = table.astype(float).copy()

    # Filter reference samples if metadata provided
    if ref_samples is not None and ref_label is not None:
        ref_series = ref_samples.to_series()
        ref_sample_indices = ref_series[ref_series == ref_label].index.tolist()
        if not ref_sample_indices:
            raise ValueError(
                f"Reference label '{ref_label}' not found in metadata column."
            )
        X_ref = X.loc[ref_sample_indices]
    else:
        X_ref = X

    # Reference spectrum (per feature)
    ref = X_ref.median(axis=0) if method == "median" else X_ref.mean(axis=0)

    # Avoid division by zero in reference
    ref_safe = ref.replace(0.0, np.nan)

    # Compute quotients per sample
    quotients = X.div(ref_safe, axis=1)

    # Median quotient per sample = dilution factor
    factors = quotients.apply(lambda row: np.nanmedian(row.to_numpy()), axis=1)

    if factors.isna().any():
        raise ValueError(
            "PQN failed: could not compute dilution factor "
            "(sample may contain only zeros)."
        )

    if (factors <= 0).any():
        raise ValueError("PQN failed: non-positive dilution factor detected.")

    # Normalize samples
    X = X.div(factors, axis=0)

    return X


def impute_table(
    table: pd.DataFrame,
    impute: str = "knn",
    knn_neighbors: int = 10,
    rf_n_estimators: int = 100,
    rf_random_state: int | None = 0,
) -> pd.DataFrame:
    """
    Helper to perform missing-value imputation on a numeric dataframe with.

    Supports:
        - "knn": sklearn.impute.KNNImputer
        - "rf": IterativeImputer with RandomForestRegressor

    Parameters:
        table (pd.DataFrame):
            Numeric data frame (samples x features) to impute; missing values as NaN.
        impute (str | None):
            Imputation method: "knn", "rf", or None to skip imputation.
        knn_neighbors (int):
            Number of neighbors for KNN imputation (used when impute == "knn").
        rf_n_estimators (int):
            Number of trees for RandomForestRegressor (used when impute == "rf").
        rf_random_state (int | None):
            Random state for RandomForest and IterativeImputer reproducibility.

    Returns:
        pd.DataFrame:
            New DataFrame with imputed values, preserving index and columns.

    Raises:
        ValueError:
            If an unknown imputation method is provided.
    """
    table = table.replace(0, np.nan)

    if impute == "knn":

        imputer = KNNImputer(n_neighbors=knn_neighbors)
        table_imp = pd.DataFrame(
            imputer.fit_transform(table), index=table.index, columns=table.columns
        )

    elif impute == "rf":

        estimator = RandomForestRegressor(
            n_estimators=rf_n_estimators, random_state=rf_random_state
        )
        imp = IterativeImputer(
            estimator=estimator,
            random_state=rf_random_state,
            max_iter=10,
            initial_strategy="mean",
        )
        table_imp = pd.DataFrame(
            imp.fit_transform(table), index=table.index, columns=table.columns
        )

    return table_imp


def transform_table(
    table: pd.DataFrame,
    transform: str = "log",
    pseudocount: float | None = None,
) -> pd.DataFrame:
    """
    Applies a transformation to a feature table.

    Supported transformations:
        - "log": natural logarithm
        - "log10": base-10 logarithm
        - "sqrt": square root

    Parameters:
        table (pd.DataFrame):
            Feature table (samples x features).
        transform (str):
            Transformation type: "log", "log10", "sqrt".
        pseudocount (float | None):
            Offset added before log/log10 transform. If None, uses the minimum
            non-zero value in the table. Must be > 0.

    Returns:
        pd.DataFrame:
            Transformed table with the same shape and index/columns as input.

    Raises:
        ValueError:
            If inputs are invalid or transformation fails.
    """
    if (table < 0).any().any():
        raise ValueError("Transformation requires all values to be non negative.")

    if transform in ("log", "log10"):
        if pseudocount is None:
            min_nonzero = table.where(table > 0).min().min()
            pseudocount = float(min_nonzero)
        if transform == "log":
            table = np.log(table + pseudocount)
        elif transform == "log10":
            table = np.log10(table + pseudocount)

    elif transform == "sqrt":
        table = np.sqrt(table)

    return pd.DataFrame(table, index=table.index, columns=table.columns)


def scale_table(
    table: pd.DataFrame,
    scale: str = "auto",
) -> pd.DataFrame:
    """
    Applies scaling to a feature table.

    Supported scaling methods:
        - "autoscale": mean-center and divide by standard deviation (unit variance)
        - "pareto": mean-center and divide by square root of standard deviation
        - "range": mean-center and divide by max-min range

    Parameters:
        table (pd.DataFrame):
            Feature table (samples x features).
        scale (str | None):
            Scaling method: "auto", "pareto", or "range".

    Returns:
        pd.DataFrame:
            Scaled table with the same shape and index/columns as input.

    Raises:
        ValueError:
            If scaling fails due to zero variance or range.
    """
    # center the table first
    table = table - table.mean(axis=0)

    if scale == "auto":
        sd = table.std(axis=0, ddof=0)
        if (sd == 0).any():
            raise ValueError(
                "Autoscaling not possible: at least one feature has zero variance."
            )
        table = table / sd

    elif scale == "pareto":
        sd = table.std(axis=0, ddof=0)
        if (sd == 0).any():
            raise ValueError(
                "Pareto scaling not possible: at least one feature has zero variance."
            )
        table = table / np.sqrt(sd)

    elif scale == "range":
        rng = table.max(axis=0) - table.min(axis=0)
        if (rng == 0).any():
            raise ValueError(
                "Range scaling not possible: at least one feature has zero range."
            )
        table = table / rng

    return table


def pretreat_metabolome(
    table: pd.DataFrame,
    sample_normalization: str | None = None,
    pqn_method: str = "median",
    pqn_ref_samples: rachis.CategoricalMetadataColumn = None,
    pqn_ref_label: str | None = None,
    transform: str | None = None,
    pseudocount: float | None = None,
    center: bool = False,
    scale: str | None = None,
    impute: str | None = None,
    knn_neighbors: int = 5,
    rf_n_estimators: int = 100,
    rf_random_state: int | None = 0,
) -> pd.DataFrame:
    """
    Applies metabolomics-friendly preprocessing to a feature table.

    Steps (in order):
        0) Optional imputation (knn or rf)
        1) Sample normalization: PQN
        2) Transform (log, log10, sqrt, or none)
        3) Centering (per feature / column)
        4) Scaling (per feature / column): auto, pareto, or range

    Parameters:
        table (pd.DataFrame):
            Feature table (samples x features). Values should be numeric.
        sample_normalization (str):
            "none" or "pqn" (Probabilistic Quotient Normalization).
        pqn_method (str):
            How to build the PQN reference spectrum: "median" or "mean".
        pqn_ref_samples (rachis.CategoricalMetadataColumn):
            Categorical metadata column used to select samples for building the PQN
            reference spectrum.
        pqn_ref_label (str):
            Label value in `pqn_ref_samples` indicating which samples to use for the
            PQN reference. Required when `pqn_ref_samples` is provided.
        transform (str):
            Transformation to apply: "log", "log10", "sqrt", or "none".
        pseudocount (float | None):
            Offset added before log transform. If None and transform starts with "log",
            uses the minimum non-zero value in the table. Must be > 0.
        center (bool):
            If True, mean-center each feature (column).
        scale (str):
            Scaling method: "none", "auto" (unit variance), "pareto"
            (divide by sqrt(std)), or "range" (divide by max-min).
        impute (str | None):
            Missing-value imputation method. Supported: "knn", "rf", or None.
        knn_neighbors (int):
            Number of neighbors for KNN imputation.
        rf_n_estimators (int):
            Number of trees for RandomForest used in iterative imputation.
        rf_random_state (int | None):
            Random state for RandomForest / IterativeImputer reproducibility.

    Returns:
        pd.DataFrame:
            Transformed table with the same shape and index/columns as input.

    Raises:
        ValueError:
            If inputs are invalid or options are unknown.
    """

    X = table.astype(float).copy()

    # 0) Optional imputation (before normalization/transform)
    if impute is not None:
        X = impute_table(
            X,
            impute=impute,
            knn_neighbors=knn_neighbors,
            rf_n_estimators=rf_n_estimators,
            rf_random_state=rf_random_state,
        )

    # 1) Sample normalization (PQN)
    if sample_normalization == "pqn":
        X = normalize_pqn(
            table=X,
            method=pqn_method,
            ref_samples=pqn_ref_samples,
            ref_label=pqn_ref_label,
        )

    # 2) Transform
    if transform is not None and transform != "none":
        X = transform_table(
            X,
            transform=transform,
            pseudocount=pseudocount,
        )

    # 3) Center per feature (column)
    if center:
        X = X - X.mean(axis=0)

    # 4) Scale per feature (column)
    if scale is not None:
        X = scale_table(X, scale=scale)

    return pd.DataFrame(X, index=table.index, columns=table.columns)
