# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import secrets
import warnings

import numpy as np
import pandas as pd
import rachis
from rachis.core.exceptions import RachisWarning
from rachis.core.type import CaptureHolder

from q2_mfa.utils import run_r_table_script


def normalize_tic(table: pd.DataFrame) -> pd.DataFrame:
    """
    Applies total ion current (TIC) normalization to a feature table.

    TIC normalization scales each sample by its total signal, so each row sums
    to one. This is also known as total-area or sum normalization.

    Args:
        table (pd.DataFrame): Feature table (samples x features) with
            non-negative values.

    Returns:
        pd.DataFrame: TIC-normalized table with the same index and columns.

    Raises:
        ValueError: If input contains negative values or any sample has zero
            total signal.
    """
    if (table < 0).any().any():
        raise ValueError("TIC normalization requires non-negative intensities.")

    sample_sums = table.sum(axis=1)

    if (sample_sums == 0).any():
        raise ValueError("TIC normalization failed: zero sample sum detected.")

    return table.div(sample_sums, axis=0)


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

    The original PQN paper includes total-area/TIC normalization before the
    quotient step. This helper implements PQN without TIC so callers can choose
    whether to run TIC first.

    Args:
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

    if (ref_samples is None) != (ref_label is None):
        raise ValueError(
            "PQN reference metadata and reference label must be provided together."
        )

    # Filter reference samples if metadata provided
    if ref_samples is not None:
        ref_series = ref_samples.to_series()
        ref_sample_indices = ref_series[ref_series == ref_label].index.tolist()
        if not ref_sample_indices:
            raise ValueError(
                f"Reference label '{ref_label}' not found in metadata column."
            )
        table_ref = table.loc[ref_sample_indices]
    else:
        table_ref = table

    # Reference spectrum (per feature)
    ref = table_ref.median(axis=0) if method == "median" else table_ref.mean(axis=0)

    # Avoid division by zero in reference
    ref_safe = ref.replace(0.0, np.nan)

    # Compute quotients per sample
    quotients = table.div(ref_safe, axis=1)

    # Median quotient per sample = dilution factor
    valid_quotients = quotients.notna().any(axis=1)
    factors = pd.Series(np.nan, index=quotients.index)
    factors.loc[valid_quotients] = quotients.loc[valid_quotients].median(
        axis=1, skipna=True
    )

    if factors.isna().any():
        raise ValueError(
            "PQN failed: could not compute dilution factor "
            "(sample may contain only zeros)."
        )

    if (factors <= 0).any():
        raise ValueError("PQN failed: non-positive dilution factor detected.")

    # Normalize samples
    table = table.div(factors, axis=0)

    return table


def resolve_capture_holder(
    capture_holder: CaptureHolder[int] | int | None,
    random: bool,
) -> int | None:
    """
    Resolves a capture-holder seed parameter.

    Args:
        capture_holder (CaptureHolder[int] | int | None):
            Capture holder or explicit seed value.
        random (bool):
            If True, resolve the capture holder to a generated random seed.
            If False, resolve it to None.

    Returns:
        int | None:
            Captured/generated seed, explicit seed value, or None.
    """
    if random:
        return CaptureHolder.get_or_set(
            capture_holder, lambda: secrets.randbelow(2**31)
        )

    return CaptureHolder.get_or_set(capture_holder, lambda: None)


def remove_missing_features(table: pd.DataFrame) -> pd.DataFrame:
    """
    Removes features that have no observed values before missForest imputation.

    Args:
        table (pd.DataFrame): Numeric feature table with missing values encoded
            as NaN.

    Returns:
        pd.DataFrame: Table without features that contain only missing values.

    Warns:
        RachisWarning: If one or more features are removed because they cannot
            be imputed by missForest.
    """
    missing_features = table.columns[table.isna().all(axis=0)]
    if len(missing_features) > 0:
        warnings.warn(
            f"Removed {len(missing_features)} features with no observed values.",
            RachisWarning,
        )
        table = table.drop(columns=missing_features)

    return table


def impute_table(
    table: pd.DataFrame,
    impute: str = "knn",
    knn_neighbors: int = 10,
    mf_ntree: int = 100,
    mf_threads: int = 1,
    mf_random_state: int | None = None,
) -> pd.DataFrame:
    """
    Helper to perform missing-value imputation on a numeric dataframe.

    Supports:
        - "knn": imputeLCMD::impute.wrapper.KNN through Rscript
        - "miss_forest": missForest::missForest through Rscript
        - "qrilc": imputeLCMD::impute.QRILC through Rscript. The table is
          log2-transformed before calling R and transformed back afterward.

    Zero values are treated as missing during imputation, so zeros and NaNs are
    imputed in the same way.

    Args:
        table (pd.DataFrame):
            Numeric data frame (samples x features) to impute; missing values
            represented as zero or NaN.
        impute (str | None):
            Imputation method: "knn", "miss_forest", "qrilc", or None to skip
            imputation.
        knn_neighbors (int):
            Number of neighbors for KNN imputation (used when impute == "knn").
        mf_ntree (int):
            Number of trees for missForest (used when impute == "miss_forest").
        mf_threads (int):
            Number of threads for missForest (used when impute == "miss_forest").
        mf_random_state (int | None):
            Seed for R's random-number generator before missForest runs.

    Returns:
        pd.DataFrame:
            New DataFrame with imputed values. For missForest, features with
            no observed values are removed because they cannot be imputed.

    Raises:
        ValueError:
            If an unknown imputation method is provided.
    """
    table = table.replace(0, np.nan)

    if impute in {"knn", "qrilc"}:
        if impute == "qrilc":
            # QRILC runs on log-scaled data.
            table = np.log2(table)

        # imputeLCMD expects features x samples, unlike the QIIME DataFrame.
        table_imp = run_r_table_script(
            table=table.T,
            script_name="impute_imputelcmd",
            parameters={"method": impute, "knn_neighbors": knn_neighbors},
            package_name="imputeLCMD",
        ).T

        if impute == "qrilc":
            table_imp = np.exp2(table_imp)

    elif impute == "miss_forest":
        table = remove_missing_features(table)

        table_imp = run_r_table_script(
            table=table,
            script_name="impute_missforest",
            parameters={
                "ntree": mf_ntree,
                "threads": mf_threads,
                "random_state": mf_random_state,
            },
            package_name="missForest",
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

    Args:
        table (pd.DataFrame):
            Feature table (samples x features).
        transform (str):
            Transformation type: "log", "log10", "sqrt".
        pseudocount (float | None):
            Offset added before log/log10 transform only when the table contains
            zero values. If None and zero values are present, uses half the
            minimum non-zero value in the table. Must be > 0 when provided.

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
        has_zero = (table == 0).any().any()
        if has_zero and pseudocount is None:
            min_nonzero = table.where(table > 0).min().min()
            if pd.isna(min_nonzero):
                raise ValueError(
                    "Log transformation requires at least one positive value "
                    "to infer a pseudocount."
                )
            pseudocount = float(min_nonzero) / 2
        elif not has_zero:
            if pseudocount is not None:
                warnings.warn(
                    "Pseudocount was provided, but no zero values were found; "
                    "the pseudocount was not applied.",
                    RachisWarning,
                )
            pseudocount = 0.0
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
        - "center": mean-center each feature
        - "auto": mean-center and divide by standard deviation (unit variance)
        - "pareto": mean-center and divide by square root of standard deviation
        - "range": mean-center and divide by max-min range

    Args:
        table (pd.DataFrame):
            Feature table (samples x features).
        scale (str | None):
            Scaling method: "center", "auto", "pareto", or "range".

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
    scale: str | None = None,
    impute: str | None = None,
    knn_neighbors: int = 5,
    mf_ntree: int = 100,
    mf_threads: int = 1,
    mf_random_state: CaptureHolder[int] = None,
) -> pd.DataFrame:
    """
    Applies metabolomics-friendly preprocessing to a feature table.

    Steps (in order):
        0) Optional imputation (knn, miss_forest, or qrilc). QRILC imputes in log2
           space and returns the result to the original scale.
        1) Sample normalization: tic, pqn, or tic_pqn
        2) Transform (log, log10, sqrt, or None)
        3) Scaling (per feature / column): center, auto, pareto, or range

    Args:
        table (pd.DataFrame):
            Feature table (samples x features). Values should be numeric.
        sample_normalization (str):
            "tic", "pqn" (Probabilistic Quotient Normalization), "tic_pqn",
            or None. The original PQN paper includes TIC as part of PQN;
            use "tic_pqn" for that workflow, or "pqn" to run PQN without TIC.
        pqn_method (str):
            How to build the PQN reference spectrum: "median" or "mean".
        pqn_ref_samples (rachis.CategoricalMetadataColumn):
            Categorical metadata column used to select samples for building the PQN
            reference spectrum.
        pqn_ref_label (str):
            Label value in `pqn_ref_samples` indicating which samples to use for the
            PQN reference. Required when `pqn_ref_samples` is provided.
        transform (str):
            Transformation to apply: "log", "log10", "sqrt", or None.
        pseudocount (float | None):
            Offset added before log/log10 transform only when the table contains
            zero values. If None and zero values are present, uses half the
            minimum non-zero value in the table. Must be > 0 when provided.
        scale (str):
            Scaling method: None, "center" (mean-center only), "auto" (unit
            variance), "pareto" (divide by sqrt(std)), or "range" (divide by
            max-min). Scaling methods are applied after mean-centering.
        impute (str | None):
            Missing-value imputation method. Supported: "knn", "miss_forest",
            "qrilc",
            or None. Zero values and NaNs are treated as missing values. QRILC
            uses log2-transformed values during imputation and returns values on
            the original scale.
        knn_neighbors (int):
            Number of neighbors for KNN imputation.
        mf_ntree (int):
            Number of trees used by missForest.
        mf_threads (int):
            Number of threads used by missForest.
        mf_random_state (CaptureHolder[int]):
            Random seed for missForest reproducibility.
            If missForest imputation is used and this is omitted, a random seed is
            generated and captured in provenance.

    Returns:
        pd.DataFrame:
            Transformed table with the same shape and index/columns as input.

    Raises:
        ValueError:
            If inputs are invalid or options are unknown.
    """
    mf_random_state = resolve_capture_holder(
        mf_random_state, random=impute == "miss_forest"
    )

    # 1: Imputation
    if impute is not None:
        table = impute_table(
            table,
            impute=impute,
            knn_neighbors=knn_neighbors,
            mf_ntree=mf_ntree,
            mf_threads=mf_threads,
            mf_random_state=mf_random_state,
        )

    # 2: Sample normalization
    if sample_normalization in ("tic", "tic_pqn"):
        table = normalize_tic(table)

    if sample_normalization in ("pqn", "tic_pqn"):
        table = normalize_pqn(
            table=table,
            method=pqn_method,
            ref_samples=pqn_ref_samples,
            ref_label=pqn_ref_label,
        )

    # 3: Transformation
    if transform is not None:
        table = transform_table(
            table,
            transform=transform,
            pseudocount=pseudocount,
        )

    # 4: Scale per feature (column)
    if scale is not None:
        table = scale_table(table, scale=scale)

    return table
