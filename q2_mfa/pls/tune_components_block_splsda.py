# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import secrets

import numpy as np
import pandas as pd
from q2templates.reports import matryoshka_template
from rachis import CategoricalMetadataColumn, Metadata
from rachis.plugin import CaptureHolder
from rpy2.robjects import r

from q2_mfa.pls.descriptions import jsonl_descriptions, report_descriptions
from q2_mfa.pls.types._format import PLSTuneComponentsDirFmt
from q2_mfa.pls.utils import (
    _build_bpparam,
    _r_vote_error_rate_to_dataframe,
    _resolve_design,
    _to_r_inputs,
)


def _tune_components_block_splsda(
    tables: pd.DataFrame,
    y: CategoricalMetadataColumn,
    design_matrix: Metadata = None,
    design_weight: float | None = None,
    ncomp: int = 2,
    scale: bool = True,
    tol: float = 1e-6,
    max_iter: int = 100,
    near_zero_var: bool = False,
    validation: str = "Mfold",
    folds: int = 10,
    nrepeat: int = 3,
    signif_threshold: float = 0.01,
    seed: CaptureHolder[int] = None,
    threads: int = 1,
) -> PLSTuneComponentsDirFmt:
    """
    Tunes block sPLS-DA (DIABLO) component counts with mixOmics.

    Aligns the input blocks and categorical response, fits a dense block
    sPLS-DA (DIABLO) model as documented for component tuning in the mixOmics
    vignette,
    evaluates its component counts by cross-validation, and serializes
    weighted- and majority-vote diagnostics as a component-tuning directory
    format.

    Args:
        tables (ResultCollection): Named feature-table artifacts used for block
            sPLS-DA (DIABLO).
        y (CategoricalMetadataColumn): Categorical response labels for the
            samples.
        design_matrix (Metadata, optional): Explicit block-relationship design
            matrix.
        design_weight (float, optional): Shared off-diagonal relationship
            weight used when no design matrix is supplied.
        ncomp (int): Number of latent components to fit and evaluate.
        scale (bool): Whether to standardize features within every block.
        tol (float): Convergence tolerance for the iterative model fit.
        max_iter (int): Maximum iterations for each model fit.
        near_zero_var (bool): Whether to remove zero or near-zero variance
            predictors before fitting.
        validation (str): Cross-validation strategy accepted by mixOmics.
        folds (int): Number of folds used for M-fold cross-validation.
        nrepeat (int): Number of cross-validation repetitions.
        signif_threshold (float): Minimum error-rate improvement required to
            retain an additional component.
        seed (CaptureHolder[int], optional): Random seed holder used for
            reproducible cross-validation.
        threads (int): Number of BiocParallel workers to use.

    Returns:
        PLSTuneComponentsDirFmt: Serialized weighted- and majority-vote error
            rates and component-choice matrices.
    """
    r("suppressPackageStartupMessages(library(mixOmics))")
    target = y.to_series()
    blocks = tables.collection
    design = _resolve_design(design_matrix, design_weight, list(blocks))
    resolved_seed = CaptureHolder.get_or_set(seed, lambda: secrets.randbelow(2**31))
    bpparam = _build_bpparam(threads, resolved_seed)
    r_blocks, r_target, r_design = _to_r_inputs(blocks, target, design)

    # Fit the dense model, as documented for component tuning in the mixOmics
    # block sPLS-DA (DIABLO) vignette.
    perf_model = r["block.plsda"](
        r_blocks,
        r_target,
        **{
            "ncomp": ncomp,
            "design": r_design,
            "scale": scale,
            "tol": tol,
            "max.iter": max_iter,
            "near.zero.var": near_zero_var,
        },
    )
    perf_result = r["perf"](
        perf_model,
        **{
            "dist": "all",
            "validation": validation,
            "folds": folds,
            "nrepeat": nrepeat,
            "signif.threshold": signif_threshold,
            "BPPARAM": bpparam,
            "seed": resolved_seed,
            "progressBar": False,
        },
    )
    weighted_choice_matrix = _r_choice_matrix_to_dataframe(perf_result, "WeightedVote")
    majority_choice_matrix = _r_choice_matrix_to_dataframe(perf_result, "MajorityVote")
    weighted_error_rate = _r_vote_error_rate_to_dataframe(perf_result, "WeightedVote")
    majority_error_rate = _r_vote_error_rate_to_dataframe(perf_result, "MajorityVote")
    _print_component_choice(weighted_choice_matrix, "WeightedVote")
    _print_component_choice(majority_choice_matrix, "MajorityVote")
    return _serialize_tune_components(
        weighted_error_rate,
        majority_error_rate,
        weighted_choice_matrix,
        majority_choice_matrix,
    )


def tune_components_block_splsda(
    ctx,
    tables,
    y,
    design_matrix=None,
    design_weight=None,
    ncomp=2,
    scale=True,
    tol=1e-6,
    max_iter=100,
    near_zero_var=False,
    validation="Mfold",
    folds=10,
    nrepeat=3,
    signif_threshold=0.01,
    seed=None,
    threads=1,
):
    """
    Tunes block sPLS-DA (DIABLO) components and creates diagnostic
    visualizations.

    Runs the component-tuning method and delegates diagnostic visualization
    generation to the component-tuning visualization pipeline.

    Args:
        ctx (Context): Pipeline execution context used to retrieve actions and
            create reports.
        tables (ResultCollection): Named feature-table artifacts used for block
            sPLS-DA (DIABLO).
        y (CategoricalMetadataColumn): Categorical response labels for the
            samples.
        design_matrix (Metadata, optional): Explicit block-relationship design
            matrix.
        design_weight (float, optional): Shared off-diagonal relationship
            weight used when no design matrix is supplied.
        ncomp (int): Number of latent components to fit and evaluate.
        scale (bool): Whether to standardize features within every block.
        tol (float): Convergence tolerance for the iterative model fit.
        max_iter (int): Maximum iterations for each model fit.
        near_zero_var (bool): Whether to remove zero or near-zero variance
            predictors before fitting.
        validation (str): Cross-validation strategy accepted by mixOmics.
        folds (int): Number of folds used for M-fold cross-validation.
        nrepeat (int): Number of cross-validation repetitions.
        signif_threshold (float): Minimum error-rate improvement required to
            retain an additional component.
        seed (int, optional): Random seed used for reproducible
            cross-validation.
        threads (int): Number of BiocParallel workers to use.

    Returns:
        tuple[Artifact, Visualization]: The component-tuning artifact and a
            report containing error-rate plots and choice-matrix tables.
    """
    tune_components_action = ctx.get_action("mfa", "_tune_components_block_splsda")
    align_samples = ctx.get_action("mfa", "_align_samples_metadata")
    visualisation = ctx.get_action("mfa", "_tune_components_block_visualisation")

    aligned_tables, aligned_metadata = align_samples(
        tables=tables,
        metadata_column=y,
    )
    (tune_components,) = tune_components_action(
        tables=aligned_tables,
        y=aligned_metadata.view(Metadata).get_column(y.name),
        design_matrix=design_matrix,
        design_weight=design_weight,
        ncomp=ncomp,
        scale=scale,
        tol=tol,
        max_iter=max_iter,
        near_zero_var=near_zero_var,
        validation=validation,
        folds=folds,
        nrepeat=nrepeat,
        signif_threshold=signif_threshold,
        seed=seed,
        threads=threads,
    )
    (report,) = visualisation(tune_components=tune_components)
    return tune_components, report


def _tune_components_block_visualisation(ctx, tune_components):
    """
    Creates diagnostic visualizations for block sPLS-DA component tuning.

    Plots overall weighted- and majority-vote error rates, tabulates both
    component-choice matrices, and combines the resulting visualizations into
    a report with Markdown guidance describing each diagnostic and the
    significance-based component-selection rule.

    Args:
        ctx (Context): Pipeline execution context used to retrieve actions and
            create reports.
        tune_components (Artifact): Component-tuning metrics artifact.

    Returns:
        Visualization: Report containing error-rate plots and component-choice
            matrices.
    """
    lineplot = ctx.get_action("vizard", "lineplot")
    tabulate = ctx.get_action("metadata", "tabulate")
    tuning_data = tune_components.view(PLSTuneComponentsDirFmt)

    weighted_error_rates = _error_rate_metadata(
        tuning_data.error_rate_weighted.view(Metadata)
    )
    majority_error_rates = _error_rate_metadata(
        tuning_data.error_rate_majority.view(Metadata)
    )
    (weighted_error_rate_plot,) = lineplot(
        metadata=weighted_error_rates,
        x_measure="component",
        y_measure="mean",
        replicate_method="none",
        group_by="error_rate",
        title="sPLS-DA weighted-vote error rates",
    )
    (majority_error_rate_plot,) = lineplot(
        metadata=majority_error_rates,
        x_measure="component",
        y_measure="mean",
        replicate_method="none",
        group_by="error_rate",
        title="sPLS-DA majority-vote error rates",
    )
    (weighted_choice_matrix,) = tabulate(
        input=tuning_data.choice_matrix_weighted.view(Metadata)
    )
    (majority_choice_matrix,) = tabulate(
        input=tuning_data.choice_matrix_majority.view(Metadata)
    )

    error_rate_report = ctx.make_report(
        matryoshka_template,
        {
            "Weighted vote": weighted_error_rate_plot,
            "Majority vote": majority_error_rate_plot,
        },
        descriptions={
            "Weighted vote": report_descriptions["error_rate_weighted"],
            "Majority vote": report_descriptions["error_rate_majority"],
        },
    )
    choice_matrix_report = ctx.make_report(
        matryoshka_template,
        {
            "Weighted vote": weighted_choice_matrix,
            "Majority vote": majority_choice_matrix,
        },
        descriptions={
            "Weighted vote": report_descriptions["choice_matrix_weighted"],
            "Majority vote": report_descriptions["choice_matrix_majority"],
        },
    )
    report = ctx.make_report(
        matryoshka_template,
        {
            "Error rates": error_rate_report,
            "Component choices": choice_matrix_report,
        },
    )
    return report


def _error_rate_metadata(error_rates: Metadata) -> Metadata:
    """
    Filters component error rates to overall classifications and labels them.

    Retains only Overall.ER and Overall.BER rows and adds an ``error_rate``
    column that combines the prediction distance with the error-rate class for
    plotting.

    Args:
        error_rates (Metadata): Error-rate metadata containing distance and
            class columns.

    Returns:
        Metadata: Overall error-rate records with an ``error_rate`` plot-group
            column.
    """
    error_rates = error_rates.to_dataframe().copy()
    error_rates = error_rates.loc[
        error_rates["class"].isin({"Overall.BER", "Overall.ER"})
    ]
    error_rates["error_rate"] = (
        error_rates["distance"].astype(str) + ": " + error_rates["class"].astype(str)
    )
    return Metadata(error_rates)


def _r_choice_matrix_to_dataframe(perf_result, vote: str) -> pd.DataFrame:
    """
    Converts one mixOmics component-choice matrix to a DataFrame.

    Extracts the requested vote matrix from a mixOmics ``perf()`` result and
    preserves its distance-measure columns and overall-error-rate row labels.

    Args:
        perf_result (rpy2.robjects.vectors.ListVector): Result returned by
            mixOmics ``perf()``.
        vote (str): Vote type to extract, such as ``WeightedVote`` or
            ``MajorityVote``.

    Returns:
        pd.DataFrame: Component-choice matrix.

    Raises:
        ValueError: If mixOmics did not provide a component-choice matrix for
            the requested vote.
    """
    matrix = perf_result.rx2("choice.ncomp").rx2(vote)
    if type(matrix).__name__ == "NULLType":
        raise ValueError(
            f"mixOmics did not provide a component-choice matrix for {vote}."
        )
    values = np.asarray(matrix)
    return pd.DataFrame(
        values,
        index=pd.Index(r["rownames"](matrix), name="id"),
        columns=r["colnames"](matrix),
    ).astype("Int64")


def _print_component_choice(choice_table: pd.DataFrame, vote: str) -> None:
    """
    Prints a component-choice matrix.

    Formats the table for user-visible action output.

    Args:
        choice_table (pd.DataFrame): Component-choice matrix to display.
        vote (str): Vote type represented by the matrix.

    Returns:
        None: This function writes the formatted message to standard output.
    """
    print(f"{vote} component-choice matrix:\n", flush=True)
    print(f"{choice_table.to_string()}\n", flush=True)


def _serialize_tune_components(
    error_rate_weighted: pd.DataFrame,
    error_rate_majority: pd.DataFrame,
    choice_matrix_weighted: pd.DataFrame,
    choice_matrix_majority: pd.DataFrame,
) -> PLSTuneComponentsDirFmt:
    """
    Serializes component-tuning tables as Table JSONL files.

    Adds stable IDs to error-rate rows so the metadata transformer can use
    them as metadata IDs, preserves choice-matrix IDs, attaches JSONL
    descriptions, and writes all four component-tuning tables into the
    directory format.

    Args:
        error_rate_weighted (pd.DataFrame): Weighted-vote cross-validated
            error rates.
        error_rate_majority (pd.DataFrame): Majority-vote cross-validated
            error rates.
        choice_matrix_weighted (pd.DataFrame): Weighted-vote selected-component
            matrix.
        choice_matrix_majority (pd.DataFrame): Majority-vote selected-component
            matrix.

    Returns:
        PLSTuneComponentsDirFmt: Directory format containing the four JSONL
            component-tuning tables.
    """
    directory_format = PLSTuneComponentsDirFmt()
    error_rate_weighted = error_rate_weighted.copy()
    error_rate_majority = error_rate_majority.copy()
    choice_matrix_weighted = choice_matrix_weighted.reset_index()
    choice_matrix_majority = choice_matrix_majority.reset_index()

    error_rate_weighted.insert(
        0, "id", [f"row{index}" for index in range(1, len(error_rate_weighted) + 1)]
    )
    error_rate_majority.insert(
        0, "id", [f"row{index}" for index in range(1, len(error_rate_majority) + 1)]
    )

    error_rate_weighted.attrs["description"] = jsonl_descriptions["error_rate_weighted"]
    error_rate_majority.attrs["description"] = jsonl_descriptions["error_rate_majority"]
    choice_matrix_weighted.attrs["description"] = jsonl_descriptions[
        "choice_matrix_weighted"
    ]
    choice_matrix_majority.attrs["description"] = jsonl_descriptions[
        "choice_matrix_majority"
    ]

    directory_format.error_rate_weighted.write_data(error_rate_weighted, pd.DataFrame)
    directory_format.error_rate_majority.write_data(error_rate_majority, pd.DataFrame)
    directory_format.choice_matrix_weighted.write_data(
        choice_matrix_weighted, pd.DataFrame
    )
    directory_format.choice_matrix_majority.write_data(
        choice_matrix_majority, pd.DataFrame
    )
    return directory_format
