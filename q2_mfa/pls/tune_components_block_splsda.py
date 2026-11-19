# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import secrets

import pandas as pd
from q2templates.reports import matryoshka_template
from rachis import CategoricalMetadataColumn, Metadata
from rachis.plugin import CaptureHolder
from rpy2.robjects.packages import importr

from q2_mfa.pls.descriptions import jsonl_descriptions, report_descriptions
from q2_mfa.pls.types._format import PLSComponentSelectionDirFmt
from q2_mfa.pls.utils import (
    _align_samples,
    _build_bpparam,
    _call_without_messages,
    _print_component_choice,
    _r_choice_matrix_to_df,
    _r_perf_error_rate_to_df,
    _resolve_design,
    _to_r_inputs,
)

mixomics = importr("mixOmics")


def _tune_components_block_splsda(
    tables: pd.DataFrame,
    y: CategoricalMetadataColumn,
    design_matrix: pd.DataFrame = None,
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
) -> PLSComponentSelectionDirFmt:
    """
    Tunes block sPLS-DA (DIABLO) component counts with mixOmics.

    Aligns the input blocks and categorical response, fits a dense block
    sPLS-DA (DIABLO) model as documented for component tuning in the mixOmics
    vignette, evaluates its component counts by cross-validation, and serializes
    weighted- and majority-vote diagnostics as a component-tuning directory
    format.

    Returns:
        PLSComponentSelectionDirFmt: Serialized weighted- and majority-vote error
            rates and component-choice matrices.
    """
    tables, target = _align_samples(tables, y.to_series())
    design = _resolve_design(design_matrix, design_weight, tables.keys())
    resolved_seed = CaptureHolder.get_or_set(seed, lambda: secrets.randbelow(2**31))
    bpparam = _build_bpparam(threads)
    r_blocks, r_target, r_design, _, _ = _to_r_inputs(tables, target, design)

    # Fit the dense model, as documented for component tuning in the mixOmics
    # block sPLS-DA (DIABLO) vignette.
    perf_model = _call_without_messages(
        mixomics.block_plsda,
        X=r_blocks,
        Y=r_target,
        ncomp=ncomp,
        design=r_design,
        scale=scale,
        tol=tol,
        **{
            "max.iter": max_iter,
            "near.zero.var": near_zero_var,
        },
    )
    perf_result = mixomics.perf(
        perf_model,
        dist="all",
        validation=validation,
        folds=folds,
        nrepeat=nrepeat,
        BPPARAM=bpparam,
        seed=resolved_seed,
        **{"signif.threshold": signif_threshold},
    )
    weighted_choice_matrix = _r_choice_matrix_to_df(perf_result, "WeightedVote")
    majority_choice_matrix = _r_choice_matrix_to_df(perf_result, "MajorityVote")
    weighted_error_rate = _r_perf_error_rate_to_df(perf_result, "WeightedVote")
    majority_error_rate = _r_perf_error_rate_to_df(perf_result, "MajorityVote")
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

    Returns:
        tuple[Artifact, Visualization]: The component-tuning artifact and a
            report containing error-rate plots and choice-matrix tables.
    """
    tune_components_action = ctx.get_action("mfa", "_tune_components_block_splsda")
    visualisation = ctx.get_action("mfa", "_tune_components_block_visualisation")

    (tune_components,) = tune_components_action(
        tables=tables,
        y=y,
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
    Creates diagnostic visualizations for sPLS-DA component tuning.

    Plots overall error rates and tabulates component choices for single-table
    or block results. Block results are grouped by voting scheme. The report
    includes guidance describing the diagnostics and component-selection rule.

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
    tuning_data = tune_components.view(PLSComponentSelectionDirFmt)

    error_rates = {
        str(path): metadata
        for path, metadata in tuning_data.error_rate.iter_views(Metadata)
    }
    choices = {
        str(path): metadata
        for path, metadata in tuning_data.choice_matrix.iter_views(Metadata)
    }
    plots = {}
    tables = {}
    error_descriptions = {}
    choice_descriptions = {}
    for suffix, label, title in (
        ("", "Single table", "sPLS-DA error rates"),
        ("_weighted", "Weighted vote", "sPLS-DA weighted-vote error rates"),
        ("_majority", "Majority vote", "sPLS-DA majority-vote error rates"),
    ):
        error_key = f"error_rate{suffix}"
        choice_key = f"choice_matrix{suffix}"
        if f"{error_key}.jsonl" not in error_rates:
            continue
        (plots[label],) = lineplot(
            metadata=_error_rate_metadata(error_rates[f"{error_key}.jsonl"]),
            x_measure="component",
            y_measure="mean",
            replicate_method="none",
            group_by="error_rate",
            color_palette="category20",
            title=title,
        )
        (tables[label],) = tabulate(input=choices[f"{choice_key}.jsonl"])
        error_descriptions[label] = report_descriptions[error_key]
        choice_descriptions[label] = report_descriptions[choice_key]

    if len(plots) == 1:
        label = next(iter(plots))
        return ctx.make_report(
            matryoshka_template,
            {"Error rates": plots[label], "Component choices": tables[label]},
            descriptions={
                "Error rates": error_descriptions[label],
                "Component choices": choice_descriptions[label],
            },
        )

    error_rate_report = ctx.make_report(
        matryoshka_template, plots, descriptions=error_descriptions
    )
    choice_matrix_report = ctx.make_report(
        matryoshka_template, tables, descriptions=choice_descriptions
    )
    return ctx.make_report(
        matryoshka_template,
        {
            "Error rates": error_rate_report,
            "Component choices": choice_matrix_report,
        },
    )


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


def _serialize_tune_components(
    error_rate_weighted: pd.DataFrame,
    error_rate_majority: pd.DataFrame,
    choice_matrix_weighted: pd.DataFrame,
    choice_matrix_majority: pd.DataFrame,
) -> PLSComponentSelectionDirFmt:
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
        PLSComponentSelectionDirFmt: Directory format containing the four JSONL
            component-tuning tables.
    """
    directory_format = PLSComponentSelectionDirFmt()
    error_rate_weighted = error_rate_weighted.copy()
    error_rate_majority = error_rate_majority.copy()
    choice_matrix_weighted = choice_matrix_weighted.reset_index()
    choice_matrix_majority = choice_matrix_majority.reset_index()

    error_rate_weighted.insert(
        0, "id", [str(index) for index in range(1, len(error_rate_weighted) + 1)]
    )
    error_rate_majority.insert(
        0, "id", [str(index) for index in range(1, len(error_rate_majority) + 1)]
    )

    error_rate_weighted.attrs["description"] = jsonl_descriptions["error_rate_weighted"]
    error_rate_majority.attrs["description"] = jsonl_descriptions["error_rate_majority"]
    choice_matrix_weighted.attrs["description"] = jsonl_descriptions[
        "choice_matrix_weighted"
    ]
    choice_matrix_majority.attrs["description"] = jsonl_descriptions[
        "choice_matrix_majority"
    ]

    directory_format.error_rate.write_data(
        error_rate_weighted, pd.DataFrame, vote="weighted"
    )
    directory_format.error_rate.write_data(
        error_rate_majority, pd.DataFrame, vote="majority"
    )
    directory_format.choice_matrix.write_data(
        choice_matrix_weighted, pd.DataFrame, vote="weighted"
    )
    directory_format.choice_matrix.write_data(
        choice_matrix_majority, pd.DataFrame, vote="majority"
    )
    return directory_format
