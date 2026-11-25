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
from rpy2.robjects import ListVector
from rpy2.robjects.packages import importr

from q2_mfa.pls.descriptions import fit_descriptions, tune_components_descriptions
from q2_mfa.pls.types._format import (
    PLSComponentSelectionDirFmt,
    PLSFeatureSelectionDirFmt,
    PLSFitDirFmt,
)
from q2_mfa.pls.utils import (
    _align_samples,
    _build_bpparam,
    _call_without_messages,
    _parse_keepx_collection,
    _r_matrix_to_dataframe,
    _r_perf_error_rate_to_df,
    _resolve_design,
    _resolve_ncomp,
    _to_r_inputs,
)

base = importr("base")
mixomics = importr("mixOmics")

descriptions = {**fit_descriptions, **tune_components_descriptions}


def _fit_block_splsda(
    tables: pd.DataFrame,
    y: CategoricalMetadataColumn,
    design_matrix: pd.DataFrame = None,
    design_weight: float | None = None,
    tuned_components: PLSComponentSelectionDirFmt = None,
    ncomp: CaptureHolder[int] = None,
    keepx: CaptureHolder[dict] = None,
    feature_selection: PLSFeatureSelectionDirFmt = None,
    scale: bool = True,
    tol: float = 1e-6,
    max_iter: int = 100,
    near_zero_var: bool = False,
    validation: str = "Mfold",
    folds: int = 10,
    nrepeat: int = 3,
    signif_threshold: float = 0.01,
    distance: str = "max.dist",
    measure: str = "BER",
    weighted: bool = True,
    seed: CaptureHolder[int] = None,
    threads: int = 1,
) -> PLSFitDirFmt:
    """Fits DIABLO with supplied or previously selected components and keepX.

    ``ncomp`` and ``tuned_components`` are mutually exclusive. Likewise,
    provide manual block-keyed ``keepx`` strings or a feature-selection
    artifact, but not both. The input tables and response are aligned to
    shared sample IDs before fitting.
    """
    blocks, target = _align_samples(dict(tables.items()), y.to_series())
    design = _resolve_design(design_matrix, design_weight, list(blocks))
    selected_ncomp = _resolve_ncomp(
        ncomp, tuned_components, weighted, measure, distance
    )
    selected_keepx = _resolve_fit_keepx(
        keepx, feature_selection, list(blocks), selected_ncomp
    )
    resolved_seed = CaptureHolder.get_or_set(seed, lambda: secrets.randbelow(2**31))
    bpparam = _build_bpparam(threads)

    r_blocks, r_target, r_design, r_keepx, _ = _to_r_inputs(
        blocks,
        target,
        design,
        keep_x=selected_keepx,
    )
    model = _call_without_messages(
        mixomics.block_splsda,
        r_blocks,
        r_target,
        ncomp=selected_ncomp,
        keepX=r_keepx,
        design=r_design,
        scale=scale,
        tol=tol,
        **{
            "max.iter": max_iter,
            "near.zero.var": near_zero_var,
        },
    )
    perf_result = mixomics.perf(
        model,
        dist="all",
        validation=validation,
        folds=folds,
        nrepeat=nrepeat,
        auc=True,
        BPPARAM=bpparam,
        seed=resolved_seed,
        **{"signif.threshold": signif_threshold},
    )
    return _serialize_fit(model, perf_result)


def fit_block_splsda(
    ctx,
    tables,
    y,
    design_matrix=None,
    design_weight=None,
    tuned_components=None,
    ncomp=None,
    keepx=None,
    feature_selection=None,
    scale=True,
    tol=1e-6,
    max_iter=100,
    near_zero_var=False,
    validation="Mfold",
    folds=10,
    nrepeat=3,
    signif_threshold=0.01,
    distance="max.dist",
    measure="BER",
    weighted=True,
    seed=None,
    threads=1,
):
    """Fits DIABLO and reports sample scores and feature correlations.

    Returns:
        tuple[Artifact, Visualization]: Fitted model tables and an interactive
            report corresponding to mixOmics plotIndiv() and plotVar().
    """
    fit_kwargs = locals().copy()
    fit_kwargs.pop("ctx")

    fit = ctx.get_action("mfa", "_fit_block_splsda")
    visualise = ctx.get_action("mfa", "_fit_block_splsda_visualiser")
    (pls_results,) = fit(**fit_kwargs)
    (visualization,) = visualise(pls_results=pls_results, y=y)
    return pls_results, visualization


def _fit_block_splsda_visualiser(ctx, pls_results, y):
    """Plots fitted sample scores and selected feature-component correlations.

    Uses per-block and consensus sample scores for plotIndiv()-style plots,
    and feature-component correlations for a plotVar()-style correlation
    scatterplot. All fitted components are selectable in vizard; components
    one and two are initially shown. Single-component fits use boxplots
    grouped by target class for sample scores and by block for feature
    correlations. The report also includes the complete fitted tables.
    """
    boxplot = ctx.get_action("vizard", "boxplot")
    scatterplot = ctx.get_action("vizard", "scatterplot_2d")
    correlation_plot = ctx.get_action("vizard", "scatterplot_correlation")
    tabulate = ctx.get_action("metadata", "tabulate")

    fitted = pls_results.view(PLSFitDirFmt)
    block_names = fitted.weights.view(pd.DataFrame)["block"].unique()
    score_table = fitted.scores.view(pd.DataFrame)
    feature_metadata = fitted.feature_component_correlation.view(Metadata)

    axes = [column for column in score_table.columns if column.startswith("comp")]
    single_component = len(axes) == 1
    target = y.to_series()
    score_plots = {}
    for group, names in (
        ("block", block_names),
        ("method", ("average", "weighted_average")),
    ):
        for name in names:
            plot_scores = score_table.loc[
                score_table["block"] == name, ["sample", *axes]
            ].set_index("sample")
            plot_scores["Target"] = target
            metadata = Metadata(plot_scores.rename_axis("sample-id"))
            title = f"DIABLO sample scores: {name}"
            if single_component:
                (plot,) = boxplot(
                    metadata=metadata,
                    distribution_measure=axes[0],
                    group_by="Target",
                    box_orientation="vertical",
                    title=title,
                )
            else:
                (plot,) = scatterplot(
                    metadata=metadata,
                    x_measure=axes[0],
                    y_measure=axes[1],
                    color_by="Target",
                    title=title,
                )
            score_plots[f"{group}: {name}"] = plot

    if single_component:
        (feature_plot,) = boxplot(
            metadata=feature_metadata,
            distribution_measure=axes[0],
            group_by="block",
            box_orientation="vertical",
            title="DIABLO feature-component correlations",
        )
    else:
        (feature_plot,) = correlation_plot(
            metadata=feature_metadata,
            x_measure=axes[0],
            y_measure=axes[1],
            color_by="block",
            title="DIABLO feature-component correlations",
        )

    scores = ctx.make_report(
        matryoshka_template,
        score_plots,
        descriptions={name: descriptions["scores"] for name in score_plots},
    )
    table_labels = {
        "loadings": "Loadings",
        "scores": "Scores",
        "weights": "Block weights",
        "prop_expl_var": "Proportion of explained variance",
        "ave": "Average variance explained",
        "feature_stability": "Feature stability",
        "auc": "AUC",
        "error_rate_weighted": "Weighted-vote error rates",
        "error_rate_majority": "Majority-vote error rates",
        "feature_component_correlation": "Feature-component correlations",
        "feature_similarity": "Feature similarities",
    }
    tables = {}
    table_descriptions = {}
    for name, label in table_labels.items():
        (tables[label],) = tabulate(input=getattr(fitted, name).view(Metadata))
        table_descriptions[label] = descriptions[name]
    table_report = ctx.make_report(
        matryoshka_template, tables, descriptions=table_descriptions
    )
    return ctx.make_report(
        matryoshka_template,
        {
            "Sample scores": scores,
            "Feature correlations": feature_plot,
            "Tables": table_report,
        },
        descriptions={
            "Feature correlations": fit_descriptions["feature_component_correlation"],
        },
    )


def _serialize_fit(model, perf_result) -> PLSFitDirFmt:
    """Serializes final DIABLO model and performance outputs to JSONL files."""
    loadings = _r_named_frames(model.rx2("loadings"))
    variates = _r_named_frames(model.rx2("variates"))
    weights = _r_matrix_to_dataframe(base.as_matrix(model.rx2("weights")))
    # mixOmics sanitizes weight row names (e.g. z-lipid becomes z.lipid).
    # model$X preserves the original names in weight row order.
    weights.index = pd.Index([str(name) for name in model.rx2("X").names], name="block")
    directory_format = PLSFitDirFmt()
    _write_fit_table(
        output_file=directory_format.loadings,
        table=_block_matrices_to_dataframe(loadings, identifier="feature"),
    )
    _write_fit_table(
        output_file=directory_format.scores,
        table=_block_matrices_to_dataframe(
            variates, _consensus_variates(variates, weights), identifier="sample"
        ),
    )
    _write_fit_table(
        output_file=directory_format.weights,
        table=weights.reset_index(),
    )
    _write_fit_table(
        output_file=directory_format.prop_expl_var,
        table=_r_prop_expl_var_to_dataframe(model.rx2("prop_expl_var")),
    )
    _write_fit_table(
        output_file=directory_format.ave,
        table=_r_ave_to_dataframe(model.rx2("AVE")),
    )
    _write_fit_table(
        output_file=directory_format.feature_stability,
        table=_r_feature_stability_to_dataframe(
            perf_result.rx2("features").rx2("stable")
        ),
    )
    _write_fit_table(
        output_file=directory_format.auc,
        table=_r_auc_to_dataframe(perf_result.rx2("auc")),
    )
    _write_fit_table(
        output_file=directory_format.error_rate_weighted,
        table=_r_perf_error_rate_to_df(perf_result, "WeightedVote"),
    )
    _write_fit_table(
        output_file=directory_format.error_rate_majority,
        table=_r_perf_error_rate_to_df(perf_result, "MajorityVote"),
    )
    correlations = _feature_component_correlations(
        _r_named_frames(model.rx2("X")), variates, loadings
    )
    _write_fit_table(
        output_file=directory_format.feature_component_correlation,
        table=correlations,
    )
    _write_fit_table(
        output_file=directory_format.feature_similarity,
        table=_feature_similarity_to_dataframe(correlations),
    )
    return directory_format


def _consensus_variates(
    variates: dict[str, pd.DataFrame], weights: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Averages input-block scores using equal or signed model weights.

    The weight index identifies the input blocks, excluding the outcome Y.
    Pandas preserves sample and component labels throughout the calculation.
    Each weighted score is divided by the sum of weights for its component,
    matching mixOmics plotIndiv().
    """
    return {
        "average": (sum(variates[block] for block in weights.index) / len(weights)),
        "weighted_average": (
            sum(
                variates[block].mul(weights.loc[block], axis="columns")
                for block in weights.index
            )
            / weights.sum(axis=0)
        ),
    }


def _write_fit_table(output_file, table: pd.DataFrame) -> None:
    table = table.copy()
    ids = pd.array([str(index) for index in range(1, len(table) + 1)], dtype="string")
    table.insert(0, "id", ids)
    table.attrs["description"] = descriptions[output_file.name]
    output_file.write_data(table, pd.DataFrame)


def _resolve_fit_keepx(
    keepx: CaptureHolder[dict], feature_selection, block_names: list[str], ncomp: int
) -> dict[str, list[int]]:
    """Captures keepX strings supplied manually or derived from an artifact."""
    manual_keepx = keepx is not None and (
        not isinstance(keepx, CaptureHolder) or keepx.is_set
    )
    if manual_keepx == (feature_selection is not None):
        raise ValueError("Provide exactly one of 'keepx' or 'feature-selection'.")
    keepx = CaptureHolder.get_or_set(
        keepx,
        lambda: _keepx_from_feature_selection(feature_selection, ncomp),
    )
    parsed = _parse_keepx_collection(keepx, block_names, ncomp)
    if manual_keepx and any(len(counts) != ncomp for counts in parsed.values()):
        raise ValueError(
            f"Each keepx block must provide exactly {ncomp} feature counts, "
            "one for each fitted component."
        )
    return parsed


def _keepx_from_feature_selection(feature_selection, ncomp: int) -> dict[str, str]:
    """Reads the wide keepX table into block-keyed parameter strings.

    Orders counts by one-based component label and converts them to strings
    so the resolved keepx parameter can be recorded in provenance.
    """
    table = feature_selection.selected_features.view(pd.DataFrame)
    expected_components = list(range(1, ncomp + 1))
    if sorted(table["component"]) != expected_components:
        raise ValueError(
            "The feature-selection artifact must contain one row per fitted "
            "component."
        )
    table = table.drop(columns="id", errors="ignore").set_index("component")
    table = table.loc[expected_components]
    return {block: ",".join(str(value) for value in table[block]) for block in table}


def _block_matrices_to_dataframe(
    *tables: dict[str, pd.DataFrame],
    identifier: str,
) -> pd.DataFrame:
    """Concatenates block matrices used for:
        - loadings
        - scores
        - feature_component_correlation
        - feature_stability

    Args:
        tables (dict[str, pd.DataFrame]): Block-keyed matrices.
        identifier (str): Column name for the matrix row labels, such as
            ``sample`` or ``feature``.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    records = []
    for blocks in tables:
        for block, table in blocks.items():
            working = table.copy()
            working.insert(0, "block", str(block))
            working.insert(0, identifier, working.index.astype(str))
            records.append(working)
    return pd.concat(records, ignore_index=True)


def _r_named_frames(values: ListVector) -> dict[str, pd.DataFrame]:
    """Converts each matrix in a named R list to a wide pandas table. Preserves R
    list names as dictionary keys. Used for:
        - loadings
        - scores
        - feature_component_correlation
    """
    return {
        str(name): _r_matrix_to_dataframe(value)
        for name, value in zip(values.names, values)
    }


def _r_prop_expl_var_to_dataframe(values: ListVector) -> pd.DataFrame:
    return pd.DataFrame(
        {"block": str(block), **dict(zip(vector.names, vector))}
        for block, vector in values.items()
    )


def _r_ave_to_dataframe(values: ListVector) -> pd.DataFrame:
    """Stores per-block AVE and labels overall rows AVE_outer and AVE_inner."""
    vectors = (
        *values.rx2("AVE_X").items(),
        ("AVE_outer", values.rx2("AVE_outer")),
        ("AVE_inner", values.rx2("AVE_inner")),
    )
    return pd.DataFrame(
        {
            "block": str(block),
            **{
                f"comp{component}": float(value)
                for component, value in enumerate(vector, start=1)
            },
        }
        for block, vector in vectors
    )


def _r_feature_stability_to_dataframe(
    selection_frequencies: ListVector,
) -> pd.DataFrame:
    """Builds wide component tables and averages frequencies across CV repeats.

    A feature omitted from a repeat was selected in zero folds, so its
    contribution is zero and the repeat still counts in the denominator.

    Args:
        selection_frequencies (ListVector): R selection frequencies nested by
            repeat, block, and component from ``perf_result$features$stable``.

    Returns:
        pd.DataFrame: One row per block and feature, with mean selection
            frequencies in ``comp1``, ``comp2``, etc. columns.
    """
    repeat_tables = []
    for repeat_values in selection_frequencies:
        blocks = {}
        for block_name, components in repeat_values.items():
            blocks[str(block_name)] = pd.DataFrame(
                {
                    str(component): pd.Series(
                        list(frequencies),
                        index=list(base.names(frequencies)),
                    )
                    for component, frequencies in components.items()
                }
            ).fillna(0)
        repeat_tables.append(_block_matrices_to_dataframe(blocks, identifier="feature"))
    return (
        pd.concat(repeat_tables, ignore_index=True)
        .groupby(["feature", "block"])
        .sum()
        .div(len(selection_frequencies))
        .reset_index()
    )


def _r_auc_to_dataframe(auc: ListVector) -> pd.DataFrame:
    """Converts mixOmics per-component, per-contrast AUC results.

    Args:
        auc (ListVector): R list of per-component AUC matrices from
            ``perf_result$auc``.

    Returns:
        pd.DataFrame: Component, contrast, AUC, and p-value records.
    """
    records = []
    for component, values in enumerate(auc, start=1):
        auc_table = (
            _r_matrix_to_dataframe(values)
            .rename(columns={"AUC": "auc", "p-value": "p_value"})
            .reset_index(names="contrast")
        )
        auc_table.insert(0, "component", component)
        records.append(auc_table)
    return pd.concat(records, ignore_index=True)


def _feature_component_correlations(
    blocks: dict[str, pd.DataFrame],
    variates: dict[str, pd.DataFrame],
    loadings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Returns Pearson correlations with one row per block and selected feature.

    Selects features with a nonzero loading on any fitted component. Each
    correlation uses the samples with both a feature value and a component
    score, as in mixOmics ``circosPlot()``.
    """
    correlations = {}
    for block, values in blocks.items():
        block_loadings = loadings[block]
        selected = block_loadings.index[block_loadings.fillna(0).ne(0).any(axis=1)]
        feature_values = values.loc[:, selected]
        block_variates = variates[block]
        correlations[block] = pd.DataFrame(
            {
                component: feature_values.corrwith(scores)
                for component, scores in block_variates.items()
            }
        )
    return _block_matrices_to_dataframe(correlations, identifier="feature")


def _feature_similarity_to_dataframe(correlations: pd.DataFrame) -> pd.DataFrame:
    """Returns one dot product of component correlations per feature pair."""
    blocks = correlations["block"].to_numpy()
    features = correlations["feature"].to_numpy()
    values = correlations.drop(columns=["feature", "block"]).to_numpy()
    first, second = np.triu_indices(len(correlations), k=1)
    return pd.DataFrame(
        {
            "block_1": blocks[first],
            "feature_1": features[first],
            "block_2": blocks[second],
            "feature_2": features[second],
            "similarity": (values @ values.T)[first, second],
        }
    )
