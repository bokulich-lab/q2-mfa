# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from functools import cache, partial
from importlib.resources import files

import pandas as pd
from q2_types.feature_table import FeatureTable, Unconstrained
from rachis import Artifact, Metadata, ResultCollection
from rpy2.robjects import conversion, default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from q2_mfa.pls.select_features_block_splsda import _serialize_feature_selection
from q2_mfa.pls.types import (
    AssociationMatrix,
    PLSComponentSelection,
    PLSFeatureSelection,
)

base = importr("base")
mixomics = importr("mixOmics")


def nutrimouse_factory(unbalanced: bool = False) -> tuple[ResultCollection, Metadata]:
    """Returns the Nutrimouse gene and lipid tables and their diet metadata.

    The fit example uses unequal diet group sizes to avoid a tie in the
    leading singular values during DIABLO initialization.
    See https://github.com/mixOmics-org/mixOmics/pull/420.
    """
    dataset = mixomics.__rdata__.fetch("nutrimouse")["nutrimouse"]
    sample_ids = list(base.rownames(dataset.rx2("gene")))
    diets = list(base.as_character(dataset.rx2("diet")))
    metadata = pd.DataFrame(
        {"diet": diets}, index=pd.Index(sample_ids, name="sample-id")
    )
    if unbalanced:
        reduced_diets = metadata.loc[metadata["diet"].isin(["lin", "ref", "sun"])]
        metadata = metadata.drop(reduced_diets.groupby("diet").tail(1).index)

    tables = {}
    for block in ("gene", "lipid"):
        data_frame = base.as_data_frame(dataset.rx2(block))
        with localconverter(default_converter + pandas2ri.converter):
            table = conversion.get_conversion().rpy2py(data_frame)
        table.index = pd.Index(table.index.astype(str), name="sample-id")
        table = table.loc[metadata.index]
        tables[block] = Artifact.import_data(FeatureTable[Unconstrained], table)
    return ResultCollection(tables), Metadata(metadata)


def _init_nutrimouse_inputs(use, unbalanced: bool = False):
    """Registers feature tables and the diet response for a Nutrimouse example."""
    inputs = cache(partial(nutrimouse_factory, unbalanced=unbalanced))
    tables = use.init_artifact_collection("nutrimouse_tables", lambda: inputs()[0])
    metadata = use.init_metadata("nutrimouse_metadata", lambda: inputs()[1])
    diet = use.get_metadata_column("diet", "diet", metadata)
    return tables, diet


def nutrimouse_tuned_ncomp_factory():
    """Imports an illustrative artifact choosing two components for weighted BER."""
    fixture = files("q2_mfa.pls.types.tests") / "data" / "pls-tune-components"
    return Artifact.import_data(PLSComponentSelection, str(fixture))


def nutrimouse_association_matrix_factory():
    design = pd.DataFrame(
        [[0.0, 0.9], [0.9, 0.0]],
        index=pd.Index(["gene", "lipid"], name="id"),
        columns=["gene", "lipid"],
    )
    return Artifact.import_data(AssociationMatrix, design)


def nutrimouse_fit_feature_selection_factory():
    selected = pd.DataFrame(
        {"gene": [5, 5], "lipid": [9, 5]},
        index=pd.Index([1, 2], name="component"),
    )
    error_rates = pd.DataFrame(
        columns=["component", "mean", "sd", "keepx_gene", "keepx_lipid"],
        dtype=float,
    )
    return Artifact.import_data(
        PLSFeatureSelection, _serialize_feature_selection(error_rates, selected)
    )


def select_features_block_splsda_nutrimouse(use):
    use.comment(
        "Load the nutrimouse dataset bundled with mixOmics: 40 mice, 120 "
        "gene-expression features, and 21 lipid features. The small grid and three "
        "repeats keep this demonstration short; use more repeats for a thorough "
        "analysis. Uses an illustrative prebuilt component-selection artifact whose "
        "weighted BER choice is two components, and a gene/lipid association "
        "matrix with integration weight 0.9."
    )
    use.comment("DOI: https://doi.org/10.1002/hep.21510")
    tables, diet = _init_nutrimouse_inputs(use)
    tuned_ncomp = use.init_artifact(
        "nutrimouse_tuned_ncomp", nutrimouse_tuned_ncomp_factory
    )
    design_matrix = use.init_artifact(
        "nutrimouse_association_matrix", nutrimouse_association_matrix_factory
    )
    use.action(
        use.UsageAction(plugin_id="mfa", action_id="select_features_block_splsda"),
        use.UsageInputs(
            tables=tables,
            y=diet,
            keepx={"gene": "5,10,15", "lipid": "5,10,15"},
            tuned_ncomp=tuned_ncomp,
            design_matrix=design_matrix,
            folds=3,
            nrepeat=3,
            measure="BER",
            seed=42,
            threads=2,
        ),
        use.UsageOutputNames(
            feature_selection="feature_selection_nutrimouse",
            visualization="feature_selection_report",
        ),
    )


def fit_block_splsda_nutrimouse(use):
    """Fits DIABLO on a Nutrimouse subset with unequal diet group sizes."""
    use.comment(
        "Fit the nutrimouse diet outcome using the gene and lipid blocks bundled "
        "with mixOmics. Use 37 mice (diet group sizes 8, 8, 7, 7, and 7). "
        "Illustrative artifacts supply two components (weighted BER, max.dist) and "
        "keepX counts of 5,5 for gene and 9,5 for lipid. Use a gene/lipid association "
        "weight of 0.9 and evaluate with 7-fold cross-validation and 10 repeats. "
        "Increase the number of folds and repeats for a real analysis."
    )
    use.comment("DOI: https://doi.org/10.1002/hep.21510")
    tables, diet = _init_nutrimouse_inputs(use, unbalanced=True)
    tuned_components = use.init_artifact(
        "nutrimouse_tuned_ncomp", nutrimouse_tuned_ncomp_factory
    )
    feature_selection = use.init_artifact(
        "nutrimouse_fit_feature_selection", nutrimouse_fit_feature_selection_factory
    )
    design_matrix = use.init_artifact(
        "nutrimouse_association_matrix", nutrimouse_association_matrix_factory
    )
    use.action(
        use.UsageAction(plugin_id="mfa", action_id="fit_block_splsda"),
        use.UsageInputs(
            tables=tables,
            y=diet,
            tuned_components=tuned_components,
            feature_selection=feature_selection,
            design_matrix=design_matrix,
            scale=True,
            tol=1e-6,
            max_iter=100,
            near_zero_var=False,
            validation="Mfold",
            folds=7,
            nrepeat=10,
            signif_threshold=0.01,
            distance="max.dist",
            measure="BER",
            weighted=True,
            seed=1,
            threads=1,
        ),
        use.UsageOutputNames(pls_results="fit_nutrimouse"),
    )


def tune_components_block_splsda_nutrimouse(use):
    use.comment(
        "Use gene expression and lipid measurements from the mixOmics Nutrimouse "
        "dataset to classify 40 mice by diet. Evaluate up to two components "
        "using three repeats of three-fold cross-validation and a gene/lipid "
        "association weight of 0.9. The report shows weighted- and majority-vote "
        "error rates and the selected component counts."
        "This example uses few folds and repeats to keep runtime short. For a real "
        "analysis, use more repeats and increase the number of folds For more "
        "information, consult the mixOmics documentation."
    )
    use.comment("DOI: https://doi.org/10.1002/hep.21510")
    tables, diet = _init_nutrimouse_inputs(use)
    design_matrix = use.init_artifact(
        "nutrimouse_association_matrix", nutrimouse_association_matrix_factory
    )
    use.action(
        use.UsageAction(plugin_id="mfa", action_id="tune_components_block_splsda"),
        use.UsageInputs(
            tables=tables,
            y=diet,
            design_matrix=design_matrix,
            ncomp=2,
            validation="Mfold",
            folds=3,
            nrepeat=3,
            seed=42,
            threads=1,
        ),
        use.UsageOutputNames(
            tune_components="tuned_components_nutrimouse",
            visualization="component_tuning_report",
        ),
    )
