# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from importlib.resources import files

import pandas as pd
from q2_types.feature_table import FeatureTable, Unconstrained
from rachis import Artifact, Metadata, ResultCollection
from rpy2.robjects import conversion, default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from q2_mfa.pls.types import AssociationMatrix, PLSComponentSelection

base = importr("base")
mixomics = importr("mixOmics")


def _load_nutrimouse():
    return mixomics.__rdata__.fetch("nutrimouse")["nutrimouse"]


def _nutrimouse_table(block):
    data_frame = base.as_data_frame(_load_nutrimouse().rx2(block))
    with localconverter(default_converter + pandas2ri.converter):
        table = conversion.get_conversion().rpy2py(data_frame)
    table.index = pd.Index(table.index.astype(str), name="sample-id")
    return Artifact.import_data(FeatureTable[Unconstrained], table)


def nutrimouse_tables_factory():
    return ResultCollection(
        {"gene": _nutrimouse_table("gene"), "lipid": _nutrimouse_table("lipid")}
    )


def nutrimouse_metadata_factory():
    data = _load_nutrimouse()
    sample_ids = list(base.rownames(data.rx2("gene")))
    diets = list(base.as_character(data.rx2("diet")))
    return Metadata(
        pd.DataFrame({"diet": diets}, index=pd.Index(sample_ids, name="sample-id"))
    )


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


def select_features_block_splsda_nutrimouse(use):
    use.comment(
        "Load the nutrimouse dataset bundled with mixOmics: 40 mice, 120 "
        "gene-expression features, and 21 lipid features. The small grid and three "
        "repeats keep this demonstration short; use more repeats for a thorough "
        "analysis. Usees an illustrative prebuilt component-selection artifact whose "
        "weighted BER choice is two components, and a gene/lipid association "
        "matrix with integration weight 0.9."
        'Martin, Pascal GP, et al. "Novel aspects of PPARα‐mediated regulation of '
        'lipid and xenobiotic metabolism revealed through a nutrigenomic study." '
        "Hepatology 45.3 (2007): 767-777."
    )
    tables = use.init_artifact_collection(
        "nutrimouse_tables", nutrimouse_tables_factory
    )
    metadata = use.init_metadata("nutrimouse_metadata", nutrimouse_metadata_factory)
    diet = use.get_metadata_column("diet", "diet", metadata)
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


def tune_components_block_splsda_nutrimouse(use):
    use.comment(
        "Use gene expression and lipid measurements from the mixOmics Nutrimouse "
        "dataset to classify 40 mice by diet. Evaluate up to two components "
        "using three repeats of three-fold cross-validation and a gene/lipid "
        "association weight of 0.9. The report shows weighted- and majority-vote "
        "error rates and the selected component counts. "
        'Martin, Pascal GP, et al. "Novel aspects of PPARα‐mediated regulation of '
        'lipid and xenobiotic metabolism revealed through a nutrigenomic study." '
        "Hepatology 45.3 (2007): 767-777."
    )
    tables = use.init_artifact_collection(
        "nutrimouse_tables", nutrimouse_tables_factory
    )
    metadata = use.init_metadata("nutrimouse_metadata", nutrimouse_metadata_factory)
    diet = use.get_metadata_column("diet", "diet", metadata)
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
