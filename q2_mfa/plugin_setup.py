# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from q2_types.feature_table import Composition, FeatureTable, Frequency, Normalized
from rachis.core.type import Categorical, Choices, Float, Int, Range, Str
from rachis.plugin import Bool, MetadataColumn, Plugin

from q2_mfa import __version__, pretreat_metabolome, transform_clr

plugin = Plugin(
    name="mfa",
    version=__version__,
    website="https://github.com/bokulich-lab/q2-mfa",
    package="q2_mfa",
    description="A QIIME 2 plugin for PCA and MFA analysis.",
    short_description="PCA and MFA analysis",
    citations=[],
)

plugin.methods.register_function(
    function=transform_clr,
    inputs={"table": FeatureTable[Frequency]},
    parameters={
        "pseudocount": Float % Range(0, None, inclusive_start=False),
    },
    outputs=[("clr_table", FeatureTable[Composition])],
    input_descriptions={"table": "The frequency table."},
    parameter_descriptions={
        "pseudocount": "The pseudocount to add to the table before the "
        "transformation. If it is set to None, the pseudocount is "
        "computed as the minimum non-zero value.",
    },
    output_descriptions={"clr_table": "The CLR transformed table."},
    name="Centered log-ratio (CLR) transformation.",
    description="A centered log-ratio transformation of the input table.",
    citations=[],
)

plugin.methods.register_function(
    function=pretreat_metabolome,
    inputs={"table": FeatureTable[Frequency]},
    parameters={
        "sample_normalization": Str % Choices(["pqn"]),
        "pqn_method": Str % Choices(["median", "mean"]),
        "pqn_ref_samples": MetadataColumn[Categorical],
        "pqn_ref_label": Str,
        "transform": Str % Choices(["log", "log10", "sqrt"]),
        "pseudocount": Float % Range(0, None, inclusive_start=False),
        "center": Bool,
        "scale": Str % Choices(["auto", "pareto", "range"]),
        "impute": Str % Choices(["knn", "rf"]),
        "knn_neighbors": Int % Range(1, None),
        "rf_n_estimators": Int % Range(1, None),
        "rf_random_state": Int,
    },
    outputs=[("pretreated_table", FeatureTable[Normalized])],
    input_descriptions={"table": "Metabolomics feature table."},
    parameter_descriptions={
        "sample_normalization": (
            "Sample-level normalization to apply: 'pqn' "
            "(Probabilistic Quotient Normalization)."
        ),
        "pqn_method": (
            "How to construct the PQN reference spectrum: 'median' or 'mean'."
        ),
        "pqn_ref_samples": (
            "Categorical metadata column name used to select samples "
            "for building the PQN reference spectrum."
        ),
        "pqn_ref_label": (
            "Label value in `pqn_ref_samples` indicating which samples to use "
            "for the PQN reference. Required when `pqn_ref_samples` is provided."
        ),
        "transform": "Transformation applied to the data ('log', 'log10', 'sqrt').",
        "pseudocount": (
            "Offset added before log transform. If omitted/None and transform='log', "
            "it is inferred as the minimum non-zero value."
        ),
        "center": "If True, mean-center each feature",
        "scale": (
            "Feature scaling method applied: 'auto' (mean-center and divide by std), "
            "'pareto' (mean-center and divide by sqrt(std)), or 'range' (mean-center "
            "and divide by max-min)."
        ),
        "impute": (
            "Missing-value imputation method. K-Nearest Neighbors imputation, or "
            "Random Forest imputation."
        ),
        "knn_neighbors": "Number of neighbors for KNN imputation.",
        "rf_n_estimators": "Number of trees for RandomForest imputation.",
        "rf_random_state": "Random state for RandomForest reproducibility.",
    },
    output_descriptions={
        "pretreated_table": (
            "Pretreated table after optional imputation, normalization, "
            "transformation, centering, and scaling."
        )
    },
    name="Metabolomics pretreatment",
    description=(
        "Applies metabolomics-friendly pretreatment in order: imputation,  sample "
        "normalization, transformation, centering, and feature scaling."
    ),
    citations=[],
)
