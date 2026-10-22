# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import importlib

from q2_types.feature_table import FeatureTable, Frequency, Unconstrained
from rachis import Citations
from rachis.core.type import (
    Bool,
    Categorical,
    Choices,
    Collection,
    Float,
    Int,
    Metadata,
    MetadataColumn,
    Range,
    Str,
)
from rachis.plugin import Plugin, Threads, Visualization

from q2_mfa import __version__
from q2_mfa.component_analysis import ComponentAnalysis, ComponentAnalysisDirFmt
from q2_mfa.component_analysis.mfa import mfa
from q2_mfa.component_analysis.pca import pca
from q2_mfa.pls import (
    AssociationMatrix,
    AssociationMatrixDirFmt,
    PLSComponentSelection,
    PLSComponentSelectionDirFmt,
    PLSFeatureSelection,
    PLSFeatureSelectionDirFmt,
    _feature_selection_block_splsda,
    _feature_selection_block_splsda_visualiser,
    _tune_components_block_splsda,
    _tune_components_block_visualisation,
    feature_selection_block_splsda,
    tune_components_block_splsda,
)
from q2_mfa.preprocessing import transform_clr

citations = Citations.load("citations.bib", package="q2_mfa")

plugin = Plugin(
    name="mfa",
    version=__version__,
    website="https://github.com/bokulich-lab/q2-mfa",
    package="q2_mfa",
    description="A QIIME 2 plugin for Multiomics analysis.",
    short_description="Multiomics analysis",
    citations=[],
)


plugin.methods.register_function(
    function=transform_clr,
    inputs={"table": FeatureTable[Frequency]},
    parameters={
        "pseudocount": Float % Range(0, None, inclusive_start=False),
        "replacement_method": Str % Choices("multiplicative", "pseudocount"),
        "delta": Float % Range(0, None, inclusive_start=False),
    },
    outputs=[("transformed_table", FeatureTable[Unconstrained])],
    input_descriptions={"table": "The frequency table."},
    parameter_descriptions={
        "pseudocount": (
            "Value used to replace zeros before CLR. If not provided, a "
            "`pseudocount` value of 1 is used. This parameter is only used "
            "when `replacement-method` is `pseudocount`."
        ),
        "replacement_method": (
            "Method used to handle zeros before CLR. `multiplicative` replaces "
            "zeros while rescaling the remaining values to preserve the "
            "composition. `pseudocount` adds the `pseudocount` value to all "
            "values in the table."
        ),
        "delta": (
            "Replacement value used for multiplicative replacement. This "
            "parameter is only used when `replacement-method` is "
            "`multiplicative`."
        ),
    },
    output_descriptions={"transformed_table": "The CLR transformed table."},
    name="Centered log-ratio (CLR) transformation.",
    description=(
        "A centered log-ratio transformation of the input table. The CLR-transformed "
        "table contains real-valued coordinates in Euclidean space, removing the "
        "constant-sum constraint of compositional data. Zeros can be handled "
        "either by additive pseudocount replacement or multiplicative "
        "replacement before the CLR is applied. For more information on the "
        "implementations or the parameters please consult the scikit-bio documentation."
    ),
    citations=[
        citations["martin2003dealing"],
        citations["aitchison1982statistical"],
        citations["aton2025scikit"],
    ],
)

ordination_parameters = {
    "n_components": Int % Range(2, None),
    "rescale_with_mean": Bool,
    "rescale_with_std": Bool,
    "filter_zero_variance": Bool,
    "engine": Str % Choices(["sklearn", "scipy"]),
    "n_iter": Int % Range(0, None),
    "random_state": Int,
}
ordination_parameter_descriptions = {
    "n_components": "Number of principal components to compute.",
    "rescale_with_mean": (
        "Whether to center each feature by subtracting its mean before SVD (Singular "
        "Value Decomposition)."
    ),
    "rescale_with_std": (
        "Whether to standardize each feature to unit variance before SVD."
    ),
    "filter_zero_variance": (
        "Whether to remove columns with zero variance before SVD."
    ),
    "engine": (
        "SVD engine used. `sklearn` uses faster randomized SVD, while `scipy` "
        "uses deterministic SciPy SVD."
    ),
    "n_iter": (
        "Number of iterations used by the `sklearn` randomized SVD "
        "engine. This parameter is ignored by the `scipy` engine."
    ),
    "random_state": (
        "Random seed used by the `sklearn` SVD engine. Pass an int for "
        "reproducible results across multiple function calls. This "
        "parameter is ignored by the `scipy` engine."
    ),
}

mfa_parameters = {
    "sample_metadata": Metadata,
    "metadata_groups": Collection[Str],
    **ordination_parameters,
}
mfa_parameter_descriptions = {
    "sample_metadata": (
        "Optional sample metadata to include as additional MFA groups."
    ),
    "metadata_groups": (
        "Optional mapping from metadata group names to comma-separated metadata "
        "column strings. Pass a collection/dict such as "
        "`{'group': 'column1,column2'}` to define explicit groups. If a single "
        "string is provided, all metadata columns are included in a group with "
        "that string as the group name. If `sample-metadata` is provided without "
        "`metadata-groups`, all metadata columns are included in a group named "
        "`metadata`. Groups must contain only numeric or only categorical columns."
    ),
    **ordination_parameter_descriptions,
}

plugin.methods.register_function(
    function=pca,
    inputs={"table": FeatureTable[Unconstrained]},
    parameters=ordination_parameters,
    outputs=[("pca_results", ComponentAnalysis)],
    input_descriptions={"table": "The frequency table."},
    parameter_descriptions=ordination_parameter_descriptions,
    output_descriptions={"pca_results": "The PCA results."},
    name="PCA",
    description=(
        "Principal component analysis implementation with the prince package. "
        "The output is a component-analysis result containing eigenvalues, "
        "sample coordinates, feature coordinates, variance percentages, and "
        "supporting PCA tables from prince. The result can also be viewed as an "
        "ordination result through the registered transformer. Features with "
        "missing values are automatically removed before analysis. Please check "
        "the prince package docs for more information:"
        "https://maxhalford.github.io/prince/pca/"
    ),
    citations=[
        citations["Halford_Prince"],
    ],
)

plugin.methods.register_function(
    function=mfa,
    inputs={"tables": Collection[FeatureTable[Unconstrained]]},
    parameters=mfa_parameters,
    outputs=[("mfa_results", ComponentAnalysis)],
    input_descriptions={
        "tables": (
            "Optional feature tables to include as MFA groups. At least two "
            "groups must be provided across feature tables and sample metadata "
            "groups."
        )
    },
    parameter_descriptions=mfa_parameter_descriptions,
    output_descriptions={"mfa_results": "MFA results"},
    name="Multiple Factor Analysis (MFA)",
    description=(
        "Multiple Factor Analysis (MFA). Each table is treated as a separate group, "
        "and the analysis is performed with the prince python package. Features with "
        "missing values are automatically removed before analysis. Please check "
        "the prince package documentation for more information: "
        "https://maxhalford.github.io/prince/mfa/"
    ),
    citations=[
        citations["escofier1994multiple"],
        citations["Halford_Prince"],
    ],
)

component_tuning_parameters = {
    "y": MetadataColumn[Categorical],
    "design_weight": Float % Range(0, 1),
    "ncomp": Int % Range(2, None),
    "scale": Bool,
    "tol": Float % Range(0, None, inclusive_start=False),
    "max_iter": Int % Range(1, None),
    "near_zero_var": Bool,
    "validation": Str % Choices("Mfold", "loo"),
    "folds": Int % Range(2, None),
    "nrepeat": Int % Range(3, None),
    "signif_threshold": Float % Range(0, 1),
    "seed": Int % Range(0, 2**31 - 1),
    "threads": Threads,
}

component_tuning_parameter_descriptions = {
    "y": "Categorical response variable to predict.",
    "design_weight": (
        "Single block-relationship weight. Provide exactly one of "
        "`design-weight` or `design-matrix`. A value from 0 to 1 creates a "
        "fully connected design matrix with this value in every off-diagonal cell; "
        "0 means no modeled block relationships and 1 is the maximum."
    ),
    "ncomp": "Number of components to fit and evaluate by cross-validation.",
    "scale": (
        "Whether to standardize every feature within each input block to zero "
        "mean and unit variance before fitting."
    ),
    "tol": (
        "Positive convergence tolerance for the iterative sPLS-DA fit. The "
        "algorithm stops updating a component when successive estimates differ "
        "by less than this value."
    ),
    "max_iter": (
        "Maximum iterative updates allowed while fitting each latent component. "
        "This limits fitting time when the convergence tolerance is not reached."
    ),
    "near_zero_var": (
        "Whether to remove predictors with zero or near-zero variance before "
        "fitting. Enable this for blocks with many zero-valued features; leave "
        "it disabled when that filtering is unnecessary to reduce computation."
    ),
    "validation": (
        "Internal cross-validation strategy used to estimate classification "
        "error rates: `Mfold` repeatedly holds out folds, while `loo` leaves out "
        "one sample at a time."
    ),
    "folds": (
        "Number of sample folds for M-fold cross-validation. Ignored when "
        "`validation` is `loo`. Each fold should contain enough samples from "
        "every response class for model fitting and evaluation."
    ),
    "nrepeat": (
        "Number of independent M-fold cross-validation repetitions. More "
        "repetitions reduce sensitivity to random fold assignments but increase "
        "runtime; it is not needed when `validation` is `loo`."
    ),
    "signif_threshold": (
        "Minimum improvement in cross-validated error rate required before an "
        "additional latent component is considered beneficial."
    ),
    "seed": "Random seed used for reproducible cross-validation fold assignments.",
    "threads": (
        "Number of BiocParallel workers used for cross-validation model fits. "
        "Use 1 for serial execution. The Rachis/QIIME value `auto` is "
        "converted to 0; both 0 and `auto` omit `workers` and let "
        "BiocParallel select its default."
    ),
}

component_tuning_input_descriptions = {
    "tables": (
        "Named feature tables used as sPLS-DA blocks. Before fitting, every "
        "block and the response are restricted to their shared sample IDs; a "
        "sample missing from one or more blocks is dropped. Table names identify "
        "the corresponding design-matrix rows and columns."
    ),
    "design_matrix": (
        "Block-relationship matrix. Provide exactly one of `design-matrix` or "
        "`design-weight`. This square table must have rows and columns named "
        "exactly as the feature-table collection, zeroes on its diagonal, and "
        "symmetric numeric off-diagonal values from 0 (no modeled relationship) "
        "to 1 (maximum modeled relationship)."
    ),
}

feature_selection_parameters = {
    **component_tuning_parameters,
    "ncomp": Int % Range(1, None),
    "keepx": Collection[Str],
    "already_tested_x": Collection[Str],
    "distance": Str % Choices("max.dist", "centroids.dist", "mahalanobis.dist"),
    "measure": Str % Choices("overall", "BER"),
    "weighted": Bool,
}

feature_selection_parameter_descriptions = {
    **component_tuning_parameter_descriptions,
    "ncomp": (
        "Number of components to tune. Provide exactly one of this value or a "
        "tuned-ncomp artifact."
    ),
    "keepx": (
        "Named comma-separated candidate feature counts for every input "
        'block. For example, microbiome="5,10" and '
        'metabolomics="20,40" test all four count pairs (5,20), '
        "(5,40), (10,20), and (10,40) for each component. Every block must "
        "provide at least one positive integer."
    ),
    "already_tested_x": (
        "Named comma-separated selected keepX counts for previously tuned "
        'components. For example, microbiome="10,8" and '
        'metabolomics="40,30" fix component 1 at (10,40) and component '
        "2 at (8,30), then tune the next component using keepx. Every block "
        "must provide the same number of selections and the block names must "
        "match the input tables."
    ),
    "distance": (
        "Prediction distance used to calculate cross-validated classification "
        "error rates and select keepX values."
    ),
    "measure": (
        "Classification error measure used to select keepX values: overall "
        "error rate or balanced error rate (BER)."
    ),
    "weighted": (
        "Whether to select keepX values using weighted voting. Disable to use "
        "majority voting."
    ),
}

feature_selection_input_descriptions = {
    **component_tuning_input_descriptions,
    "tuned_ncomp": (
        "Component-tuning artifact used to determine ncomp. Provide exactly "
        "one of this artifact or ncomp."
    ),
}

plugin.methods.register_function(
    function=_tune_components_block_splsda,
    inputs={
        "tables": Collection[FeatureTable[Unconstrained]],
        "design_matrix": AssociationMatrix,
    },
    parameters=component_tuning_parameters,
    outputs=[("tune_components", PLSComponentSelection)],
    input_descriptions=component_tuning_input_descriptions,
    parameter_descriptions=component_tuning_parameter_descriptions,
    output_descriptions={
        "tune_components": (
            "sPLS-DA weighted- and majority-vote component-tuning metrics."
        )
    },
    name="Tune block sPLS-DA (DIABLO) components",
    description=(
        "Tunes the number of components for a block sPLS-DA (DIABLO) model with the "
        "mixOmics package. The action fits a dense model across component "
        "counts and uses cross-validation to report weighted- and "
        "majority-vote classification error rates, allowing users to identify "
        "a suitable component count. For questions about the implementation or "
        "parameters, consult the mixOmics documentation: "
        "https://mixomics.org/methods/diablo/"
    ),
    citations=[
        citations["rohart2017mixomics"],
    ],
)

plugin.pipelines.register_function(
    function=_tune_components_block_visualisation,
    inputs={"tune_components": PLSComponentSelection},
    parameters={},
    outputs=[("visualization", Visualization)],
    input_descriptions={
        "tune_components": (
            "sPLS-DA weighted- and majority-vote component-tuning metrics."
        )
    },
    parameter_descriptions={},
    output_descriptions={
        "visualization": (
            "Report containing weighted and majority-vote error-rate plots "
            "and component-choice matrices."
        ),
    },
    name="Visualize block sPLS-DA (DIABLO) component tuning",
    description=(
        "Creates a report of weighted and majority-vote component-tuning "
        "diagnostics for block sPLS-DA (DIABLO)."
    ),
    citations=[
        citations["rohart2017mixomics"],
    ],
)

plugin.pipelines.register_function(
    function=tune_components_block_splsda,
    inputs={
        "tables": Collection[FeatureTable[Unconstrained]],
        "design_matrix": AssociationMatrix,
    },
    parameters=component_tuning_parameters,
    outputs=[
        ("tune_components", PLSComponentSelection),
        ("visualization", Visualization),
    ],
    input_descriptions=component_tuning_input_descriptions,
    parameter_descriptions=component_tuning_parameter_descriptions,
    output_descriptions={
        "tune_components": (
            "sPLS-DA weighted- and majority-vote component-tuning metrics."
        ),
        "visualization": (
            "Report containing weighted and majority-vote error-rate plots "
            "and component-choice matrices."
        ),
    },
    name="Tune block sPLS-DA (DIABLO) components",
    description=(
        "Tunes the number of components for a block sPLS-DA (DIABLO) model with the "
        "mixOmics package. The pipeline fits a dense model across component "
        "counts, uses cross-validation to calculate weighted- and "
        "majority-vote classification error rates, and creates visualizations "
        "of the component-tuning diagnostics. For questions about the "
        "implementation or parameters, consult the mixOmics documentation: "
        "https://mixomics.org/methods/diablo/"
    ),
    citations=[
        citations["rohart2017mixomics"],
    ],
)

plugin.methods.register_function(
    function=_feature_selection_block_splsda,
    inputs={
        "tables": Collection[FeatureTable[Unconstrained]],
        "tuned_ncomp": PLSComponentSelection,
        "design_matrix": AssociationMatrix,
    },
    parameters=feature_selection_parameters,
    outputs=[("feature_selection", PLSFeatureSelection)],
    input_descriptions=feature_selection_input_descriptions,
    parameter_descriptions=feature_selection_parameter_descriptions,
    output_descriptions={
        "feature_selection": (
            "Cross-validated keepX error rates and selected feature counts for "
            "block sPLS-DA (DIABLO) feature selection."
        )
    },
    name="Internal block sPLS-DA (DIABLO) feature selection",
    description=(
        "Internal feature-selection action used by the public block sPLS-DA "
        "(DIABLO) feature-selection pipeline."
    ),
    citations=[citations["rohart2017mixomics"]],
)

plugin.pipelines.register_function(
    function=_feature_selection_block_splsda_visualiser,
    inputs={"feature_selection": PLSFeatureSelection},
    parameters={},
    outputs=[("visualization", Visualization)],
    input_descriptions={
        "feature_selection": (
            "Cross-validated keepX error rates and selected feature counts "
            "for block sPLS-DA (DIABLO) feature selection."
        )
    },
    parameter_descriptions={},
    output_descriptions={
        "visualization": (
            "Tabulated feature-selection error rates and selected feature "
            "counts and, for exactly two blocks, one mean-error heatmap per "
            "component."
        )
    },
    name="Visualize block sPLS-DA (DIABLO) feature selection",
    description=(
        "Creates tables of block sPLS-DA (DIABLO) feature-selection error "
        "rates and selected feature counts and adds per-component heatmaps "
        "when exactly two input blocks were tuned."
    ),
    citations=[citations["rohart2017mixomics"]],
)

plugin.pipelines.register_function(
    function=feature_selection_block_splsda,
    inputs={
        "tables": Collection[FeatureTable[Unconstrained]],
        "tuned_ncomp": PLSComponentSelection,
        "design_matrix": AssociationMatrix,
    },
    parameters=feature_selection_parameters,
    outputs=[
        ("feature_selection", PLSFeatureSelection),
        ("visualization", Visualization),
    ],
    input_descriptions=feature_selection_input_descriptions,
    parameter_descriptions=feature_selection_parameter_descriptions,
    output_descriptions={
        "feature_selection": (
            "Cross-validated keepX error rates and selected feature counts for "
            "block sPLS-DA (DIABLO) feature selection."
        ),
        "visualization": (
            "Tabulated feature-selection error rates and selected feature "
            "counts and, for exactly two blocks, one mean-error heatmap per "
            "component."
        ),
    },
    name="Select block sPLS-DA (DIABLO) features",
    description=(
        "Aligns input blocks and the response, tunes sparse feature counts for "
        "a block sPLS-DA (DIABLO) model with mixOmics, and creates diagnostic "
        "visualizations. Reported error rates use weighted voting when "
        "weighted is enabled and majority voting otherwise. The measure "
        "parameter determines whether overall error rate (ER) or balanced "
        "error rate (BER) is used."
    ),
    citations=[citations["rohart2017mixomics"]],
)

plugin.register_formats(
    ComponentAnalysisDirFmt,
    PLSComponentSelectionDirFmt,
    AssociationMatrixDirFmt,
    PLSFeatureSelectionDirFmt,
)
plugin.register_semantic_types(
    ComponentAnalysis,
    PLSComponentSelection,
    AssociationMatrix,
    PLSFeatureSelection,
)
plugin.register_artifact_class(
    ComponentAnalysis,
    directory_format=ComponentAnalysisDirFmt,
    description=(
        "Represents the output for PCA and MFA actions implemented with the Prince "
        "package."
    ),
)
plugin.register_artifact_class(
    PLSComponentSelection,
    directory_format=PLSComponentSelectionDirFmt,
    description="Represents PLS component tuning results produced by mixOmics.",
)
plugin.register_artifact_class(
    AssociationMatrix,
    directory_format=AssociationMatrixDirFmt,
    description="Represents a matrix that denotes the association between multiple "
    "omics feature tables.",
)

plugin.register_artifact_class(
    PLSFeatureSelection,
    directory_format=PLSFeatureSelectionDirFmt,
    description="Represents PLS feature-selection results produced by mixOmics.",
)

importlib.import_module("q2_mfa.component_analysis.types._transformer")
