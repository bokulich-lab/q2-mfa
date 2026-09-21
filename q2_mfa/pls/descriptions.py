# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
error_rate_criteria_description = """### Error-rate criteria

- **Overall.ER** is the fraction of all held-out samples classified incorrectly.
- **Overall.BER** is the average of the error rates calculated separately for
  each class, so each class has equal influence on the final value.

Classes are *unbalanced* when they have different sample counts. In that case,
the mixOmics vignette recommends considering **Overall.BER**, which weights
each class equally. **Overall.ER** weights errors by their observed sample
frequencies."""

prediction_distance_description = """### Prediction distances

**max.dist**, **centroids.dist**, and **mahalanobis.dist** are alternative rules for
turning a sample's predicted values into a class assignment. The mixOmics
vignette characterizes **max.dist** as linear and **centroids.dist** and
**mahalanobis.dist** as non-linear. It notes that the prediction distance can
change classification performance and suggests increasing distance complexity
from maximum to Mahalanobis as separating the classes becomes more
challenging."""

voting_scheme_description = """### Voting schemes

In multiblock classification, every data block produces a class prediction for
a held-out sample.
**MajorityVote** gives each block one vote and uses the majority decision.
**WeightedVote** combines block predictions using weights based on the
correlation between the predicted components and the outcome (**Y**).

Use majority voting when every block should have equal influence. Use weighted
voting when blocks whose components are more strongly correlated with the
outcome should have greater influence."""

curve_usage_description = """Use the curves as a diagnostic: inspect how
cross-validated error changes as components are added and compare the voting
schemes, error criteria, and prediction distances. Select a component count
from the matching **choice.ncomp** table entry, which applies the repeated
cross-validation *t*-test rule rather than choosing the lowest plotted point.

For more information, consult the mixOmics documentation.
"""

component_selection_description = """#### Choosing a component count

The reported value is **not** simply the component with the smallest observed
error. mixOmics compares mean cross-validation error as components are added
with one-sided *t*-tests. An additional component is kept
only when it significantly reduces mean error at **signif-threshold**; a later,
lower raw error can therefore be rejected when it is not statistically
supported.

*PLS-DA is iterative: each component is orthogonal to the preceding components
and gradually increases discrimination between sample classes. A model with a
specified **ncomp** is compounding; for example, component 3 includes the
model trained on components 1 and 2.*

For more information, consult the mixOmics documentation."""

report_descriptions = {
    "error_rate_weighted": (
        "### Weighted-vote predictions\n\n"
        + curve_usage_description
        + "\n\n"
        + voting_scheme_description
        + "\n\n"
        + error_rate_criteria_description
        + "\n\n"
        + prediction_distance_description
    ),
    "error_rate_majority": (
        "### Majority-vote predictions\n\n"
        + curve_usage_description
        + "\n\n"
        + voting_scheme_description
        + "\n\n"
        + error_rate_criteria_description
        + "\n\n"
        + prediction_distance_description
    ),
    "choice_matrix_weighted": (
        "### Weighted-vote component choices\n\n"
        + component_selection_description
        + "\n\n"
        + voting_scheme_description
        + "\n\n"
        + error_rate_criteria_description
        + "\n\n"
        + prediction_distance_description
    ),
    "choice_matrix_majority": (
        "### Majority-vote component choices\n\n"
        + component_selection_description
        + "\n\n"
        + voting_scheme_description
        + "\n\n"
        + error_rate_criteria_description
        + "\n\n"
        + prediction_distance_description
    ),
}


jsonl_descriptions = {
    "error_rate_weighted": (
        "Cross-validated weighted-vote error-rate means and standard "
        "deviations from mixOmics perf() WeightedVote.error.rate and "
        "WeightedVote.error.rate.sd."
    ),
    "error_rate_majority": (
        "Cross-validated majority-vote error-rate means and standard "
        "deviations from mixOmics perf() MajorityVote.error.rate and "
        "MajorityVote.error.rate.sd."
    ),
    "choice_matrix_weighted": (
        "Component-choice matrix for weighted voting from mixOmics "
        "perf() choice.ncomp$WeightedVote."
    ),
    "choice_matrix_majority": (
        "Component-choice matrix for majority voting from mixOmics "
        "perf() choice.ncomp$MajorityVote."
    ),
}
