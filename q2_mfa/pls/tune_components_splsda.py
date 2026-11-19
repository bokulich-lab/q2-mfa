# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import secrets

import pandas as pd
from rachis import CategoricalMetadataColumn
from rachis.plugin import CaptureHolder
from rpy2.robjects.packages import importr

from q2_mfa.pls.types._format import PLSComponentSelectionDirFmt
from q2_mfa.pls.utils import (
    _align_samples,
    _build_bpparam,
    _call_without_messages,
    _print_component_choice,
    _r_choice_matrix_to_df,
    _r_perf_error_rate_to_df,
    _to_r_table,
    _to_r_target,
    _write_component_tuning_table,
    base,
)

mixomics = importr("mixOmics")


def tune_components_splsda(
    table: pd.DataFrame,
    y: CategoricalMetadataColumn,
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
    """Cross-validates a dense PLS-DA model to tune single-table components."""
    tables, target = _align_samples({"table": table}, y.to_series(), is_block=False)
    perf_model = _call_without_messages(
        mixomics.plsda,
        X=base.as_matrix(_to_r_table(tables["table"])),
        Y=_to_r_target(target),
        ncomp=ncomp,
        scale=scale,
        tol=tol,
        **{
            "max.iter": max_iter,
            "near.zero.var": near_zero_var,
        },
    )
    resolved_seed = CaptureHolder.get_or_set(seed, lambda: secrets.randbelow(2**31))
    perf_result = mixomics.perf(
        perf_model,
        dist="all",
        validation=validation,
        folds=folds,
        nrepeat=nrepeat,
        BPPARAM=_build_bpparam(threads),
        seed=resolved_seed,
        progressBar=False,
        **{"signif.threshold": signif_threshold},
    )
    choices = _r_choice_matrix_to_df(perf_result)
    error_rates = _r_perf_error_rate_to_df(perf_result)
    _print_component_choice(choices)

    data = PLSComponentSelectionDirFmt()
    _write_component_tuning_table(
        data.error_rate,
        error_rates,
        "Cross-validated overall and balanced error rates by distance and component.",
        error_rates=True,
    )
    _write_component_tuning_table(
        data.choice_matrix,
        choices,
        "Recommended component counts by error measure and prediction distance.",
    )
    return data
