"""
This module contains functions for regression and factor analysis.
"""

from typing import Dict, List, Optional
import logging

import pandas as pd

from anndata import AnnData

from shackett_utils.genomics import adata_utils
from shackett_utils.statistics import multi_model_fitting
from shackett_utils.statistics.utils import (
    get_stat_abbreviation,
    validate_regression_output_types,
)
from shackett_utils.statistics.constants import (
    STATISTICS_DEFS,
    TIDY_DEFS,
    STATISTICAL_SUMMARIES,
)
from shackett_utils.genomics.constants import REGRESSION_DEFAULT_STATS


def adata_model_fitting(
    adata: AnnData,
    formula: str,
    layer: Optional[str] = None,
    model_class: Optional[str] = None,
    outputs: Optional[List[str]] = [STATISTICAL_SUMMARIES.TIDY],
    n_jobs: int = -1,
    batch_size: int = 100,
    model_name: Optional[str] = None,
    progress_bar: bool = True,
    fdr_control: bool = True,
    allow_failures: bool = False,
    **model_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Apply a regression model to each feature in an AnnData object and return statistics.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    formula : str
        Formula for regression in patsy format (e.g., '~ batch').
        Don't include the dependent variable, as each feature will be used.
    layer : Optional[str]
        If provided, use this layer instead of X. The layer can be a string referring
        to a layer in adata.layers, or a key in adata.obsm for alternative feature matrices.
    model_class : Optional[str]
        Type of model to fit ('ols', 'gam', etc.). If None, will be auto-detected from formula.
    outputs : Optional[List[str]]
        Which summaries to return. Any combination of 'tidy', 'glance', 'augment'.
        Default is [STATISTICAL_SUMMARIES.TIDY].
    n_jobs : int
        Number of parallel jobs. -1 means using all processors.
    batch_size : int
        Number of features to process in each batch.
    model_name : Optional[str]
        Name of the model for the output dataframe.
    progress_bar : bool
        Whether to display a progress bar.
    fdr_control : bool
        Whether to apply FDR control to the p-values.
    allow_failures : bool
        If True, handle errors gracefully and return empty DataFrame for failed fits.
        If False, raise exceptions for debugging. Default is False.
    **model_kwargs :
        Additional arguments passed to model fitting

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary keyed by output type containing the requested summaries.
    """

    validate_regression_output_types(outputs)
    feature_names, data_matrix = adata_utils.get_adata_features_and_data(
        adata, layer=layer
    )

    results = multi_model_fitting.fit_parallel_models_formula(
        X_features=data_matrix,
        data=adata.obs,
        feature_names=feature_names,
        formula=formula,
        model_class=model_class,
        n_jobs=n_jobs,
        batch_size=batch_size,
        model_name=model_name,
        fdr_control=fdr_control,
        progress_bar=progress_bar,
        allow_failures=allow_failures,
        outputs=outputs,
        **model_kwargs,
    )
    return results


def add_regression_results_to_anndata(
    adata: AnnData,
    results: Dict[str, pd.DataFrame],
    model_label: Optional[str] = None,
    stats_to_add: Optional[List[str]] = None,
    key_added: str = "regression_results",
    fdr_cutoff: Optional[float] = None,
    inplace: bool = False,
) -> Optional[AnnData]:
    """
    Add regression results to an AnnData object's .var dataframe.

    Parameters
    ----------
    adata : AnnData
        AnnData object to add results to
    results : Dict[str, pd.DataFrame]
        Dictionary keyed by output type containing the requested summaries.
    model_label : Optional[str]
        Label for the model. If provided, all summaries will be prefixed by this key. If missing, no prefix is used.
    stats_to_add : Optional[List[str]]
        List of statistics to add to the AnnData object.
        If None, uses REGRESSION_DEFAULT_STATS
    key_added : str, optional
        Key under which to add the results in adata.uns.
    fdr_cutoff : Optional[float]
        If provided, adds significance mask columns using this cutoff
        on q-values (if available) or p-values
    inplace : bool
        Whether to modify adata inplace or return a copy

    Returns
    -------
    Optional[AnnData]
        If inplace=False, returns modified copy of adata.
        If inplace=True, returns None.

    Raises
    ------
    ValueError
        If results_df is empty
        If a feature has multiple values for the same term
        If invalid statistics are requested
    """
    if not inplace:
        adata = adata.copy()

    _check_conflicts(adata, {key_added}, "adata.uns")
    adata.uns[key_added] = dict(results)

    if STATISTICAL_SUMMARIES.TIDY in results:
        _add_tidy_results(
            adata,
            results[STATISTICAL_SUMMARIES.TIDY],
            model_label,
            stats_to_add,
            fdr_cutoff,
        )

    if STATISTICAL_SUMMARIES.GLANCE in results:
        _add_glance_results(adata, results[STATISTICAL_SUMMARIES.GLANCE], model_label)

    if STATISTICAL_SUMMARIES.AUGMENT in results:
        _add_augment_results(adata, results[STATISTICAL_SUMMARIES.AUGMENT], model_label)

    return None if inplace else adata


def _add_tidy_results(
    adata: AnnData,
    tidy_df: pd.DataFrame,
    model_label: Optional[str],
    stats_to_add: Optional[List[str]],
    fdr_cutoff: Optional[float],
) -> None:
    available_stats = [s for s in REGRESSION_DEFAULT_STATS if s in tidy_df.columns]
    if stats_to_add is not None:
        invalid = set(stats_to_add) - set(REGRESSION_DEFAULT_STATS)
        if invalid:
            raise ValueError(
                f"Invalid statistics requested: {invalid}. Available: {REGRESSION_DEFAULT_STATS}"
            )
        stats_to_add = [s for s in stats_to_add if s in available_stats]
    else:
        stats_to_add = available_stats

    var_df = _build_term_results(tidy_df, stats_to_add, fdr_cutoff, model_label)
    _check_conflicts(adata, set(var_df.columns), "adata.var")
    adata.var = adata.var.join(var_df, how="left")


def _add_glance_results(
    adata: AnnData,
    glance_df: pd.DataFrame,
    model_label: Optional[str],
) -> None:
    stat_cols = [
        c
        for c in glance_df.columns
        if c not in (STATISTICS_DEFS.FEATURE_NAME, STATISTICS_DEFS.MODEL_NAME)
    ]
    new_cols = {_col_name(c, model_label) for c in stat_cols}
    _check_conflicts(adata, new_cols, "adata.var")

    summary = glance_df.set_index(STATISTICS_DEFS.FEATURE_NAME)[stat_cols].rename(
        columns={c: _col_name(c, model_label) for c in stat_cols}
    )
    adata.var = adata.var.join(summary, how="left")


def _add_augment_results(
    adata: AnnData,
    augment_df: pd.DataFrame,
    model_label: Optional[str],
) -> None:
    fitted_key = _col_name("fitted", model_label)
    resid_key = _col_name("residuals", model_label)
    _check_conflicts(adata, {fitted_key, resid_key}, "adata.layers")

    for layer_col, layer_key in [(".fitted", fitted_key), (".resid", resid_key)]:
        aug_work = augment_df.reset_index()
        obs_col = aug_work.columns[0]
        pivoted = aug_work.pivot(
            index=obs_col,
            columns=STATISTICS_DEFS.FEATURE_NAME,
            values=layer_col,
        )
        missing_obs = set(adata.obs_names) - set(pivoted.index)
        missing_var = set(adata.var_names) - set(pivoted.columns)
        if missing_obs or missing_var:
            raise ValueError(
                f"Augment results do not align with adata for layer '{layer_key}'. "
                f"Missing obs: {missing_obs or 'none'}. Missing vars: {missing_var or 'none'}."
            )
        adata.layers[layer_key] = pivoted.loc[adata.obs_names, adata.var_names].values


def _build_term_results(
    results: pd.DataFrame,
    stats_to_add: List[str],
    fdr_cutoff: Optional[float],
    model_label: Optional[str],
) -> pd.DataFrame:
    if results.empty:
        raise ValueError("Tidy results DataFrame is empty")

    for term in results[TIDY_DEFS.TERM].unique():
        term_results = results[results[TIDY_DEFS.TERM] == term]
        if len(term_results[STATISTICS_DEFS.FEATURE_NAME].unique()) != len(
            term_results
        ):
            raise ValueError(
                f"Found duplicate features for term '{term}'. "
                "Each feature should have exactly one value per term."
            )

    if fdr_cutoff is not None:
        if STATISTICS_DEFS.Q_VALUE in results.columns:
            results = results.copy()
            results["significance"] = results[STATISTICS_DEFS.Q_VALUE] < fdr_cutoff
            stats_to_add = list(stats_to_add) + ["significance"]
        else:
            logging.warning(
                "FDR cutoff provided but no q-values found. Significance mask will not be created."
            )

    stat_dfs = []
    for stat in stats_to_add:
        if stat not in results.columns:
            continue
        stat_prefix = get_stat_abbreviation(stat)
        stat_df = results.pivot(
            index=STATISTICS_DEFS.FEATURE_NAME,
            columns=TIDY_DEFS.TERM,
            values=stat,
        )
        stat_df.columns = [
            _col_name(
                f"{stat_prefix}_{col.replace(' ', '_').replace('-', '_')}", model_label
            )
            for col in stat_df.columns
        ]
        stat_dfs.append(stat_df)

    if not stat_dfs:
        logging.warning("No statistics to add")
        return pd.DataFrame()

    return pd.concat(stat_dfs, axis=1).dropna(axis=1, how="all")


def _check_conflicts(adata: AnnData, new_cols: set, context: str) -> None:
    if context == "adata.var":
        conflicts = {
            col
            for col in new_cols
            if col in adata.var.columns and adata.var[col].notna().any()
        }
    elif context == "adata.layers":
        conflicts = {key for key in new_cols if key in adata.layers}
    else:  # adata.uns
        conflicts = {key for key in new_cols if key in adata.uns}

    if conflicts:
        raise ValueError(f"Non-empty values already present in {context}: {conflicts}")


def _col_name(base: str, model_label: Optional[str]) -> str:
    return f"{model_label}_{base}" if model_label else base
