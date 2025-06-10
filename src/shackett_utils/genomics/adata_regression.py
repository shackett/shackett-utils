"""
This module contains functions for regression and factor analysis.
"""

import os
from typing import List, Optional, Tuple, Dict, Union, Any, Set
import logging
import numpy as np

import pandas as pd

from anndata import AnnData

from shackett_utils.genomics import adata_utils
from shackett_utils.statistics import multi_model_fitting
from shackett_utils.statistics.constants import STATISTICS_DEFS, TIDY_DEFS
from shackett_utils.genomics.constants import REGRESSION_DEFAULT_STATS
from shackett_utils.statistics.constants import STATISTICS_ABBREVIATIONS

def adata_model_fitting(
    adata: AnnData,
    formula: str,
    layer: Optional[str] = None,
    model_class: Optional[str] = None,
    n_jobs: int = -1,
    batch_size: int = 100,
    model_name: Optional[str] = None,
    progress_bar: bool = True,
    fdr_control: bool = True,
    allow_failures: bool = False,
    **model_kwargs
) -> pd.DataFrame:
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
    pd.DataFrame
        DataFrame with regression statistics for each feature.
        
    Raises
    ------
    Exception
        If allow_failures is False and an error occurs during model fitting
    """
        
    feature_names, data_matrix = adata_utils.get_adata_features_and_data(adata, layer=layer)

    return multi_model_fitting.fit_parallel_models_formula(
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
        **model_kwargs
    )

def add_regression_results_to_anndata(
    adata: AnnData,
    results_df: pd.DataFrame,
    stats_to_add: Optional[List[str]] = None,
    key_added: str = "regression_results",
    fdr_cutoff: Optional[float] = None,
    inplace: bool = False
) -> Optional[AnnData]:
    """
    Add regression results to an AnnData object's .var dataframe.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to add results to
    results_df : pd.DataFrame
        DataFrame containing regression results
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
    # Work on a copy if not inplace
    if not inplace:
        adata = adata.copy()
    
    # Make a copy to avoid modifying the original results
    results = results_df.copy()
    
    # Determine which statistics are available in the results
    available_stats = [
        stat for stat in REGRESSION_DEFAULT_STATS 
        if stat in results.columns
    ]
    
    # Validate stats_to_add
    if stats_to_add is not None:
        invalid_stats = set(stats_to_add) - set(REGRESSION_DEFAULT_STATS)
        if invalid_stats:
            raise ValueError(
                f"Invalid statistics requested: {invalid_stats}. "
                f"Available statistics are: {REGRESSION_DEFAULT_STATS}"
            )
        # Filter stats_to_add to only include available statistics
        stats_to_add = [stat for stat in stats_to_add if stat in available_stats]
    else:
        # If None, use all available statistics
        stats_to_add = available_stats
    
    # Store in adata.uns (serialization-friendly: only DataFrame, not nested dict)
    adata.uns[key_added] = results
    
    # Build term-specific results
    all_results = _build_term_results(results, stats_to_add, fdr_cutoff)
    
    # Join results with adata.var
    adata.var = adata.var.join(all_results, how='left')
    
    return None if inplace else adata

def _build_term_results(
    results: pd.DataFrame,
    stats_to_add: List[str],
    fdr_cutoff: Optional[float] = None
) -> pd.DataFrame:
    """
    Build a DataFrame of term-specific results from regression output.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with regression results.
        Must contain TIDY_DEFS.TERM and STATISTICS_DEFS.FEATURE_NAME columns.
    stats_to_add : List[str]
        List of statistics to include in the output.
    fdr_cutoff : Optional[float]
        If provided, adds significance mask columns using this cutoff
        on q-values. If q-values are not present, a warning is logged.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features as index and term-specific columns.
        
    Raises
    ------
    ValueError
        If results DataFrame is empty
        If a feature has multiple values for the same term
    """
    if results.empty:
        raise ValueError("Results DataFrame is empty")
    
    # Check for duplicate features within terms
    for term in results[TIDY_DEFS.TERM].unique():
        term_results = results[results[TIDY_DEFS.TERM] == term]
        if len(term_results[STATISTICS_DEFS.FEATURE_NAME].unique()) != len(term_results):
            raise ValueError(
                f"Found duplicate features for term '{term}'. "
                "Each feature should have exactly one value per term."
            )
    
    # Create significance mask if cutoff provided and q-values are present
    if fdr_cutoff is not None:
        if STATISTICS_DEFS.Q_VALUE in results.columns:
            results = results.copy()
            results['significance'] = results[STATISTICS_DEFS.Q_VALUE] < fdr_cutoff
            stats_to_add = list(stats_to_add) + ['significance']
        else:
            logging.warning(
                "FDR cutoff was provided but no q-values found in results. "
                "Significance mask will not be created."
            )
    
    # Initialize list to store DataFrames for each statistic
    stat_dfs = []
    
    # Process each statistic
    for stat in stats_to_add:
        if stat in results.columns:
            # Get prefix for column names
            stat_prefix = STATISTICS_ABBREVIATIONS.get(stat, stat.split('_')[0].lower())
            
            # Pivot the data for this statistic
            stat_df = results.pivot(
                index=STATISTICS_DEFS.FEATURE_NAME,
                columns=TIDY_DEFS.TERM,
                values=stat
            )
            
            # Rename columns to include statistic prefix
            stat_df.columns = [f"{stat_prefix}_{col.replace(' ', '_').replace('-', '_')}" 
                             for col in stat_df.columns]
            
            stat_dfs.append(stat_df)
    
    # Combine all statistics
    if not stat_dfs:
        logging.warning("No statistics to add")
        return pd.DataFrame()
    
    # Concatenate and drop columns that are all NaN
    result = pd.concat(stat_dfs, axis=1).dropna(axis=1, how='all')
    return result