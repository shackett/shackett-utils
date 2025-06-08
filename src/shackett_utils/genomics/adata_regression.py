"""
This module contains functions for regression and factor analysis.
"""

import os
from typing import List, Optional, Tuple, Dict, Union, Any
import logging

import pandas as pd

from anndata import AnnData

from shackett_utils.genomics import adata_utils
from shackett_utils.statistics import multi_model_fitting
from shackett_utils.statistics.constants import STATISTICS_DEFS, TIDY_DEFS

def adata_model_fitting(
    adata: AnnData,
    formula: str,
    layer: Optional[str] = None,
    model_class: str = 'ols',
    n_jobs: int = -1,
    batch_size: int = 100,
    model_name: Optional[str] = None,
    progress_bar: bool = True,
    fdr_control: bool = True,
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
    model_class : str
        Type of model to fit ('ols', 'gam', etc.)
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
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regression statistics for each feature.
    """
        
    feature_names, data_matrix = adata_utils.get_adata_features_and_data(adata, layer = layer)

    return multi_model_fitting.fit_parallel_models_formula(
        X_features = data_matrix,
        data = adata.obs,
        feature_names = feature_names,
        formula = formula,
        model_class = model_class,
        n_jobs = n_jobs,
        batch_size = batch_size,
        model_name = model_name,
        fdr_control = fdr_control,
        progress_bar = progress_bar,
        **model_kwargs
    )


def add_regression_results_to_anndata(
    adata: AnnData,
    results_df: pd.DataFrame,
    key_added: str = "regression_results",
    fdr_cutoff: float = 0.05,
    inplace: bool = True,
    effect_only_for_significant: bool = False,  # New parameter
    add_all_stats: bool = True  # Add all statistics columns
) -> Optional[AnnData]:
    """
    Add regression results dataframe to an AnnData object in a structured way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    results_df : pd.DataFrame
        DataFrame with regression results as produced by apply_regression_per_feature().
    key_added : str, optional
        Key under which to add the results in adata.uns.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
    inplace : bool, optional
        If True, modify adata inplace. Otherwise return a copy. Default is True.
    effect_only_for_significant : bool, optional
        If True, effect sizes are only stored for significant features.
        If False, effect sizes are stored for all features. Default is False.
    add_all_stats : bool, optional
        If True, add all available statistics (p-values, q-values, t-statistics, std errors)
        to the adata.var dataframe. Default is True.
        
    Returns
    -------
    Optional[anndata.AnnData]
        Depends on `inplace` parameter. If `inplace=True`, returns None,
        otherwise returns a modified copy of the AnnData object.
    """
    if not inplace:
        adata = adata.copy()
    
    # Make a copy to avoid modifying the original results
    results = results_df.copy()
    
    # Store in adata.uns (serialization-friendly: only DataFrame, not nested dict)
    adata.uns[key_added] = results
    
    # Also annotate significant associations in var
    # Create a significance mask for each coefficient
    for coef in results["coefficient"].unique():
        coef_safe = coef.replace(" ", "_").replace("-", "_")
        
        # Determine the p-value column to use (FDR-corrected if available)
        p_col = STATISTICS_DEFS.Q_VALUE if STATISTICS_DEFS.Q_VALUE in results.columns else STATISTICS_DEFS.P_VALUE
        
        # Get significant features for this coefficient
        sig_mask = (results["coefficient"] == coef) & (results[p_col] < fdr_cutoff)
        sig_features = results.loc[sig_mask, "feature"].unique()
        
        # Create a binary indicator in adata.var
        adata.var[f"sig_{coef_safe}"] = adata.var_names.isin(sig_features)
        
        # Filter for the current coefficient
        coef_results = results[results["coefficient"] == coef]
        
        # Create effect size dictionary for all features
        effect_dict: Dict[str, float] = {}
        for _, row in coef_results.iterrows():
            effect_dict[row[STATISTICS_DEFS.FEATURE_NAME]] = row[TIDY_DEFS.ESTIMATE]
        
        # Create effect size column for all features or only significant ones
        if effect_only_for_significant:
            # Original behavior: add effect sizes only for significant features
            adata.var[f"effect_{coef_safe}"] = pd.Series(
                [effect_dict.get(f, np.nan) if f in sig_features else np.nan for f in adata.var_names], 
                index=adata.var_names
            )
        else:
            # New behavior: add effect sizes for all features
            adata.var[f"effect_{coef_safe}"] = pd.Series(
                [effect_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
        
        # Add all statistics as separate columns for all features
        if add_all_stats:
            # Raw p-values
            pval_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                pval_dict[row[STATISTICS_DEFS.FEATURE_NAME]] = row[STATISTICS_DEFS.P_VALUE]
                
            adata.var[f"pval_{coef_safe}"] = pd.Series(
                [pval_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
            
            # FDR-corrected p-values (q-values)
            if STATISTICS_DEFS.Q_VALUE in coef_results.columns:
                qval_dict: Dict[str, float] = {}
                for _, row in coef_results.iterrows():
                    qval_dict[row[STATISTICS_DEFS.FEATURE_NAME]] = row[STATISTICS_DEFS.Q_VALUE]
                    
                adata.var[f"qval_{coef_safe}"] = pd.Series(
                    [qval_dict.get(f, np.nan) for f in adata.var_names], 
                    index=adata.var_names
                )
            
            # T-statistics
            tstat_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                tstat_dict[row[STATISTICS_DEFS.FEATURE_NAME]] = row["statistic"]
                
            adata.var[f"tstat_{coef_safe}"] = pd.Series(
                [tstat_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
            
            # Standard errors
            stderr_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                stderr_dict[row[STATISTICS_DEFS.FEATURE_NAME]] = row[TIDY_DEFS.STD_ERROR]
                
            adata.var[f"stderr_{coef_safe}"] = pd.Series(
                [stderr_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
        else:
            # Just add the selected p-value column as before
            pval_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                pval_dict[row["feature"]] = row[p_col]
                
            adata.var[f"pval_{coef_safe}"] = pd.Series(
                [pval_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
    
    return None if inplace else adata
