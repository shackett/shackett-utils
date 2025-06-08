"""
This module contains functions for regression and factor analysis.
"""

import os
from typing import List, Optional, Tuple, Dict, Union, Any
import logging

from anndata import AnnData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mudata import MuData
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import kstest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from shackett_utils.genomics.adata_utils import get_adata_features_and_data
from ..statistics.multi_model_fitting import fit_parallel_models_formula
from ..statistics.constants import STATISTICS_DEFS, TIDY_DEFS


def adata_model_fitting(
    adata: AnnData,
    formula: str,
    layer: Optional[str] = None,
    model_class: str = 'ols',
    n_jobs: int = -1,
    batch_size: int = 100,
    progress_bar: bool = True,
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
    progress_bar : bool
        Whether to display a progress bar.
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regression statistics for each feature.
    """
        
    # Create feature formula if needed (for GAM)
    feature_formula = None
    if model_class.lower() == 'gam':
        feature_formula = 'y ~ ' + ' + '.join(coefficient_names[1:])  # Skip intercept
    
    # Run parallel model fitting
    return fit_parallel_models(
        X_features=X_features,
        X_model=X_model,
        feature_names=feature_names,
        model_class=model_class,
        formula=feature_formula,
        coefficient_names=coefficient_names,
        n_jobs=n_jobs,
        batch_size=batch_size,
        progress_bar=progress_bar,
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


def plot_pvalue_histograms(
    data: Union[AnnData, MuData],
    regression_key: str = "regression_results",
    output_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    include_stats: bool = True,
    show_ks_test: bool = True,
    fdr_cutoff: float = 0.05,
    terms: Optional[list] = None
) -> Dict[str, Dict[str, plt.Figure]]:
    """
    Generate histograms of p-values and FDR-corrected p-values for all coefficients 
    in regression results stored in AnnData or MuData objects.
    Optionally, only plot for a subset of coefficients/terms if 'terms' is provided.
    
    Parameters
    ----------
    data : Union[anndata.AnnData, mudata.MuData]
        The annotated data matrix or multi-modal data object containing regression results.
    regression_key : str, optional
        Key under which regression results are stored in .uns. Default is "regression_results".
    output_dir : Optional[str], optional
        Directory to save histogram plots. If None, plots are displayed but not saved.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (12, 8).
    include_stats : bool, optional
        Whether to include summary statistics on the plots. Default is True.
    show_ks_test : bool, optional
        Whether to show Kolmogorov-Smirnov test results comparing p-value distribution
        to the uniform distribution. Default is True.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
    terms : list, optional
        List of coefficient names/terms to plot. If None, plot all coefficients.
    
    Returns
    -------
    Dict[str, Dict[str, plt.Figure]]
        Dictionary of generated figures indexed by modality and coefficient.
    """
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Initialize dictionary to store figure objects
    figures: Dict[str, Dict[str, plt.Figure]] = {}
    
    # Set up logger
    logger = logging.getLogger(__name__)
    
    # Function to process a single AnnData object
    def process_anndata(
        adata: AnnData, 
        modality_name: Optional[str] = None,
        terms=terms  # capture from outer scope
    ) -> Dict[str, plt.Figure]:
        # Check if regression results exist
        if regression_key not in adata.uns:
            print(f"No regression results found under key '{regression_key}'"
                  f"{' in modality ' + modality_name if modality_name else ''}.")
            return {}
        
        # Get regression results
        regression_results = adata.uns[regression_key]
        
        # Convert results to DataFrame if needed
        if isinstance(regression_results, dict) and "results" in regression_results:
            results_df = pd.DataFrame(regression_results["results"])
        elif isinstance(regression_results, pd.DataFrame):
            results_df = regression_results
        else:
            print(f"Unexpected format for regression results" 
                  f"{' in modality ' + modality_name if modality_name else ''}.")
            return {}
        
        mod_figures: Dict[str, plt.Figure] = {}
        
        # Get unique coefficients
        coefficients: List[str] = results_df["coefficient"].unique().tolist()
        # If terms is provided, filter coefficients
        if terms is not None:
            # Support string input as a single term
            if isinstance(terms, str):
                terms = [terms]
            missing_terms = [t for t in terms if t not in coefficients]
            found_terms = [t for t in terms if t in coefficients]
            if len(found_terms) == 0:
                raise ValueError("None of the specified terms were found in the regression results.")
            if missing_terms:
                logger.warning(f"The following terms were not found in the regression results and will be skipped: {missing_terms}")
            coefficients = found_terms
        
        for coef in coefficients:
            coef_results = results_df[results_df["coefficient"] == coef]
            fig = _plot_single_term_pvalue_histograms(
                coef_results=coef_results,
                coef=coef,
                modality_name=modality_name,
                include_stats=include_stats,
                show_ks_test=show_ks_test,
                fdr_cutoff=fdr_cutoff,
                output_dir=output_dir,
                figsize=figsize
            )
            mod_figures[coef] = fig
        
        return mod_figures
    
    # Check if input is MuData or AnnData
    if hasattr(data, 'mod'):  # MuData object
        for modality in data.mod:
            print(f"Processing {modality}...")
            mod_figures = process_anndata(data[modality], modality_name=modality)
            if mod_figures:
                figures[modality] = mod_figures
    else:  # AnnData object
        figures['anndata'] = process_anndata(data)
    
    # Display figures if not saving
    if output_dir is None:
        for modality in figures:
            for coef in figures[modality]:
                plt.figure(figures[modality][coef].number)
                plt.show()
    
    print("Histogram generation complete.")
    return figures


def _plot_single_term_pvalue_histograms(
    coef_results: pd.DataFrame,
    coef: str,
    modality_name: Optional[str],
    include_stats: bool,
    show_ks_test: bool,
    fdr_cutoff: float,
    output_dir: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """
    Utility to plot p-value histograms and QQ plots for a single coefficient/term.
    Returns the matplotlib Figure.
    """
    n_panels = 3 if STATISTICS_DEFS.Q_VALUE in coef_results.columns else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0]*n_panels/2, figsize[1]))
    fig.suptitle(f"P-value Distribution for Coefficient: {coef}" +
                 (f" in {modality_name}" if modality_name else ""),
                 fontsize=16)

    # Raw p-values
    sns.histplot(coef_results[STATISTICS_DEFS.P_VALUE], bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Raw P-values")
    axes[0].set_xlabel("P-value")
    axes[0].set_ylabel("Count")

    # Add reference uniform distribution line
    x = np.linspace(0, 1, 100)
    y = len(coef_results) * 0.05  # Adjust height based on data
    axes[0].plot(x, [y] * 100, 'r--', label='Uniform Distribution')
    axes[0].legend()

    # Add summary statistics
    if include_stats:
        avg_p = coef_results[STATISTICS_DEFS.P_VALUE].mean()
        median_p = coef_results[STATISTICS_DEFS.P_VALUE].median()
        sig_count = sum(coef_results[STATISTICS_DEFS.P_VALUE] < 0.05)
        sig_pct = 100 * sig_count / len(coef_results)
        stats_text = f"Mean: {avg_p:.4f}\nMedian: {median_p:.4f}\n"
        stats_text += f"Significant: {sig_count}/{len(coef_results)} ({sig_pct:.1f}%)"
        axes[0].text(0.05, 0.95, stats_text,
                     transform=axes[0].transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # QQ plot of p-values
    pvals = np.sort(coef_results[STATISTICS_DEFS.P_VALUE])
    expected = np.linspace(0, 1, len(pvals) + 1)[1:]
    axes[1].scatter(expected, pvals, alpha=0.5)
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_title("P-value QQ Plot")
    axes[1].set_xlabel("Expected P-value")
    axes[1].set_ylabel("Observed P-value")

    # FDR-corrected p-values (if available)
    if n_panels == 3:
        sns.histplot(coef_results[STATISTICS_DEFS.Q_VALUE], bins=20, kde=True, ax=axes[2])
        axes[2].set_title("FDR-corrected P-values (Benjamini-Hochberg)")
        axes[2].set_xlabel("Adjusted P-value")
        axes[2].set_ylabel("Count")
        axes[2].plot(x, [y] * 100, 'r--', label='Uniform Distribution')
        axes[2].legend()
        if include_stats:
            avg_fdr = coef_results[STATISTICS_DEFS.Q_VALUE].mean()
            median_fdr = coef_results[STATISTICS_DEFS.Q_VALUE].median()
            sig_count_fdr = sum(coef_results[STATISTICS_DEFS.Q_VALUE] < fdr_cutoff)
            sig_pct_fdr = 100 * sig_count_fdr / len(coef_results)
            stats_text = f"Mean: {avg_fdr:.4f}\nMedian: {median_fdr:.4f}\n"
            stats_text += f"Significant: {sig_count_fdr}/{len(coef_results)} ({sig_pct_fdr:.1f}%)"
            axes[2].text(0.05, 0.95, stats_text,
                         transform=axes[2].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # KS test
    if show_ks_test:
        ks_stat, ks_pval = kstest(coef_results[STATISTICS_DEFS.P_VALUE], 'uniform')
        fig.text(0.5, 0.01,
                 f"Kolmogorov-Smirnov test against uniform distribution: "
                 f"statistic={ks_stat:.4f}, p-value={ks_pval:.4f}",
                 ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if output_dir is provided
    if output_dir is not None:
        mod_dir = os.path.join(output_dir, modality_name if modality_name else "anndata")
        os.makedirs(mod_dir, exist_ok=True)
        safe_coef = coef.replace(" ", "_").replace("/", "_").replace("\\", "_")
        file_path = os.path.join(mod_dir, f"pvalue_hist_{safe_coef}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to {file_path}")

    return fig