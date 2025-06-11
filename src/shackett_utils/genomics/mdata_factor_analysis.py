"""
MOFA Factor Analysis Utilities

This module provides functions for running and analyzing Multi-Omics Factor Analysis (MOFA)
with different numbers of factors.
"""

from datetime import datetime
import functools
import io
import os
import sys
from typing import Dict, List, Tuple, Optional, Iterable, Any, Union
import json

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
from mudata import MuData
import muon
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from shackett_utils.genomics.mdata_utils import create_minimal_mudata
from shackett_utils.statistics import multi_model_fitting
from shackett_utils.statistics.constants import STATISTICS_DEFS, FDR_METHODS_DEFS, TIDY_DEFS
from shackett_utils.genomics.constants import MOFA_DEFS, FACTOR_REGRESSIONS_KEY, FACTOR_NAME_PATTERN, VARIANCE_METRICS_DEFS, K_SELECTION_CRITERIA, AUTO_K_SELECTION_CRITERIA
from shackett_utils.utils.decorators import suppress_stdout
from shackett_utils.utils.time import get_timestamp

# Configure MuData to use new update behavior
md.set_options(pull_on_update=False)

FACTOR_REGRESSION_STR = "mofa_regression_{}"

@suppress_stdout
def _mofa(
    mdata: md.MuData,
    *args,
    **kwargs
) -> None:
    """
    Helper function to run MOFA with suppressed output.
    Just wraps muon.tl.mofa to suppress its verbose output.
    
    Parameters
    ----------
    mdata : md.MuData
        Multi-modal data object to be updated in-place
    *args
        Additional positional arguments passed to muon.tl.mofa
    **kwargs
        Additional keyword arguments passed to muon.tl.mofa
    """
    muon.tl.mofa(mdata, *args, **kwargs)

def run_mofa_factor_scan(
    mdata: md.MuData,
    factor_range: Iterable[int] = range(5, 31, 5),
    use_layer: Optional[str] = "log2_centered",
    use_var: Optional[str] = None,
    seed: int = 42,
    models_dir: str = "mofa_models",
    overwrite: bool = False,
) -> Dict[int, Dict[str, str]]:
    """
    Run MOFA with different numbers of factors and save variance metrics summaries.
    
    Parameters
    ----------
    mdata : md.MuData
        Multi-modal data object
    factor_range : Iterable[int], optional
        Range of factor numbers to try, by default range(5, 31, 5)
    use_layer : Optional[str], optional
        Layer to use for MOFA, by default "log2_centered"
    use_var : Optional[str], optional
        Variable subset to use, by default None
    seed : int, optional
        Random seed for reproducibility, by default 42
    models_dir : str, optional
        Directory to save summaries, by default "mofa_models"
    overwrite : bool, optional
        Whether to overwrite existing results, by default False
        
    Returns
    -------
    Dict[int, Dict[str, str]]
        Dictionary mapping number of factors to summary file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Store results
    results = {}
    
    for n_factors in factor_range:
        # Set up file paths
        temp_model_file = os.path.join(models_dir, f"temp_model_{n_factors}.h5")
        summary_file = os.path.join(models_dir, f"summary_{n_factors}.json")
        
        # Skip if summary exists and not overwriting
        if not overwrite and os.path.exists(summary_file):
            results[n_factors] = {"summary_file": summary_file}
            continue
        
        try:
            # Create a minimal copy of the data for this iteration
            model_data = create_minimal_mudata(
                mdata,
                include_layers=[use_layer] if use_layer else None
            )
            
            # Run MOFA in-place
            _mofa(
                model_data,
                n_factors=n_factors,
                use_layer=use_layer,
                use_var=use_var,
                outfile=temp_model_file,
                seed=seed
            )
            
            # Calculate metrics directly from the fitted model
            metrics = _calculate_mofa_variance_metrics(model_data, use_layer=use_layer)
            
            # Save metrics to JSON
            with open(summary_file, 'w') as f:
                json.dump(metrics, f)
            
            # Store summary file path
            results[n_factors] = {"summary_file": summary_file}
            print(f"[{get_timestamp()}] Saved metrics for {n_factors} factors.")
                
        except Exception as e:
            print(f"[{get_timestamp()}] Error with {n_factors} factors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results


def _calculate_mofa_variance_metrics(
    mdata: md.MuData,
    use_layer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate variance metrics from MOFA results.
    
    Parameters
    ----------
    mdata : md.MuData
        Multi-modal data object with MOFA results
    use_layer : Optional[str]
        Layer to use for variance calculation, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with variance metrics:
        - total_variance: Total variance explained across all modalities
        - modality_variance: Dict mapping modality names to their variance explained
        - raw_tss: Dict mapping modality names to their total sum of squares
        - raw_ess: Dict mapping modality names to their explained sum of squares
    """
    metrics = {
        "total_variance": 0.0,
        "modality_variance": {},
        "raw_tss": {},
        "raw_ess": {}
    }
    
    factors = mdata.obsm[MOFA_DEFS.X_MOFA]
    loadings = mdata.varm[MOFA_DEFS.LFS]
    
    # Calculate TSS and ESS for each modality
    total_ss = {}
    ess = {}
    
    for modality, mod_data in mdata.mod.items():
        # Get original data
        X = mod_data.layers[use_layer] if use_layer in mod_data.layers else mod_data.X
        total_ss[modality] = np.sum(X**2)
        
        # Get the var_names for this modality
        mod_vars = mod_data.var_names
        
        # Create boolean mask for this modality's variables
        var_mask = np.isin(mdata.var_names, mod_vars)
        
        # Extract loadings for this modality using the mask
        mod_loadings = loadings[var_mask]
        
        # Calculate explained sum of squares
        # Reconstruct data using factors and loadings
        X_reconstructed = factors @ mod_loadings.T
        ess[modality] = np.sum(X_reconstructed**2)
    
    # Calculate variance explained per modality
    modality_variance = {
        modality: (ess[modality] / total_ss[modality]) * 100
        for modality in total_ss.keys()
    }
    
    # Calculate total variance as sum of ESS divided by sum of TSS
    total_ess = sum(ess.values())
    total_tss = sum(total_ss.values())
    total_variance = (total_ess / total_tss) * 100
    
    return {
        VARIANCE_METRICS_DEFS.TOTAL_VARIANCE: total_variance,
        VARIANCE_METRICS_DEFS.MODALITY_VARIANCE: modality_variance,
        VARIANCE_METRICS_DEFS.RAW_TSS: total_ss,
        VARIANCE_METRICS_DEFS.RAW_ESS: ess
    }


def calculate_variance_metrics(
    factor_results: Dict[int, Dict[str, str]],
    mdata: Optional[md.MuData] = None,  # Made optional since we don't need it anymore
    use_layer: Optional[str] = None,  # Kept for backwards compatibility
) -> Dict[int, Dict[str, float]]:
    """
    Load variance metrics for each factor model from JSON summaries.
    
    Parameters
    ----------
    factor_results : Dict[int, Dict[str, str]]
        Dictionary mapping number of factors to summary file paths
    mdata : Optional[md.MuData], optional
        Not used anymore, kept for backwards compatibility
    use_layer : Optional[str], optional
        Not used anymore, kept for backwards compatibility
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Dictionary mapping number of factors to variance metrics
    """
    metrics = {}
    
    for n_factors, paths in factor_results.items():
        try:
            # Load metrics from JSON
            summary_file = paths["summary_file"]
            with open(summary_file, 'r') as f:
                metrics[n_factors] = json.load(f)
            print(f"Loaded metrics for {n_factors} factors. Total variance: {metrics[n_factors]['total_variance']:.2f}%")
            
        except Exception as e:
            print(f"Error loading metrics for {n_factors} factors: {str(e)}")
            import traceback
            traceback.print_exc()
            # Initialize with empty metrics to avoid KeyError
            metrics[n_factors] = {
                VARIANCE_METRICS_DEFS.TOTAL_VARIANCE: 0.0,
                VARIANCE_METRICS_DEFS.MODALITY_VARIANCE: {mod: 0.0 for mod in (mdata.mod.keys() if mdata else [])},
                VARIANCE_METRICS_DEFS.RAW_TSS: {},
                VARIANCE_METRICS_DEFS.RAW_ESS: {}
            }
    
    return metrics


def _determine_optimal_factors(
    metrics: Dict[int, Dict[str, Any]], 
    criterion: str = K_SELECTION_CRITERIA.ELBOW, 
    threshold: float = 0.01
) -> Optional[int]:
    """
    Determine the optimal number of factors based on variance metrics.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    criterion : str
        Method to use for determining optimal factors
        ('elbow', 'threshold', or 'balanced'), by default "elbow"
    threshold : float
        Threshold for marginal improvement (used if criterion='threshold'),
        by default 0.01
        
    Returns
    -------
    Optional[int]
        Optimal number of factors or None if it cannot be determined
        
    Examples
    --------
    >>> # Determine optimal factors using different criteria
    >>> optimal_elbow = _determine_optimal_factors(metrics, criterion='elbow')
    >>> optimal_threshold = _determine_optimal_factors(metrics, criterion='threshold', threshold=0.01)
    >>> optimal_balanced = _determine_optimal_factors(metrics, criterion='balanced')
    >>> print(f"Optimal factors: elbow={optimal_elbow}, threshold={optimal_threshold}, balanced={optimal_balanced}")
    """
    # Extract factors and variance values
    factors = sorted([k for k in metrics.keys() if metrics[k][VARIANCE_METRICS_DEFS.TOTAL_VARIANCE] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        return None
        
    variance = [metrics[k][VARIANCE_METRICS_DEFS.TOTAL_VARIANCE] for k in factors]

    if len(factors) <= 1:
        return factors[0] if factors else None

    # Calculate marginal improvements
    diffs = np.diff(variance)

    if criterion == K_SELECTION_CRITERIA.ELBOW:
        return _determine_optimal_elbow(factors, variance, diffs)
    elif criterion == K_SELECTION_CRITERIA.THRESHOLD:
        return _determine_optimal_threshold(factors, diffs, threshold)
    elif criterion == K_SELECTION_CRITERIA.BALANCED:
        return _determine_optimal_balanced(factors, variance)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use {', '.join(AUTO_K_SELECTION_CRITERIA)}.")


def _visualize_variance_explained(
    metrics: Dict[int, Dict[str, Any]],
    optimal_factors: Optional[Dict[str, int]] = None,
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a plot of total variance explained by number of factors.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    optimal_factors : Optional[Dict[str, int]]
        Dictionary mapping method names to optimal factor values, by default None
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size, by default (12, 8)
    save_path : Optional[str]
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    # Extract factors and variance values
    factors = sorted([k for k in metrics.keys() if metrics[k][VARIANCE_METRICS_DEFS.TOTAL_VARIANCE] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid variance metrics available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
        
    variance = [metrics[k][VARIANCE_METRICS_DEFS.TOTAL_VARIANCE] for k in factors]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot total variance explained
    plt.plot(factors, variance, "o-", linewidth=2, color="blue", markersize=8)
    
    # Highlight optimal factors if provided
    colors = {K_SELECTION_CRITERIA.ELBOW: "red", K_SELECTION_CRITERIA.THRESHOLD: "green", K_SELECTION_CRITERIA.BALANCED: "purple"}
    
    if optimal_factors:
        for method, n_factors in optimal_factors.items():
            if n_factors in factors:
                color = colors.get(method, "gray")
                idx = factors.index(n_factors)
                plt.axvline(
                    x=n_factors, 
                    color=color, 
                    linestyle="--", 
                    label=f"{method.capitalize()}: {n_factors} factors"
                )
                plt.scatter(
                    [n_factors], 
                    [variance[idx]], 
                    marker="*", 
                    s=200, 
                    color=color, 
                    zorder=5
                )
    
    # Highlight user-specified factors if provided
    if user_factors and user_factors in factors:
        idx = factors.index(user_factors)
        plt.axvline(
            x=user_factors, 
            color="orange", 
            linestyle="--", 
            label=f"User-specified: {user_factors} factors"
        )
        plt.scatter(
            [user_factors], 
            [variance[idx]], 
            marker="*", 
            s=200, 
            color="orange", 
            zorder=5
        )
    
    plt.xlabel("Number of factors")
    plt.ylabel("Total variance explained (%)")
    plt.title("MOFA performance vs. number of factors")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Set y-axis to percentage scale
    plt.ylim(0, min(100, max(variance) * 1.1))
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig

def plot_mofa_factor_histograms(
    mdata,
    factors: Optional[Union[int, List[int]]] = None,
    n_bins: int = 30,
    n_cols: int = 3,
    figsize: Tuple[float, float] = None,
    kde: bool = True,
    color: str = 'steelblue',
    return_fig: bool = False
):
    """
    Plot histograms of MOFA factor values.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with MOFA results
    factors : int or list of ints, optional
        Factors to plot. If None, all factors are plotted.
    n_bins : int, default=30
        Number of bins for histograms
    n_cols : int, default=3
        Number of columns in the grid plot
    figsize : tuple, optional
        Figure size
    kde : bool, default=True
        Whether to overlay a kernel density estimate
    color : str, default='steelblue'
        Color for the histogram bars
    return_fig : bool, default=False
        Whether to return the figure object
    
    Returns
    -------
    matplotlib.figure.Figure, optional
        Figure object, if return_fig is True
    """
    # Extract MOFA factors
    X_mofa = mdata.obsm[MOFA_DEFS.X_MOFA]
    
    # Get number of factors
    n_factors = X_mofa.shape[1]
    
    # Select factors to plot
    if factors is None:
        factors = list(range(n_factors))
    elif isinstance(factors, int):
        factors = [factors]
    
    # Create a DataFrame with factor values
    df = pd.DataFrame(
        X_mofa[:, factors],
        index=mdata.obs_names,
        columns=[_factor_idx_to_name(i) for i in factors]
    )
    
    # Calculate number of rows needed
    n_factors_to_plot = len(factors)
    n_rows = int(np.ceil(n_factors_to_plot / n_cols))
    
    # Set up the figure
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_factors_to_plot == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Plot each factor
    for i, factor_idx in enumerate(factors):
        if i < len(axes):
            ax = axes[i]
            factor_name = _factor_idx_to_name(factor_idx)
            
            # Get factor values
            factor_values = df[factor_name]
            
            # Plot histogram
            sns.histplot(
                factor_values,
                bins=n_bins,
                kde=kde,
                color=color,
                ax=ax
            )
            
            # Set labels and title
            ax.set_xlabel("Factor Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {factor_name}")
            
            # Add vertical line at mean
            mean_val = factor_values.mean()
            ax.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.2f}')
            
            # Add vertical line at median
            median_val = factor_values.median()
            ax.axvline(median_val, color='green', linestyle=':', 
                       label=f'Median: {median_val:.2f}')
            
            # Add legend
            ax.legend(fontsize='small')
    
    # Remove any unused subplots
    for i in range(n_factors_to_plot, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()


def visualize_factor_scan_results(
    metrics: Dict[int, Dict[str, Any]],
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Visualize the results of a MOFA factor scan with highlighted optimal factor selections.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size for individual plots, by default (12, 8)
    save_dir : Optional[str]
        Directory to save figures, by default None
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of created figures
        
    Examples
    --------
    >>> # Visualize factor scan results with user-specified factors
    >>> figures = visualize_factor_scan_results(metrics, user_factors=15)
    >>> # Display the figures
    >>> for name, fig in figures.items():
    ...     plt.figure(fig.number)
    ...     plt.show()
    """
    # Calculate optimal factors using different criteria
    optimal_factors = {
        K_SELECTION_CRITERIA.ELBOW: _determine_optimal_factors(metrics, criterion=K_SELECTION_CRITERIA.ELBOW),
        K_SELECTION_CRITERIA.THRESHOLD: _determine_optimal_factors(metrics, criterion=K_SELECTION_CRITERIA.THRESHOLD, threshold=0.01),
        K_SELECTION_CRITERIA.BALANCED: _determine_optimal_factors(metrics, criterion=K_SELECTION_CRITERIA.BALANCED)
    }
    
    # Filter out None values
    optimal_factors = {k: v for k, v in optimal_factors.items() if v is not None}
    
    # Create save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figures
    figures = {}
    
    # 1. Total variance explained plot
    save_path = os.path.join(save_dir, "total_variance.png") if save_dir else None
    figures["total_variance"] = _visualize_variance_explained(
        metrics, 
        optimal_factors, 
        user_factors, 
        figsize=figsize, 
        save_path=save_path
    )
    
    # 2. Marginal improvement plot
    save_path = os.path.join(save_dir, "marginal_improvement.png") if save_dir else None
    figures["marginal_improvement"] = _visualize_marginal_improvement(
        metrics, 
        optimal_factors, 
        user_factors, 
        figsize=figsize, 
        save_path=save_path
    )
    
    # 3. Modality-specific variance plot
    save_path = os.path.join(save_dir, "modality_variance.png") if save_dir else None
    figures["modality_variance"] = _visualize_modality_variance(
        metrics, 
        optimal_factors, 
        user_factors, 
        figsize=figsize, 
        save_path=save_path
    )
    
    # Print summary of optimal factors
    print("Optimal number of factors based on different criteria:")
    for method, n_factors in optimal_factors.items():
        print(f"  {method.capitalize()} method: {n_factors} factors")
    if user_factors:
        print(f"  User-specified: {user_factors} factors")
    
    return figures

def regress_factors_with_formula(
    mdata: MuData,
    formula: str,
    factors: Optional[Union[int, List[int]]] = None,
    modality: Optional[str] = None,
    n_jobs: int = 1,
    fdr_control: bool = True,
    fdr_method: str = FDR_METHODS_DEFS.BH,
    model_class: Optional[str] = None,
    **model_kwargs
) -> pd.DataFrame:
    """
    Regress factors against variables using a formula interface.
    Uses the multi-model fitting approach for efficient parallel processing.
    
    Parameters
    ----------
    mdata : MuData
        Multi-modal data object containing factor analysis results.
    formula : str
        Formula for regression (e.g. "~ condition + batch").
        Will be validated to ensure 'y ~ ...' format.
    factors : Optional[Union[int, List[int]]], optional
        Specific factors to analyze. If None, analyze all factors.
    modality : str, optional
        Modality to analyze. Default of None will choose the first modality. This won't usually matter too much because we are only getting sample attributes for the regression; the acutal factors are still informed by all modalities. 
    n_jobs : int, optional
        Number of parallel jobs to run. Default is 1.
    fdr_control : bool, optional
        Whether to apply FDR correction. Default is True.
    fdr_method : str, optional
        Method for FDR correction. Default is "fdr_bh".
    model_class : Optional[str], optional
        Type of model to fit ('ols' or 'gam'). If None, will be detected from formula.
    **model_kwargs
        Additional arguments passed to model fitting.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regression results.
        
    Notes
    -----
    The function uses the multi-model fitting approach which:
    1. Efficiently processes multiple factors in parallel
    2. Handles missing values automatically
    3. Provides proper FDR control across all tests
    4. Supports both OLS and GAM models
    """
    
    # Get factor data
    if modality is None:
        modality = mdata.mod_names[0]

    if modality not in mdata.mod:
        raise ValueError(f"Modality {modality} not found in MuData object")
    
    # Extract MOFA factors
    X_mofa = mdata.obsm[MOFA_DEFS.X_MOFA]
    
    # Get number of factors
    n_factors = X_mofa.shape[1]
    
    # Select factors to analyze
    if factors is None:
        factors = list(range(n_factors))
    elif isinstance(factors, int):
        factors = [factors]
    
    # Create feature matrix (factors are our features)
    X_features = X_mofa[:, factors]
    
    # Create feature names
    feature_names = [_factor_idx_to_name(factor_idx) for factor_idx in factors]
    
    # Get observation data from the specified modality
    data = mdata[modality].obs.copy()
    
    # Run parallel regression using multi_model_fitting
    results_df = multi_model_fitting.fit_parallel_models_formula(
        X_features=X_features,
        data=data,
        feature_names=feature_names,
        formula=formula,
        model_class=model_class,
        n_jobs=n_jobs,
        fdr_control=fdr_control,
        fdr_method=fdr_method,
        **model_kwargs
    ).rename(columns={STATISTICS_DEFS.FEATURE_NAME: MOFA_DEFS.FACTOR_NAME})
    
    return results_df


def summarize_factor_regression(
    regression_results: pd.DataFrame,
    alpha: float = 0.05,
    group_by_factor: bool = False
) -> pd.DataFrame:
    """
    Create a summary table of regression results.
    
    Parameters
    ----------
    regression_results : pd.DataFrame
        DataFrame with regression results from regress_factors_with_formula()
    alpha : float, default=0.05
        Significance threshold for q-values
    group_by_factor : bool, default=False
        If True, organize results by factor; otherwise by term
        
    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    if regression_results.empty:
        print("No regression results to summarize.")
        return pd.DataFrame()
    
    # Use q_value if available, otherwise use p_value
    p_col = TIDY_DEFS.Q_VALUE if TIDY_DEFS.Q_VALUE in regression_results.columns else TIDY_DEFS.P_VALUE
    
    # Create significance mask
    results_df = regression_results.copy()
    results_df['significant'] = results_df[p_col] < alpha
    
    # Filter for significant results
    sig_results = results_df[results_df['significant']].copy()
    
    if sig_results.empty:
        print(f"No significant associations found at alpha = {alpha}")
        return pd.DataFrame()
    
    # Select relevant columns
    # TODO - code smell
    summary_cols = [
        MOFA_DEFS.FACTOR_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.STD_ERROR,
        TIDY_DEFS.P_VALUE, TIDY_DEFS.Q_VALUE if TIDY_DEFS.Q_VALUE in results_df.columns else TIDY_DEFS.P_VALUE,
        TIDY_DEFS.RSQUARED, TIDY_DEFS.NOBS
    ]
    
    # Check for excluded columns
    available_cols = []
    for col in summary_cols:
        if col in sig_results.columns or (col == TIDY_DEFS.Q_VALUE and TIDY_DEFS.Q_VALUE not in sig_results.columns):
            if col != TIDY_DEFS.Q_VALUE or TIDY_DEFS.Q_VALUE in sig_results.columns:
                available_cols.append(col)
    
    summary_df = sig_results[available_cols].copy()
    
    # Group by factor or term depending on preference
    # Always sort by p-value and q-value (if available), then by factor or term
    sort_cols = []
    if TIDY_DEFS.P_VALUE in summary_df.columns:
        sort_cols.append(TIDY_DEFS.P_VALUE)
    if TIDY_DEFS.Q_VALUE in summary_df.columns:
        sort_cols.append(TIDY_DEFS.Q_VALUE)
    # Add grouping column
    if group_by_factor:
        sort_cols = [MOFA_DEFS.FACTOR_NAME] + sort_cols
    else:
        sort_cols = [TIDY_DEFS.TERM] + sort_cols
    summary_df = summary_df.sort_values(sort_cols, ascending=True)

    # Format numeric columns (after sorting)
    for col in [TIDY_DEFS.ESTIMATE, TIDY_DEFS.STD_ERROR]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.3f}'.format)

    for col in [TIDY_DEFS.P_VALUE, TIDY_DEFS.Q_VALUE]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.2e}'.format)

    for col in [TIDY_DEFS.RSQUARED]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.3f}'.format)
    
    return summary_df


def factor_term_scatterplot(
    mdata: MuData,
    factor: Union[int, str],
    formula_name: str,
    term: Optional[str] = None
) -> None:
    """
    Scatterplot of a MOFA factor against a sample attribute, with regression line.

    Parameters
    ----------
    mdata : MuData
        MuData object containing MOFA results and sample attributes.
    factor : int or str
        Factor index (int) [starting from 0] or name (str) [starting from 1] to plot.
    formula_name : str
        Name of the regression formula used in the analysis.
    term : str or None, optional
        Name of the sample attribute to plot on the x-axis. If None, uses formula_name.

    Returns
    -------
    None
    """
    # If term is None, use formula_name as the term
    if term is None:
        term = formula_name
    
    df = _factor_term_sample_attrs(mdata, factor, formula_name, term)
    
    # Get the factor name
    if isinstance(factor, int):
        factor_name = _factor_idx_to_name(factor)
    else:
        factor_name = factor
    
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x="sample_attribute", y="factor_values", scatter_kws={'alpha':0.7})
    plt.xlabel(term)
    plt.ylabel(factor_name)
    # Only show (formula_name) in title if term != formula_name
    if term == formula_name:
        title = f"{factor_name} vs {term}"
    else:
        title = f"{factor_name} vs {term} ({formula_name})"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def _factor_term_sample_attrs(
    mdata: MuData,
    factor: Union[int, str],
    formula_name: str,
    term: str
) -> pd.DataFrame:
    """
    Helper to extract a DataFrame of sample attribute and factor values for plotting.

    Parameters
    ----------
    mdata : MuData
        MuData object containing MOFA results and sample attributes.
    factor : int or str
        Factor index (int) [starting from 0] or name (str) [starting from 1] to extract.
    formula_name : str
        Name of the regression formula used in the analysis.
    term : str
        Name of the sample attribute to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'sample_attribute' and 'factor_values'.
    """
    # Check if regression results exist
    if FACTOR_REGRESSIONS_KEY not in mdata.uns:
        raise KeyError(f"No regression results found in mdata.uns under key '{FACTOR_REGRESSIONS_KEY}'")

    # Get regression results
    factor_regressions = mdata.uns[FACTOR_REGRESSIONS_KEY]
    
    # Filter by formula name
    formula_results = factor_regressions[factor_regressions[MOFA_DEFS.FORMULA_NAME] == formula_name]
    if len(formula_results) == 0:
        raise ValueError(f"No regression results found for formula '{formula_name}'")
    
    # Convert factor to name if it's an index
    if isinstance(factor, int):
        factor_name = _factor_idx_to_name(factor)
    else:
        factor_name = factor

    # Check if factor exists in results for this formula
    if not (formula_results[MOFA_DEFS.FACTOR_NAME] == factor_name).any():
        raise ValueError(f"Factor '{factor_name}' not found in regression results for formula '{formula_name}'")

    # Get factor index from name
    factor_idx = _factor_name_to_idx(factor_name)

    # Get factor values from obsm
    factor_values = mdata.obsm[MOFA_DEFS.X_MOFA][:, factor_idx]

    # Get sample attribute from first modality (they should all be the same)
    sample_attribute = mdata[mdata.mod_names[0]].obs

    # Check for the term in the sample attribute
    if term in sample_attribute.columns:
        sample_attribute = sample_attribute[term]
    else:
        raise ValueError(f"Term '{term}' not found in sample attribute")

    df = pd.DataFrame({
        "sample_attribute": sample_attribute,
        "factor_values": factor_values
    })

    return df

# Additional utility functions with underscore prefix

def _determine_optimal_elbow(
    factors: List[int], 
    variance: List[float], 
    diffs: np.ndarray
) -> int:
    """Determine optimal factors using the elbow method."""
    # Elbow method: find where the rate of change of the differences is greatest
    if len(diffs) <= 1:
        return factors[0]

    # Calculate the acceleration (approximating second derivative)
    acceleration = np.diff(diffs)
    acceleration = np.append(acceleration, acceleration[-1])  # Pad to match length

    # Find the point of maximum acceleration
    elbow_idx = np.argmax(np.abs(acceleration)) + 1  # +1 because we're working with diffs

    return factors[elbow_idx]


def _determine_optimal_threshold(
    factors: List[int], 
    diffs: np.ndarray, 
    threshold: float
) -> int:
    """Determine optimal factors using the threshold method."""
    # Threshold method: find first point where marginal improvement falls below threshold
    for i, diff in enumerate(diffs):
        if diff < threshold:
            return factors[i]

    # If all improvements are above threshold, return the maximum
    return factors[-1]


def _determine_optimal_balanced(
    factors: List[int], 
    variance: List[float]
) -> int:
    """Determine optimal factors using a balanced approach."""
    # Balanced approach: consider both variance explained and number of factors
    # Normalize factors and variance to [0, 1]
    norm_factors = (np.array(factors) - min(factors)) / (max(factors) - min(factors))
    norm_variance = (np.array(variance) - min(variance)) / (max(variance) - min(variance))

    # Calculate a score balancing parsimony and variance explained
    # Higher variance is better, fewer factors is better
    scores = norm_variance - 0.5 * norm_factors

    # Find the factor value with the highest score
    best_idx = np.argmax(scores)

    return factors[best_idx]


def _visualize_marginal_improvement(
    metrics: Dict[int, Dict[str, Any]],
    optimal_factors: Optional[Dict[str, int]] = None,
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a plot of marginal variance improvement by number of factors.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    optimal_factors : Optional[Dict[str, int]]
        Dictionary mapping method names to optimal factor values, by default None
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size, by default (12, 8)
    save_path : Optional[str]
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Created figure
        
    Examples
    --------
    >>> # Visualize marginal improvement with optimal factors highlighted
    >>> fig = _visualize_marginal_improvement(metrics, optimal_factors, user_factors=15)
    >>> plt.show()
    """
    # Extract factors and variance values
    factors = sorted([k for k in metrics.keys() if metrics[k]["total_variance"] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid variance metrics available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
    
    if len(factors) <= 1:
        print("Need at least two factor values to calculate marginal improvement")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "Need at least two factor values to calculate marginal improvement", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
        
    variance = [metrics[k]["total_variance"] for k in factors]
    
    # Calculate marginal improvements
    diffs = np.diff(variance)
    diffs = np.append(diffs, diffs[-1])  # Pad to match length
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot bars for all factors
    bars = plt.bar(factors, diffs, color="green", alpha=0.6, width=min(2, (factors[-1] - factors[0]) / len(factors) / 2))
    
    # Calculate mean improvement
    mean_improvement = np.mean(diffs)
    half_mean = mean_improvement / 2
    
    # Add threshold line
    plt.axhline(
        y=half_mean, 
        color="red", 
        linestyle="--", 
        label=f"Half of mean improvement ({half_mean:.4f})"
    )
    
    # Highlight optimal factors if provided
    colors = {K_SELECTION_CRITERIA.ELBOW: "red", K_SELECTION_CRITERIA.THRESHOLD: "green", K_SELECTION_CRITERIA.BALANCED: "purple"}
    highlighted_factors = set()
    
    if optimal_factors:
        for method, n_factors in optimal_factors.items():
            if n_factors in factors and n_factors not in highlighted_factors:
                highlighted_factors.add(n_factors)
                color = colors.get(method, "gray")
                idx = factors.index(n_factors)
                plt.bar(
                    [n_factors], 
                    [diffs[idx]], 
                    color=color, 
                    alpha=1.0, 
                    width=min(2, (factors[-1] - factors[0]) / len(factors) / 2),
                    label=f"{method.capitalize()}: {n_factors} factors"
                )
    
    # Highlight user-specified factors if provided
    if user_factors and user_factors in factors and user_factors not in highlighted_factors:
        idx = factors.index(user_factors)
        plt.bar(
            [user_factors], 
            [diffs[idx]], 
            color="orange", 
            alpha=1.0, 
            width=min(2, (factors[-1] - factors[0]) / len(factors) / 2),
            label=f"User-specified: {user_factors} factors"
        )
    
    plt.xlabel("Number of factors")
    plt.ylabel("Marginal variance explained")
    plt.title("Marginal improvement per additional factor")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def _visualize_modality_variance(
    metrics: Dict[int, Dict[str, Any]],
    optimal_factors: Optional[Dict[str, int]] = None,
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a plot of variance explained per modality by number of factors.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    optimal_factors : Optional[Dict[str, int]]
        Dictionary mapping method names to optimal factor values, by default None
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size, by default (12, 8)
    save_path : Optional[str]
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    # Extract factors with valid metrics
    factors = sorted([k for k in metrics.keys() if metrics[k][VARIANCE_METRICS_DEFS.TOTAL_VARIANCE] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid variance metrics available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
    
    # Gather modality data
    mod_data = []
    for n_factors in factors:
        if metrics[n_factors][VARIANCE_METRICS_DEFS.MODALITY_VARIANCE]:
            mod_vars = metrics[n_factors][VARIANCE_METRICS_DEFS.MODALITY_VARIANCE]
            for mod, var in mod_vars.items():
                if var is not None:
                    mod_data.append({
                        "factors": n_factors, 
                        "modality": mod, 
                        "variance": var
                    })
    
    if not mod_data:
        print("No modality-specific variance data available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No modality-specific variance data available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
    
    # Create DataFrame for plotting
    mod_df = pd.DataFrame(mod_data)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot modality-specific variances
    sns.lineplot(data=mod_df, x="factors", y="variance", hue="modality", marker="o", markersize=8)
    
    # Highlight optimal factors if provided
    colors = {K_SELECTION_CRITERIA.ELBOW: "red", K_SELECTION_CRITERIA.THRESHOLD: "green", K_SELECTION_CRITERIA.BALANCED: "purple"}
    
    if optimal_factors:
        for method, n_factors in optimal_factors.items():
            if n_factors in factors:
                color = colors.get(method, "gray")
                plt.axvline(
                    x=n_factors, 
                    color=color, 
                    linestyle="--", 
                    alpha=0.7,
                    label=f"{method.capitalize()}: {n_factors} factors"
                )
    
    # Highlight user-specified factors if provided
    if user_factors and user_factors in factors:
        plt.axvline(
            x=user_factors, 
            color="orange", 
            linestyle="--", 
            alpha=0.7,
            label=f"User-specified: {user_factors} factors"
        )
    
    plt.xlabel("Number of factors")
    plt.ylabel("Variance explained (%)")
    plt.title("Modality-specific variance explained")
    
    # Set y-axis to percentage scale
    plt.ylim(0, min(100, mod_df["variance"].max() * 1.1))
    
    plt.legend(title="Modality", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig

def _factor_idx_to_name(idx: int) -> str:
    """Convert 0-based factor index to factor name (e.g. 0 -> 'Factor_1')."""
    return FACTOR_NAME_PATTERN.format(idx + 1)

def _factor_name_to_idx(name: str) -> int:
    """Convert factor name to 0-based index (e.g. 'Factor_1' -> 0)."""
    try:
        return int(name.split('_')[1]) - 1
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid factor name '{name}'. Expected format: 'Factor_N' where N is a positive integer.") from e
