"""
Statistical visualization functions.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest
from typing import Optional, Tuple, Dict
from .constants import STATISTICS_DEFS

def plot_term_pvalue_histogram(
    data: pd.DataFrame,
    term: str,
    figsize: Tuple[int, int] = (12, 4),
    include_stats: bool = True,
    show_ks_test: bool = True,
    fdr_cutoff: float = 0.05,
) -> plt.Figure:
    """
    Plot p-value histograms and QQ plot for a single term.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing p-values and optionally FDR values for a single term.
        Must have columns STATISTICS_DEFS.P_VALUE and optionally STATISTICS_DEFS.Q_VALUE.
    term : str
        Name of the term being plotted.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (12, 4).
    include_stats : bool, optional
        Whether to include summary statistics on the plots. Default is True.
    show_ks_test : bool, optional
        Whether to show Kolmogorov-Smirnov test results comparing p-value distribution
        to the uniform distribution. Default is True.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    n_panels = 3 if STATISTICS_DEFS.Q_VALUE in data.columns else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(f"P-value Distribution for Term: {term}", fontsize=16)

    # Raw p-values
    sns.histplot(data[STATISTICS_DEFS.P_VALUE], bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Raw P-values")
    axes[0].set_xlabel("P-value")
    axes[0].set_ylabel("Count")

    # Add reference uniform distribution line
    x = np.linspace(0, 1, 100)
    y = len(data) * 0.05  # Adjust height based on data
    axes[0].plot(x, [y] * 100, 'r--', label='Uniform Distribution')
    axes[0].legend()

    # Add summary statistics
    if include_stats:
        avg_p = data[STATISTICS_DEFS.P_VALUE].mean()
        median_p = data[STATISTICS_DEFS.P_VALUE].median()
        sig_count = sum(data[STATISTICS_DEFS.P_VALUE] < 0.05)
        sig_pct = 100 * sig_count / len(data)
        stats_text = f"Mean: {avg_p:.4f}\nMedian: {median_p:.4f}\n"
        stats_text += f"Significant: {sig_count}/{len(data)} ({sig_pct:.1f}%)"
        axes[0].text(0.05, 0.95, stats_text,
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # QQ plot of p-values
    pvals = np.sort(data[STATISTICS_DEFS.P_VALUE])
    expected = np.linspace(0, 1, len(pvals) + 1)[1:]
    axes[1].scatter(expected, pvals, alpha=0.5)
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_title("P-value QQ Plot")
    axes[1].set_xlabel("Expected P-value")
    axes[1].set_ylabel("Observed P-value")

    # FDR-corrected p-values (if available)
    if n_panels == 3:
        sns.histplot(data[STATISTICS_DEFS.Q_VALUE], bins=20, kde=True, ax=axes[2])
        axes[2].set_title("FDR-corrected P-values (Benjamini-Hochberg)")
        axes[2].set_xlabel("Adjusted P-value")
        axes[2].set_ylabel("Count")
        axes[2].plot(x, [y] * 100, 'r--', label='Uniform Distribution')
        axes[2].legend()
        if include_stats:
            avg_fdr = data[STATISTICS_DEFS.Q_VALUE].mean()
            median_fdr = data[STATISTICS_DEFS.Q_VALUE].median()
            sig_count_fdr = sum(data[STATISTICS_DEFS.Q_VALUE] < fdr_cutoff)
            sig_pct_fdr = 100 * sig_count_fdr / len(data)
            stats_text = f"Mean: {avg_fdr:.4f}\nMedian: {median_fdr:.4f}\n"
            stats_text += f"Significant: {sig_count_fdr}/{len(data)} ({sig_pct_fdr:.1f}%)"
            axes[2].text(0.05, 0.95, stats_text,
                        transform=axes[2].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # KS test
    if show_ks_test:
        ks_stat, ks_pval = kstest(data[STATISTICS_DEFS.P_VALUE], 'uniform')
        fig.text(0.5, 0.01,
                 f"Kolmogorov-Smirnov test against uniform distribution: "
                 f"statistic={ks_stat:.4f}, p-value={ks_pval:.4f}",
                 ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_pvalue_histograms(
    data: pd.DataFrame,
    term_column: str = STATISTICS_DEFS.TERM,
    figsize: Tuple[int, int] = (12, 8),
    include_stats: bool = True,
    show_ks_test: bool = True,
    fdr_cutoff: float = 0.05,
    terms: Optional[list] = None
) -> Dict[str, plt.Figure]:
    """
    Generate histograms of p-values and FDR-corrected p-values for all terms
    in a tidy data table.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy data table containing regression results. Must have columns:
        - term_column (str): column containing term names
        - STATISTICS_DEFS.P_VALUE (float): raw p-values
        - STATISTICS_DEFS.Q_VALUE (float, optional): FDR-corrected p-values
    term_column : str, optional
        Name of the column containing term names. Default is STATISTICS_DEFS.TERM.
    figsize : Tuple[int, int], optional
        Base figure size (width, height) in inches. Default is (12, 8).
    include_stats : bool, optional
        Whether to include summary statistics on the plots. Default is True.
    show_ks_test : bool, optional
        Whether to show Kolmogorov-Smirnov test results. Default is True.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
    terms : list, optional
        List of terms to plot. If None, plot all terms.

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping term names to their corresponding figure objects.
    """
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Get unique terms
    unique_terms = data[term_column].unique()
    if terms is not None:
        if isinstance(terms, str):
            terms = [terms]
        missing_terms = [t for t in terms if t not in unique_terms]
        if missing_terms:
            raise ValueError(f"Terms not found in data: {missing_terms}")
        unique_terms = [t for t in terms if t in unique_terms]

    # Generate plots for each term
    figures = {}
    for term in unique_terms:
        term_data = data[data[term_column] == term]
        figures[term] = plot_term_pvalue_histogram(
            term_data,
            term,
            figsize=figsize,
            include_stats=include_stats,
            show_ks_test=show_ks_test,
            fdr_cutoff=fdr_cutoff
        )

    return figures 