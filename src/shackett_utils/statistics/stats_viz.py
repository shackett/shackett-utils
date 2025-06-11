"""
Statistical visualization functions.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest
from typing import Optional, Tuple, Dict, List, Union
from .constants import STATISTICS_DEFS


def plot_pvalue_histograms(
    data: pd.DataFrame,
    term_column: str = STATISTICS_DEFS.TERM,
    partition_column: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    include_stats: bool = True,
    show_ks_test: bool = True,
    fdr_cutoff: float = 0.05,
    terms: Optional[Union[str, List[str]]] = None,
    partition_values: Optional[Union[str, List[str]]] = None,
    n_cols: int = 3,
) -> Dict[str, Union[plt.Figure, Dict[str, plt.Figure]]]:
    """
    Generate histograms of p-values and FDR-corrected p-values for all terms
    in a tidy data table, optionally stratified by a partition column.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy data table containing regression results. Must have columns:
        - term_column (str): column containing term names
        - STATISTICS_DEFS.P_VALUE (float): raw p-values
        - STATISTICS_DEFS.Q_VALUE (float, optional): FDR-corrected p-values
    term_column : str, optional
        Name of the column containing term names. Default is STATISTICS_DEFS.TERM.
    partition_column : str, optional
        Name of the column to partition by (e.g., 'data_modality'). If None,
        plots are not stratified. Default is None.
    figsize : Tuple[int, int], optional
        Base figure size (width, height) in inches. Default is (12, 8).
    include_stats : bool, optional
        Whether to include summary statistics on the plots. Default is True.
    show_ks_test : bool, optional
        Whether to show Kolmogorov-Smirnov test results. Default is True.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
    terms : str or list, optional
        Term(s) to plot. If None, plot all terms.
    partition_values : str or list, optional
        Partition value(s) to include. If None, use all values.
    n_cols : int, optional
        Number of columns in the grid when plotting multiple terms/partitions.
        Default is 3.

    Returns
    -------
    Dict[str, Union[plt.Figure, Dict[str, plt.Figure]]]
        If partition_column is None:
            Dictionary mapping term names to their corresponding figure objects.
        If partition_column is specified:
            Dictionary mapping term names to dictionaries of partition values
            and their corresponding figure objects.
    """
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Handle terms input
    if isinstance(terms, str):
        terms = [terms]
    unique_terms = data[term_column].unique() if terms is None else terms
    if terms is not None:
        missing_terms = [t for t in terms if t not in data[term_column].unique()]
        if missing_terms:
            raise ValueError(f"Terms not found in data: {missing_terms}")
        unique_terms = [t for t in terms if t in data[term_column].unique()]

    # If no partition column, use original behavior
    if partition_column is None:
        figures = {}
        for term in unique_terms:
            term_data = data[data[term_column] == term]
            figures[term] = plot_term_pvalue_histogram(
                term_data,
                term,
                figsize=figsize,
                include_stats=include_stats,
                show_ks_test=show_ks_test,
                fdr_cutoff=fdr_cutoff,
            )
        return figures

    # Handle partition values input
    if isinstance(partition_values, str):
        partition_values = [partition_values]
    unique_partitions = (
        data[partition_column].unique()
        if partition_values is None
        else partition_values
    )
    if partition_values is not None:
        missing_partitions = [
            p for p in partition_values if p not in data[partition_column].unique()
        ]
        if missing_partitions:
            raise ValueError(
                f"Partition values not found in data: {missing_partitions}"
            )
        unique_partitions = [
            p for p in partition_values if p in data[partition_column].unique()
        ]

    # Create figures for each term-partition combination
    figures = {}
    for term in unique_terms:
        figures[term] = {}
        for partition in unique_partitions:
            # Filter data for this term-partition combination
            mask = (data[term_column] == term) & (data[partition_column] == partition)
            term_partition_data = data[mask]

            if len(term_partition_data) > 0:  # Only create plot if data exists
                figures[term][partition] = plot_term_pvalue_histogram(
                    term_partition_data,
                    term,
                    figsize=figsize,
                    include_stats=include_stats,
                    show_ks_test=show_ks_test,
                    fdr_cutoff=fdr_cutoff,
                    title_prefix=f"{partition} - ",
                )

    return figures


def plot_term_pvalue_histogram(
    data: pd.DataFrame,
    term: str,
    figsize: Tuple[int, int] = (12, 4),
    include_stats: bool = True,
    show_ks_test: bool = True,
    fdr_cutoff: float = 0.05,
    title_prefix: str = "",
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
    title_prefix : str, optional
        Prefix to add to the plot title. Default is "".

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    n_panels = 3 if STATISTICS_DEFS.Q_VALUE in data.columns else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(f"{title_prefix}P-value Distribution for Term: {term}", fontsize=16)

    # Plot raw p-values histogram
    _plot_raw_pvalue_histogram(data, axes[0], include_stats)

    # Plot QQ plot
    _plot_pvalue_qq(data, axes[1])

    # Plot FDR-corrected p-values if available
    if n_panels == 3:
        _plot_qvalue_histogram(data, axes[2], fdr_cutoff, include_stats)

    # KS test
    if show_ks_test:
        ks_stat, ks_pval = kstest(data[STATISTICS_DEFS.P_VALUE], "uniform")
        fig.text(
            0.5,
            0.01,
            f"Kolmogorov-Smirnov test against uniform distribution: "
            f"statistic={ks_stat:.4f}, p-value={ks_pval:.4f}",
            ha="center",
            fontsize=12,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def _plot_raw_pvalue_histogram(
    data: pd.DataFrame, ax: plt.Axes, include_stats: bool = True
) -> None:
    """Plot histogram of raw p-values with uniform distribution reference line.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing p-values
    ax : plt.Axes
        Matplotlib axes to plot on
    include_stats : bool
        Whether to include summary statistics on the plot
    """
    sns.histplot(data[STATISTICS_DEFS.P_VALUE], bins=20, kde=True, ax=ax)
    ax.set_title("Raw P-values")
    ax.set_xlabel("P-value")
    ax.set_ylabel("Count")

    # Add reference uniform distribution line
    x = np.linspace(0, 1, 100)
    y = len(data) * 0.05  # Adjust height based on data
    ax.plot(x, [y] * 100, "r--", label="Uniform Distribution")
    ax.legend()

    if include_stats:
        avg_p = data[STATISTICS_DEFS.P_VALUE].mean()
        median_p = data[STATISTICS_DEFS.P_VALUE].median()
        sig_count = sum(data[STATISTICS_DEFS.P_VALUE] < 0.05)
        sig_pct = 100 * sig_count / len(data)
        stats_text = f"Mean: {avg_p:.4f}\nMedian: {median_p:.4f}\n"
        stats_text += f"Significant: {sig_count}/{len(data)} ({sig_pct:.1f}%)"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )


def _plot_pvalue_qq(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot QQ plot of p-values against uniform distribution.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing p-values
    ax : plt.Axes
        Matplotlib axes to plot on
    """
    pvals = np.sort(data[STATISTICS_DEFS.P_VALUE])
    expected = np.linspace(0, 1, len(pvals) + 1)[1:]
    ax.scatter(expected, pvals, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_title("P-value QQ Plot")
    ax.set_xlabel("Expected P-value")
    ax.set_ylabel("Observed P-value")


def _plot_qvalue_histogram(
    data: pd.DataFrame,
    ax: plt.Axes,
    fdr_cutoff: float = 0.05,
    include_stats: bool = True,
) -> None:
    """Plot histogram of FDR-corrected p-values with cutoff line.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing q-values
    ax : plt.Axes
        Matplotlib axes to plot on
    fdr_cutoff : float
        FDR significance threshold to mark with vertical line
    include_stats : bool
        Whether to include summary statistics on the plot
    """
    sns.histplot(data[STATISTICS_DEFS.Q_VALUE], bins=20, kde=True, ax=ax)
    ax.set_title("FDR-corrected P-values (Benjamini-Hochberg)")
    ax.set_xlabel("Adjusted P-value")
    ax.set_ylabel("Count")

    # Set x-axis limits from 0 to 1
    ax.set_xlim(0, 1)

    # Add vertical line at FDR cutoff
    ax.axvline(
        x=fdr_cutoff, color="red", linestyle="--", label=f"FDR cutoff: {fdr_cutoff}"
    )
    ax.legend()

    if include_stats:
        avg_fdr = data[STATISTICS_DEFS.Q_VALUE].mean()
        median_fdr = data[STATISTICS_DEFS.Q_VALUE].median()
        sig_count_fdr = sum(data[STATISTICS_DEFS.Q_VALUE] < fdr_cutoff)
        sig_pct_fdr = 100 * sig_count_fdr / len(data)
        stats_text = f"Mean: {avg_fdr:.4f}\nMedian: {median_fdr:.4f}\n"
        stats_text += f"Significant: {sig_count_fdr}/{len(data)} ({sig_pct_fdr:.1f}%)"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
