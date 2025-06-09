"""Tests for statistical visualization functions."""

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from shackett_utils.statistics import stats_viz
from shackett_utils.statistics.constants import STATISTICS_DEFS

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create a mix of uniform and biased p-values
    uniform_p = np.random.uniform(0, 1, n_samples // 2)
    biased_p = np.random.beta(0.5, 5, n_samples // 2)  # More small p-values
    p_values = np.concatenate([uniform_p, biased_p])
    
    # Create FDR-corrected p-values (simulated)
    q_values = np.minimum(p_values * 2, 1)
    
    # Create DataFrame with multiple terms and modalities
    data = []
    terms = ['term1', 'term2']
    modalities = ['RNA', 'ATAC']
    
    for term in terms:
        for modality in modalities:
            # Shuffle p-values and q-values for each combination
            np.random.shuffle(p_values)
            np.random.shuffle(q_values)
            
            data.extend([{
                STATISTICS_DEFS.TERM: term,
                'data_modality': modality,
                STATISTICS_DEFS.P_VALUE: p,
                STATISTICS_DEFS.Q_VALUE: q,
                'statistic': np.random.normal(),
                'std_error': np.random.uniform(0.1, 0.5)
            } for p, q in zip(p_values, q_values)])
    
    return pd.DataFrame(data)

def test_plot_term_pvalue_histogram_basic(sample_data):
    """Test basic functionality of plot_term_pvalue_histogram."""
    # Filter for a single term
    term_data = sample_data[sample_data[STATISTICS_DEFS.TERM] == 'term1']
    term_data = term_data[term_data['data_modality'] == 'RNA']
    
    # Test with default parameters
    fig = stats_viz.plot_term_pvalue_histogram(term_data, 'term1')
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3  # Should have 3 panels (histogram, QQ plot, FDR)
    plt.close(fig)
    
    # Test without FDR values
    term_data_no_fdr = term_data.drop(columns=[STATISTICS_DEFS.Q_VALUE])
    fig = stats_viz.plot_term_pvalue_histogram(term_data_no_fdr, 'term1')
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Should have 2 panels (histogram, QQ plot)
    plt.close(fig)

def test_plot_term_pvalue_histogram_options(sample_data):
    """Test various options of plot_term_pvalue_histogram."""
    term_data = sample_data[sample_data[STATISTICS_DEFS.TERM] == 'term1']
    term_data = term_data[term_data['data_modality'] == 'RNA']
    
    # Test without stats
    fig = stats_viz.plot_term_pvalue_histogram(
        term_data, 'term1', include_stats=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test without KS test
    fig = stats_viz.plot_term_pvalue_histogram(
        term_data, 'term1', show_ks_test=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with custom figsize
    fig = stats_viz.plot_term_pvalue_histogram(
        term_data, 'term1', figsize=(8, 3))
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches().tolist() == [8, 3]
    plt.close(fig)
    
    # Test with title prefix
    fig = stats_viz.plot_term_pvalue_histogram(
        term_data, 'term1', title_prefix='RNA - ')
    assert isinstance(fig, plt.Figure)
    assert 'RNA - ' in fig._suptitle.get_text()
    plt.close(fig)

def test_plot_pvalue_histograms_basic(sample_data):
    """Test basic functionality of plot_pvalue_histograms."""
    # Test without partitioning
    figures = stats_viz.plot_pvalue_histograms(sample_data)
    assert isinstance(figures, dict)
    assert set(figures.keys()) == {'term1', 'term2'}
    assert all(isinstance(fig, plt.Figure) for fig in figures.values())
    for fig in figures.values():
        plt.close(fig)
    
    # Test with specific terms
    figures = stats_viz.plot_pvalue_histograms(
        sample_data, terms=['term1'])
    assert isinstance(figures, dict)
    assert set(figures.keys()) == {'term1'}
    for fig in figures.values():
        plt.close(fig)

def test_plot_pvalue_histograms_partitioned(sample_data):
    """Test partitioned functionality of plot_pvalue_histograms."""
    # Test with partitioning
    figures = stats_viz.plot_pvalue_histograms(
        sample_data,
        partition_column='data_modality'
    )
    assert isinstance(figures, dict)
    assert set(figures.keys()) == {'term1', 'term2'}
    assert all(isinstance(term_dict, dict) for term_dict in figures.values())
    assert all(set(term_dict.keys()) == {'RNA', 'ATAC'} 
              for term_dict in figures.values())
    
    # Close all figures
    for term_dict in figures.values():
        for fig in term_dict.values():
            plt.close(fig)
    
    # Test with specific partition values
    figures = stats_viz.plot_pvalue_histograms(
        sample_data,
        partition_column='data_modality',
        partition_values=['RNA']
    )
    assert isinstance(figures, dict)
    assert all(set(term_dict.keys()) == {'RNA'} 
              for term_dict in figures.values())
    
    # Close all figures
    for term_dict in figures.values():
        for fig in term_dict.values():
            plt.close(fig)

def test_plot_pvalue_histograms_validation(sample_data):
    """Test input validation in plot_pvalue_histograms."""
    # Test invalid term
    with pytest.raises(ValueError, match="Terms not found in data"):
        stats_viz.plot_pvalue_histograms(
            sample_data, terms=['nonexistent_term'])
    
    # Test invalid partition value
    with pytest.raises(ValueError, match="Partition values not found in data"):
        stats_viz.plot_pvalue_histograms(
            sample_data,
            partition_column='data_modality',
            partition_values=['nonexistent_modality']
        )
    
    # Test invalid partition column
    with pytest.raises(KeyError):
        stats_viz.plot_pvalue_histograms(
            sample_data,
            partition_column='nonexistent_column'
        )

def test_plot_pvalue_histograms_empty_combinations(sample_data):
    """Test handling of empty term-partition combinations."""
    # Create a case where some combinations have no data
    filtered_data = sample_data[
        ~((sample_data[STATISTICS_DEFS.TERM] == 'term1') & 
          (sample_data['data_modality'] == 'RNA'))
    ]
    
    figures = stats_viz.plot_pvalue_histograms(
        filtered_data,
        partition_column='data_modality'
    )
    
    # term1 should not have 'RNA' partition
    assert 'RNA' not in figures['term1']
    assert 'ATAC' in figures['term1']
    
    # term2 should have both partitions
    assert set(figures['term2'].keys()) == {'RNA', 'ATAC'}
    
    # Close all figures
    for term_dict in figures.values():
        for fig in term_dict.values():
            plt.close(fig) 