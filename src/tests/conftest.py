"""
Global pytest configuration and fixtures.
"""
import pytest
import warnings
import numpy as np
import pandas as pd
import mudata as md
import anndata as ad

# Configure pytest to ignore specific numpy deprecation warnings
@pytest.fixture(autouse=True)
def ignore_numpy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="pandas.core.dtypes.cast",
            message=".*find_common_type is deprecated.*"
        )
        yield 

@pytest.fixture
def simple_mdata():
    """Create a simple MuData object with two modalities for testing."""
    # Create first modality with clear structure
    n_obs = 100
    n_vars1 = 50
    n_vars2 = 30
    
    # Generate structured data for first modality
    np.random.seed(42)
    # Create two clear patterns in the data
    pattern1 = np.sin(np.linspace(0, 4*np.pi, n_obs))[:, np.newaxis]
    pattern2 = np.cos(np.linspace(0, 4*np.pi, n_obs))[:, np.newaxis]
    
    # Create RNA data with two main patterns
    X1 = np.zeros((n_obs, n_vars1))
    X1[:, :n_vars1//2] = pattern1  # First half of genes follow pattern1
    X1[:, n_vars1//2:] = pattern2  # Second half follow pattern2
    X1 += np.random.normal(0, 0.1, (n_obs, n_vars1))  # Add noise
    
    # Create ATAC data with similar structure
    pattern3 = np.sin(np.linspace(0, 2*np.pi, n_obs))[:, np.newaxis]
    pattern4 = np.cos(np.linspace(0, 2*np.pi, n_obs))[:, np.newaxis]
    
    X2 = np.zeros((n_obs, n_vars2))
    X2[:, :n_vars2//2] = pattern3
    X2[:, n_vars2//2:] = pattern4
    X2 += np.random.normal(0, 0.1, (n_obs, n_vars2))
    
    # Create sample metadata
    obs = pd.DataFrame(
        index=[f'sample_{i}' for i in range(n_obs)],
        data={
            'group': np.repeat(['A', 'B'], n_obs//2),
            'continuous_var': np.random.normal(0, 1, n_obs)
        }
    )
    
    # Create modalities
    rna = ad.AnnData(
        X=X1,
        obs=obs,
        var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_vars1)])
    )
    
    atac = ad.AnnData(
        X=X2,
        obs=obs,
        var=pd.DataFrame(index=[f'peak_{i}' for i in range(n_vars2)])
    )
    
    # Create log2 centered layer
    rna.layers['log2_centered'] = X1 - np.mean(X1, axis=0)
    atac.layers['log2_centered'] = X2 - np.mean(X2, axis=0)
    
    # Create MuData object
    mdata = md.MuData({'rna': rna, 'atac': atac})
    
    return mdata 