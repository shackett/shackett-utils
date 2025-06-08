import pytest
import numpy as np
import pandas as pd
import anndata as ad
from shackett_utils.genomics import adata_regression

@pytest.fixture
def minimal_adata():
    """Create a minimal AnnData object with integer data"""
    np.random.seed(42)
    n_cells = 100
    n_genes = 10
    
    # Create integer covariates
    batch = np.random.randint(0, 3, n_cells)  # 3 batches
    condition = np.random.randint(0, 2, n_cells)  # 2 conditions
    
    # Create expression matrix with some structure
    X = np.random.negative_binomial(10, 0.5, size=(n_cells, n_genes))
    
    # Create AnnData
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({
            'batch': batch,
            'condition': condition
        })
    )
    
    # Add gene names
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    
    return adata

def test_adata_regression_integer_covariates(minimal_adata):
    """Test that AnnData regression works with integer covariates"""
    # Test OLS
    results_ols = adata_regression.adata_model_fitting(
        minimal_adata,
        formula='~ batch + condition',
        model_class='ols',
        n_jobs=1,
        allow_failures=False
    )
    
    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0
    assert 'feature_name' in results_ols.columns
    assert 'term' in results_ols.columns
    assert 'estimate' in results_ols.columns
    
    # Test GAM
    results_gam = adata_regression.adata_model_fitting(
        minimal_adata,
        formula='~ batch + s(condition)',
        model_class='gam',
        n_jobs=1,
        allow_failures=False
    )
    
    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0
    assert 'feature_name' in results_gam.columns
    assert 'term' in results_gam.columns

def test_adata_regression_dtype_handling(minimal_adata):
    """Test that AnnData regression handles different data types correctly"""
    import numpy as np
    import pandas as pd
    
    # Add columns with different dtypes to obs
    minimal_adata.obs['numerical'] = pd.Series(np.random.randn(minimal_adata.n_obs), dtype='float64')
    minimal_adata.obs['integer'] = pd.Series(np.random.randint(0, 10, minimal_adata.n_obs), dtype='int64')
    minimal_adata.obs['category'] = pd.Categorical(['A', 'B'] * (minimal_adata.n_obs // 2))
    minimal_adata.obs['object'] = pd.Series(['X', 'Y'] * (minimal_adata.n_obs // 2), dtype='object')
    minimal_adata.obs['bool'] = pd.Series(np.random.choice([True, False], minimal_adata.n_obs))
    
    # Print dtypes to help debug
    print("\nData types in test DataFrame:")
    print(minimal_adata.obs.dtypes)
    
    # Test with different formula combinations
    formulas = [
        '~ numerical + integer',  # numeric only
        '~ numerical + category',  # with categorical
        '~ integer + bool',  # with boolean
        '~ numerical + object',  # with object dtype
        '~ s(numerical) + integer',  # GAM with numeric
        '~ s(integer) + numerical',  # GAM with integer
    ]
    
    for formula in formulas:
        print(f"\nTesting formula: {formula}")
        try:
            if 's(' in formula:
                model_class = 'gam'
            else:
                model_class = 'ols'
                
            results = adata_regression.adata_model_fitting(
                minimal_adata,
                formula=formula,
                model_class=model_class,
                n_jobs=1,
                allow_failures=False
            )
            print(f"Success with formula: {formula}")
            print(f"Results shape: {results.shape}")
            print(f"Results columns: {results.columns}")
        except Exception as e:
            print(f"Error with formula {formula}: {str(e)}")
            print(f"Data types of variables in formula:")
            for var in formula.split('~')[1].strip().split('+'):
                var = var.strip()
                if 's(' in var:
                    var = var[2:-1]  # Remove s( and )
                print(f"  {var}: {minimal_adata.obs[var].dtype}") 