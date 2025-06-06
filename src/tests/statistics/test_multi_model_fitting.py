import pytest
import numpy as np
import pandas as pd
from shackett_utils.statistics import multi_model_fitting as mmf

@pytest.fixture
def test_data():
    """Create test data for model fitting"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create features with known relationships
    X_features = np.random.randn(n_samples, n_features)
    X_model = np.random.randn(n_samples, 2)  # Two predictors
    
    # Create response with known relationships
    y = 1.0 + 2.0 * X_model[:, 0] + 0.5 * X_model[:, 1] + np.random.normal(0, 0.1, n_samples)
    X_features[:, 0] = y  # First feature has strong relationship
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    term_names = ['x1', 'x2']
    
    # Add intercept to X_model for matrix interface
    X_model_with_intercept = np.column_stack([np.ones(n_samples), X_model])
    term_names_with_intercept = ['intercept'] + term_names
    
    return {
        'X_features': X_features,
        'X_model': X_model,
        'X_model_with_intercept': X_model_with_intercept,
        'feature_names': feature_names,
        'term_names': term_names,
        'term_names_with_intercept': term_names_with_intercept,
        'n_samples': n_samples
    }

@pytest.fixture
def zero_var_data():
    """Create test data with zero variance features"""
    n_samples = 10
    X_features = np.zeros((n_samples, 2))  # All zero variance
    X_model = np.random.randn(n_samples, 2)
    X_model_with_intercept = np.column_stack([np.ones(n_samples), X_model])
    
    return {
        'X_features': X_features,
        'X_model': X_model,
        'X_model_with_intercept': X_model_with_intercept,
        'feature_names': ['f1', 'f2'],
        'term_names': ['x1', 'x2'],
        'term_names_with_intercept': ['intercept', 'x1', 'x2'],
        'n_samples': n_samples
    }

def test_fit_feature_model_matrix(test_data):
    """Test matrix-based OLS fitting for a single feature"""
    results = mmf.fit_feature_model_matrix(
        test_data['X_features'][:, 0],
        test_data['X_model_with_intercept'],
        test_data['feature_names'][0],
        test_data['term_names_with_intercept']
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(test_data['term_names_with_intercept'])
    assert all(col in results.columns for col in ['feature', 'term', 'estimate', 'std_error', 'p_value'])
    assert results.iloc[0]['feature'] == test_data['feature_names'][0]
    assert results.iloc[0]['term'] == test_data['term_names_with_intercept'][0]

def test_fit_feature_model_formula(test_data):
    """Test formula-based fitting for a single feature"""
    # Test OLS
    results_ols = mmf.fit_feature_model_formula(
        test_data['X_features'][:, 0],
        test_data['X_model'],
        test_data['feature_names'][0],
        formula='y ~ x1 + x2',
        term_names=test_data['term_names'],
        model_class='ols'
    )
    
    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0
    assert all(col in results_ols.columns for col in ['feature', 'term', 'estimate', 'std_error', 'p_value'])
    
    # Test GAM
    results_gam = mmf.fit_feature_model_formula(
        test_data['X_features'][:, 0],
        test_data['X_model'],
        test_data['feature_names'][0],
        formula='y ~ x1 + s(x2)',
        term_names=test_data['term_names'],
        model_class='gam'
    )
    
    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0
    assert all(col in results_gam.columns for col in ['feature', 'term'])

def test_fit_parallel_models_matrix(test_data):
    """Test parallel matrix-based OLS fitting"""
    results_df = mmf.fit_parallel_models_matrix(
        test_data['X_features'],
        test_data['X_model_with_intercept'],
        test_data['feature_names'],
        test_data['term_names_with_intercept'],
        n_jobs=2
    )
    
    assert isinstance(results_df, pd.DataFrame)
    # Each feature should have results for each term
    expected_rows = len(test_data['feature_names']) * len(test_data['term_names_with_intercept'])
    assert len(results_df) == expected_rows
    assert all(col in results_df.columns for col in ['feature', 'term', 'estimate', 'p_value', 'fdr_bh'])

def test_fit_parallel_models_formula(test_data):
    """Test parallel formula-based fitting"""
    # Test OLS
    results_df_ols = mmf.fit_parallel_models_formula(
        test_data['X_features'],
        test_data['X_model'],
        test_data['feature_names'],
        formula='y ~ x1 + x2',
        term_names=test_data['term_names'],
        model_class='ols',
        n_jobs=2
    )
    
    assert isinstance(results_df_ols, pd.DataFrame)
    assert len(results_df_ols) > 0
    assert all(col in results_df_ols.columns for col in ['feature', 'term', 'estimate', 'p_value', 'fdr_bh'])
    
    # Test GAM
    results_df_gam = mmf.fit_parallel_models_formula(
        test_data['X_features'],
        test_data['X_model'],
        test_data['feature_names'],
        formula='y ~ x1 + s(x2)',
        term_names=test_data['term_names'],
        model_class='gam',
        n_jobs=2
    )
    
    assert isinstance(results_df_gam, pd.DataFrame)
    assert len(results_df_gam) > 0
    assert all(col in results_df_gam.columns for col in ['feature', 'term'])

def test_zero_variance_features(zero_var_data):
    """Test handling of zero variance features"""
    # Matrix interface
    results_matrix = mmf.fit_parallel_models_matrix(
        zero_var_data['X_features'],
        zero_var_data['X_model_with_intercept'],
        zero_var_data['feature_names'],
        zero_var_data['term_names_with_intercept'],
        n_jobs=1
    )
    assert isinstance(results_matrix, pd.DataFrame)
    assert len(results_matrix) == 0
    
    # Formula interface
    results_formula = mmf.fit_parallel_models_formula(
        zero_var_data['X_features'],
        zero_var_data['X_model'],
        zero_var_data['feature_names'],
        formula='y ~ x1 + x2',
        term_names=zero_var_data['term_names'],
        model_class='ols',
        n_jobs=1
    )
    assert isinstance(results_formula, pd.DataFrame)
    assert len(results_formula) == 0

def test_input_validation():
    """Test input validation"""
    X_features = np.random.randn(10, 2)
    X_model = np.random.randn(5, 2)  # Mismatched samples
    feature_names = ['f1', 'f2']
    term_names = ['x1', 'x2']
    
    # Matrix interface
    with pytest.raises(ValueError, match="must have same number of samples"):
        mmf.fit_parallel_models_matrix(X_features, X_model, feature_names, term_names)
    
    # Formula interface
    with pytest.raises(ValueError, match="must have same number of samples"):
        mmf.fit_parallel_models_formula(
            X_features, X_model, feature_names,
            formula='y ~ x1 + x2', term_names=term_names
        )
    
    # Test invalid model class
    with pytest.raises(ValueError, match="Unsupported model class"):
        mmf.fit_feature_model_formula(
            X_features[:, 0], X_model, 'f1',
            formula='y ~ x1 + x2', term_names=term_names,
            model_class='invalid'
        )

def test_progress_bar(test_data, caplog):
    """Test progress bar and logging"""
    caplog.set_level('INFO')
    
    # Test with progress bar
    _ = mmf.fit_parallel_models_matrix(
        test_data['X_features'],
        test_data['X_model_with_intercept'],
        test_data['feature_names'],
        test_data['term_names_with_intercept'],
        progress_bar=True
    )
    assert any("Starting parallel model fitting" in record.message for record in caplog.records)
    assert any("Completed model fitting" in record.message for record in caplog.records)
    
    caplog.clear()
    
    # Test without progress bar
    _ = mmf.fit_parallel_models_matrix(
        test_data['X_features'],
        test_data['X_model_with_intercept'],
        test_data['feature_names'],
        test_data['term_names_with_intercept'],
        progress_bar=False
    )
    assert any("Starting parallel model fitting" in record.message for record in caplog.records)
    assert any("Completed model fitting" in record.message for record in caplog.records) 