import pytest
import numpy as np
import pandas as pd
from shackett_utils.statistics import multi_model_fitting as mmf
from shackett_utils.statistics.constants import STATISTICS_DEFS, TIDY_DEFS

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
    
    # Create DataFrame for formula interface
    data = pd.DataFrame(X_model, columns=['x1', 'x2'])
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    term_names = ['x1', 'x2']
    
    # Add intercept to X_model for matrix interface
    X_model_with_intercept = np.column_stack([np.ones(n_samples), X_model])
    term_names_with_intercept = ['intercept'] + term_names
    
    return {
        'data': data,
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

@pytest.fixture
def missing_data():
    """Create test data with missing values"""
    np.random.seed(42)
    n_samples = 20
    
    # Create data with missing values
    X_features = np.random.randn(n_samples, 3)
    X_model = np.random.randn(n_samples, 2)
    data = pd.DataFrame(X_model, columns=['x1', 'x2'])
    
    # Add missing values to different features
    X_features[0:5, 0] = np.nan  # 5 missing at start
    X_features[15:, 1] = np.nan  # 5 missing at end
    # Make middle feature fail by having perfect collinearity
    X_features[::2, 2] = np.nan  # Every other sample is missing
    X_features[1::2, 2] = 1.0  # All non-missing values are identical
    
    feature_names = ['missing_start', 'missing_end', 'missing_middle']
    term_names = ['x1', 'x2']
    
    # Add intercept to X_model for matrix interface
    X_model_with_intercept = np.column_stack([np.ones(n_samples), X_model])
    term_names_with_intercept = ['intercept'] + term_names
    
    return {
        'data': data,
        'X_features': X_features,
        'X_model': X_model,
        'X_model_with_intercept': X_model_with_intercept,
        'feature_names': feature_names,
        'term_names': term_names,
        'term_names_with_intercept': term_names_with_intercept,
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
    assert all(col in results.columns for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.STD_ERROR, TIDY_DEFS.P_VALUE])
    assert results.iloc[0][STATISTICS_DEFS.FEATURE_NAME] == test_data['feature_names'][0]
    assert results.iloc[0][STATISTICS_DEFS.TERM] == test_data['term_names_with_intercept'][0]

def test_fit_feature_model_formula(test_data):
    """Test formula-based fitting for a single feature"""
    # Test OLS
    results_ols = mmf.fit_feature_model_formula(
        test_data['X_features'][:, 0],
        test_data['data'],
        test_data['feature_names'][0],
        formula='y ~ x1 + x2',
        model_class='ols'
    )
    
    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0
    assert all(col in results_ols.columns for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.STD_ERROR, TIDY_DEFS.P_VALUE])
    
    # Test GAM
    results_gam = mmf.fit_feature_model_formula(
        test_data['X_features'][:, 0],
        test_data['data'],
        test_data['feature_names'][0],
        formula='y ~ x1 + s(x2)',
        model_class='gam'
    )
    
    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0
    assert all(col in results_gam.columns for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM])

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
    assert all(col in results_df.columns for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.P_VALUE, STATISTICS_DEFS.Q_VALUE])

def test_fit_parallel_models_formula(test_data):
    """Test parallel formula-based fitting"""
    # Test OLS
    results_df_ols = mmf.fit_parallel_models_formula(
        test_data['X_features'],
        test_data['data'],
        test_data['feature_names'],
        formula='y ~ x1 + x2',
        model_class='ols',
        n_jobs=2
    )
    
    assert isinstance(results_df_ols, pd.DataFrame)
    assert len(results_df_ols) > 0
    assert all(col in results_df_ols.columns for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.P_VALUE, STATISTICS_DEFS.Q_VALUE])
    
    # Test GAM
    results_df_gam = mmf.fit_parallel_models_formula(
        test_data['X_features'],
        test_data['data'],
        test_data['feature_names'],
        formula='y ~ x1 + s(x2)',
        model_class='gam',
        n_jobs=2
    )
    
    assert isinstance(results_df_gam, pd.DataFrame)
    assert len(results_df_gam) > 0
    assert all(col in results_df_gam.columns for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM])

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

def test_missing_values_formula(missing_data, caplog):
    """Test handling of missing values in formula interface"""
    caplog.set_level('DEBUG')
    
    # Test single feature with missing values
    results = mmf.fit_feature_model_formula(
        missing_data['X_features'][:, 0],
        missing_data['data'],
        missing_data['feature_names'][0],
        formula='y ~ x1 + x2',
        model_class='ols'
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "Filtering 5 missing values" in caplog.text
    
    # Test parallel fitting with different missing value patterns
    results_all = mmf.fit_parallel_models_formula(
        missing_data['X_features'],
        missing_data['data'],
        missing_data['feature_names'],
        formula='y ~ x1 + x2',
        model_class='ols'
    )
    
    assert isinstance(results_all, pd.DataFrame)
    assert len(results_all) > 0
    # Should have results for features with sufficient data
    assert len(set(results_all[STATISTICS_DEFS.FEATURE_NAME].unique()) & set(['missing_start', 'missing_end'])) == 2
    # Feature with too many missing values should be skipped
    assert 'missing_middle' not in results_all[STATISTICS_DEFS.FEATURE_NAME].unique()

def test_missing_values_matrix(missing_data, caplog):
    """Test handling of missing values in matrix interface"""
    caplog.set_level('DEBUG')
    
    # Test single feature with missing values
    results = mmf.fit_feature_model_matrix(
        missing_data['X_features'][:, 0],
        missing_data['X_model_with_intercept'],
        missing_data['feature_names'][0],
        missing_data['term_names_with_intercept']
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "Filtering 5 missing values" in caplog.text
    
    # Test parallel fitting with different missing value patterns
    results_all = mmf.fit_parallel_models_matrix(
        missing_data['X_features'],
        missing_data['X_model_with_intercept'],
        missing_data['feature_names'],
        missing_data['term_names_with_intercept']
    )
    
    assert isinstance(results_all, pd.DataFrame)
    assert len(results_all) > 0
    # Should have results for features with sufficient data
    assert len(set(results_all[STATISTICS_DEFS.FEATURE_NAME].unique()) & set(['missing_start', 'missing_end'])) == 2
    # Feature with too many missing values should be skipped
    assert 'missing_middle' not in results_all[STATISTICS_DEFS.FEATURE_NAME].unique()

def test_insufficient_samples():
    """Test handling of features with insufficient non-missing samples"""
    # Create data with only 1 valid sample
    X_features = np.array([[1.0], [np.nan], [np.nan]])
    X_model = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = pd.DataFrame(X_model, columns=['x1', 'x2'])
    
    # Test formula interface
    results_formula = mmf.fit_feature_model_formula(
        X_features[:, 0],
        data,
        'insufficient_samples',
        formula='y ~ x1 + x2',
        model_class='ols'
    )
    assert len(results_formula) == 0
    
    # Test matrix interface
    results_matrix = mmf.fit_feature_model_matrix(
        X_features[:, 0],
        X_model,
        'insufficient_samples',
        ['x1', 'x2']
    )
    assert len(results_matrix) == 0

def test_input_validation():
    """Test input validation"""
    X_features = np.random.randn(10, 2)
    X_model = np.random.randn(5, 2)  # Mismatched samples
    data = pd.DataFrame(X_model, columns=['x1', 'x2'])
    feature_names = ['f1', 'f2']
    term_names = ['x1', 'x2']
    
    # Matrix interface
    with pytest.raises(ValueError, match="must have same number of samples"):
        mmf.fit_parallel_models_matrix(X_features, X_model, feature_names, term_names)
    
    # Formula interface
    with pytest.raises(ValueError, match="must have same number of samples"):
        mmf.fit_parallel_models_formula(X_features, data, feature_names, formula='y ~ x1 + x2')
    
    # Test mismatched feature names
    X_features_small = np.random.randn(5, 1)
    with pytest.raises(ValueError, match="Length of feature_names .* must match number of features"):
        mmf.fit_parallel_models_formula(X_features_small, data, feature_names, formula='y ~ x1 + x2')
    
    # Test unsupported model class
    with pytest.raises(ValueError, match="Unsupported model class"):
        mmf.fit_feature_model_formula(
            X_features_small[:, 0],
            data,
            'feature',
            formula='y ~ x1 + x2',
            model_class='unsupported'
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