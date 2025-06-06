import pytest
import pandas as pd
import numpy as np

from shackett_utils.statistics import model_fitting

@pytest.fixture
def test_data():
    """Fixture to provide test data for pytest"""
    np.random.seed(123)
    n = 50
    data = pd.DataFrame({
        'y': np.random.normal(5, 1, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.uniform(0, 5, n),
    })
    data['y'] = 1 + 2 * data['x1'] + 0.5 * data['x2'] + np.random.normal(0, 0.5, n)
    return data

@pytest.fixture
def test_matrices():
    """Fixture to provide X, y matrices for testing"""
    np.random.seed(123)
    n = 50
    X = np.random.randn(n, 2)
    y = 1 + 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.5, n)
    return X, y

def test_ols_creation(test_data):
    """Test OLS model creation and fitting"""
    model = model_fitting.fit_model('y ~ x1 + x2', data=test_data, method='ols')
    assert model.fitted_model is not None
    assert model.formula == 'y ~ x1 + x2'

def test_ols_tidy(test_data):
    """Test OLS tidy output format"""
    model = model_fitting.fit_model('y ~ x1 + x2', data=test_data, method='ols')
    tidy_df = model.tidy()
    assert isinstance(tidy_df, pd.DataFrame)
    assert len(tidy_df) == 3  # Intercept + 2 predictors
    required_cols = ['term', 'estimate', 'std_error', 't_statistic', 'p_value']
    assert all(col in tidy_df.columns for col in required_cols)

def test_ols_glance(test_data):
    """Test OLS glance output format"""
    model = model_fitting.fit_model('y ~ x1 + x2', data=test_data, method='ols')
    glance_df = model.glance()
    assert isinstance(glance_df, pd.DataFrame)
    assert len(glance_df) == 1  # Single row
    required_cols = ['r_squared', 'adj_r_squared', 'aic', 'bic', 'nobs']
    assert all(col in glance_df.columns for col in required_cols)
    assert glance_df['nobs'].iloc[0] == len(test_data)

def test_ols_augment(test_data):
    """Test OLS augment output format"""
    model = model_fitting.fit_model('y ~ x1 + x2', data=test_data, method='ols')
    aug_df = model.augment()
    assert isinstance(aug_df, pd.DataFrame)
    assert len(aug_df) == len(test_data)
    assert '.fitted' in aug_df.columns
    assert '.resid' in aug_df.columns
    # Check that original columns are preserved
    assert all(col in aug_df.columns for col in test_data.columns)

def test_ols_fit_xy(test_matrices):
    """Test OLS fitting with X, y matrices"""
    X, y = test_matrices
    model = model_fitting.fit_model_xy(X, y, method='ols', feature_names=['var1', 'var2'])
    assert model.fitted_model is not None
    assert model.feature_names == ['const', 'var1', 'var2']
    
    # Test tidy output
    tidy_df = model.tidy()
    assert len(tidy_df) == 3  # Intercept + 2 predictors
    assert 'const' in tidy_df['term'].values

def test_gam_creation(test_data):
    """Test GAM model creation"""
    model = model_fitting.fit_model('y ~ x1 + x2', data=test_data, method='gam')
    assert model.fitted_model is not None
    assert model.formula == 'y ~ x1 + x2'
    assert model.feature_names == ['x1', 'x2']

def test_gam_formula_smooth_syntax(test_data):
    """Test GAM with smooth terms specified in formula"""
    # Test s() syntax in formula
    model = model_fitting.fit_model('y ~ x1 + s(x2)', data=test_data, method='gam')
    assert model.fitted_model is not None
    assert 'x2' in model.smooth_terms
    assert 'x1' not in model.smooth_terms  # x1 should be linear
    
    # Test multiple smooth terms
    model2 = model_fitting.fit_model('y ~ s(x1) + s(x2)', data=test_data, method='gam')
    assert set(model2.smooth_terms) == {'x1', 'x2'}

def test_gam_tidy(test_data):
    """Test GAM tidy output format"""
    model = model_fitting.fit_model('y ~ x1 + s(x2)', data=test_data, method='gam')
    tidy_df = model.tidy()
    assert isinstance(tidy_df, pd.DataFrame)
    assert len(tidy_df) == 2  # 2 predictors
    assert 'term' in tidy_df.columns
    assert 'type' in tidy_df.columns
    
    # Check that x1 is linear and x2 is smooth
    term_types = dict(zip(tidy_df['term'], tidy_df['type']))
    assert term_types['x1'] == 'linear'
    assert term_types['s(x2)'] == 'smooth'

def test_gam_glance(test_data):
    """Test GAM glance output format"""
    model = model_fitting.fit_model('y ~ x1 + s(x2)', data=test_data, method='gam')
    glance_df = model.glance()
    assert isinstance(glance_df, pd.DataFrame)
    assert len(glance_df) == 1
    assert 'r_squared' in glance_df.columns
    assert 'nobs' in glance_df.columns
    assert glance_df['nobs'].iloc[0] == len(test_data)

def test_gam_fit_xy(test_matrices):
    """Test GAM fitting with X, y matrices"""
    X, y = test_matrices
    with pytest.raises(NotImplementedError, match="Matrix-based fitting is not supported for GAM models"):
        model_fitting.fit_model_xy(X, y, method='gam', feature_names=['var1', 'var2'],
                             smooth_terms=['var1'])


def test_model_registry_valid_methods(test_data):
    """Test that all valid methods work"""
    for method in ['ols', 'lm', 'linear', 'gam', 'smooth']:
        model = model_fitting.fit_model('y ~ x1', data=test_data, method=method)
        assert model.fitted_model is not None

def test_model_registry_invalid_method(test_data):
    """Test that invalid methods raise appropriate errors"""
    with pytest.raises(ValueError, match="Unsupported method"):
        model_fitting.fit_model('y ~ x1', data=test_data, method='invalid_method')

def test_formula_parsing():
    """Test formula parsing functionality"""
    model = model_fitting.GAMModel()
    
    # Test simple linear formula
    y_var, x_vars, smooth_terms = model._parse_formula('response ~ pred1 + pred2 + pred3')
    assert y_var == 'response'
    assert x_vars == ['pred1', 'pred2', 'pred3']
    assert smooth_terms == []
    
    # Test formula with smooth terms
    y_var, x_vars, smooth_terms = model._parse_formula('y ~ x1 + s(x2) + x3 + s(x4)')
    assert y_var == 'y'
    assert x_vars == ['x1', 'x2', 'x3', 'x4']
    assert smooth_terms == ['x2', 'x4']
    
    # Test formula with only smooth terms
    y_var, x_vars, smooth_terms = model._parse_formula('outcome ~ s(feature1) + s(feature2)')
    assert y_var == 'outcome'
    assert x_vars == ['feature1', 'feature2']
    assert smooth_terms == ['feature1', 'feature2']

def test_formula_parsing_invalid():
    """Test that invalid formulas raise errors"""
    model = model_fitting.GAMModel()
    with pytest.raises(ValueError, match="Formula must be in format"):
        model._parse_formula('invalid formula')

def test_unfitted_model_errors():
    """Test that unfitted models raise appropriate errors"""
    model = model_fitting.OLSModel()
    
    with pytest.raises(ValueError, match="Model must be fitted first"):
        model.tidy()
    
    with pytest.raises(ValueError, match="Model must be fitted first"):
        model.glance()
    
    with pytest.raises(ValueError, match="Model must be fitted first"):
        model.augment()

def test_input_validation():
    """Test input validation for fit_xy methods"""
    
    # Test valid inputs
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    model_fitting._validate_xy_inputs(X, y)  # Should not raise
    
    # Test mismatched lengths
    with pytest.raises(ValueError, match="must have same length"):
        model_fitting._validate_xy_inputs(X, np.random.randn(5))
    
    # Test wrong dimensions
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        model_fitting._validate_xy_inputs(np.random.randn(10), y)
    
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        model_fitting._validate_xy_inputs(X, np.random.randn(10, 2))
    
    # Test wrong types
    with pytest.raises(TypeError, match="must be a numpy array"):
        model_fitting._validate_xy_inputs([[1, 2], [3, 4]], y)

def test_augment_xy_fitting(test_matrices):
    """Test augment method works with matrix-based fitting"""
    X, y = test_matrices
    model = model_fitting.fit_model_xy(X, y, method='ols', feature_names=['var1', 'var2'])
    
    aug_df = model.augment()
    assert isinstance(aug_df, pd.DataFrame)
    assert len(aug_df) == len(y)
    assert '.fitted' in aug_df.columns
    assert '.resid' in aug_df.columns
    assert 'y' in aug_df.columns
    assert 'var1' in aug_df.columns
    assert 'var2' in aug_df.columns

def test_residual_stats_calculation():
    """Test utility function for calculating residual statistics"""
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    stats = model_fitting._calculate_residual_stats(y_true, y_pred)
    
    assert 'r_squared' in stats
    assert 'ss_res' in stats
    assert 'ss_tot' in stats
    assert 'residuals' in stats
    assert len(stats['residuals']) == len(y_true)
    assert stats['r_squared'] > 0.9  # Should be high with good predictions