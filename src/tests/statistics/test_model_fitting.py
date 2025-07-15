import pytest
import pandas as pd
import numpy as np

from shackett_utils.statistics import model_fitting
from shackett_utils.statistics.model_fitting import _validate_tidy_df
from shackett_utils.statistics.constants import STATISTICS_DEFS


@pytest.fixture
def test_data():
    """Fixture to provide test data for pytest"""
    np.random.seed(123)
    n = 50
    data = pd.DataFrame(
        {
            "y": np.random.normal(5, 1, n),
            "x1": np.random.normal(0, 0.2, n),
            "x2": np.random.uniform(0, 5, n),
        }
    )
    data["y"] = 1 + 2 * data["x1"] + 0.5 * data["x2"] + np.random.normal(0, 0.5, n)
    return data


@pytest.fixture
def test_matrices():
    """Fixture to provide X, y matrices for testing"""
    np.random.seed(123)
    n = 50
    X = np.random.randn(n, 2)
    y = 1 + 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.5, n)
    return X, y


@pytest.fixture
def binary_test_data():
    """Fixture to provide binary response data for testing"""
    np.random.seed(123)
    n = 50
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.uniform(0, 5, n)
    logits = 1 + 2 * X1 + 0.5 * X2
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    return pd.DataFrame({"y": y, "x1": X1, "x2": X2})


@pytest.fixture
def count_test_data():
    """Fixture to provide count data for testing"""
    np.random.seed(123)
    n = 50
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.uniform(0, 5, n)
    lambda_ = np.exp(1 + 0.5 * X1 + 0.2 * X2)
    y = np.random.poisson(lambda_)
    return pd.DataFrame({"y": y, "x1": X1, "x2": X2})


def test_ols_creation(test_data):
    """Test OLS model creation and fitting"""
    model = model_fitting.fit_model("y ~ x1 + x2", data=test_data, method="ols")
    assert model.fitted_model is not None
    assert model.formula == "y ~ x1 + x2"


def test_ols_tidy(test_data):
    """Test OLS tidy output format"""
    model = model_fitting.fit_model("y ~ x1 + x2", data=test_data, method="ols")
    tidy_df = model.tidy()

    # Validate tidy DataFrame format
    _validate_tidy_df(tidy_df)
    assert len(tidy_df) == 3  # Intercept + 2 predictors

    # validate that log p-values are properly calculated
    for i, row in tidy_df.iterrows():
        assert np.isclose(row["log10_p_value"], np.log10(row["p_value"]))


def test_ols_glance(test_data):
    """Test OLS glance output format"""
    model = model_fitting.fit_model("y ~ x1 + x2", data=test_data, method="ols")
    glance_df = model.glance()
    assert isinstance(glance_df, pd.DataFrame)
    assert len(glance_df) == 1  # Single row
    required_cols = ["r_squared", "adj_r_squared", "aic", "bic", "nobs"]
    assert all(col in glance_df.columns for col in required_cols)
    assert glance_df["nobs"].iloc[0] == len(test_data)


def test_ols_augment(test_data):
    """Test OLS augment output format"""
    model = model_fitting.fit_model("y ~ x1 + x2", data=test_data, method="ols")
    aug_df = model.augment()
    assert isinstance(aug_df, pd.DataFrame)
    assert len(aug_df) == len(test_data)
    assert ".fitted" in aug_df.columns
    assert ".resid" in aug_df.columns
    # Check that original columns are preserved
    assert all(col in aug_df.columns for col in test_data.columns)


def test_ols_fit_xy(test_matrices):
    """Test OLS fitting with X, y matrices"""
    X, y = test_matrices
    model = model_fitting.fit_model_xy(X, y, method="ols", term_names=["var1", "var2"])
    assert model.fitted_model is not None
    assert model.term_names == ["var1", "var2"]

    # Test tidy output
    tidy_df = model.tidy()
    assert len(tidy_df) == 2  # 2 predictors
    assert "var1" in tidy_df["term"].values
    assert "var2" in tidy_df["term"].values


def test_gam_creation(test_data):
    """Test GAM model creation"""
    model = model_fitting.fit_model("y ~ x1 + x2", data=test_data, method="gam")
    assert model.fitted_model is not None
    assert model.formula == "y ~ x1 + x2"
    assert model.term_names == ["x1", "x2"]


def test_gam_formula_smooth_syntax(test_data):
    """Test GAM with smooth terms specified in formula"""
    # Test s() syntax in formula
    model = model_fitting.fit_model("y ~ x1 + s(x2)", data=test_data, method="gam")
    assert model.fitted_model is not None
    assert "x2" in model.smooth_terms
    assert "x1" not in model.smooth_terms  # x1 should be linear

    # Test multiple smooth terms
    model2 = model_fitting.fit_model("y ~ s(x1) + s(x2)", data=test_data, method="gam")
    assert set(model2.smooth_terms) == {"x1", "x2"}


def test_gam_tidy(test_data):
    """Test GAM tidy output format"""
    model = model_fitting.fit_model("y ~ x1 + s(x2)", data=test_data, method="gam")
    tidy_df = model.tidy()
    assert isinstance(tidy_df, pd.DataFrame)
    assert len(tidy_df) == 3  # 2 predictors + intercept
    assert "term" in tidy_df.columns
    assert "type" in tidy_df.columns
    assert "std_error" in tidy_df.columns
    assert "statistic" in tidy_df.columns

    # Check that x1 is linear and x2 is smooth
    term_types = dict(zip(tidy_df["term"], tidy_df["type"]))
    assert term_types["x1"] == "linear"
    assert term_types["x2"] == "smooth"

    # Check that linear terms have real statistics but smooth terms have NaN
    x1_row = tidy_df[tidy_df["term"] == "x1"].iloc[0]
    s_x2_row = tidy_df[tidy_df["term"] == "x2"].iloc[0]

    # Linear term should have real values for estimate, std_error, statistic
    assert isinstance(x1_row["estimate"], float)
    assert isinstance(x1_row["std_error"], float)
    assert isinstance(x1_row["statistic"], float)
    assert isinstance(x1_row["p_value"], float)

    # Smooth term: estimate, p_value may be real or NaN; std_error, statistic must be NaN
    for col in ["p_value"]:
        val = s_x2_row[col]
        assert np.isnan(val) or isinstance(
            val, float
        ), f"{col} for smooth term should be float or NaN, got {val}"
    for col in ["estimate", "std_error", "statistic"]:
        val = s_x2_row[col]
        assert np.isnan(val), f"{col} for smooth term should be NaN, got {val}"

    # validate that log p-values are properly calculated
    assert np.isclose(x1_row["log10_p_value"], np.log10(x1_row["p_value"]))


def test_gam_smooth_only(test_data):
    """Test GAM with only smooth terms (no linear terms)"""
    # Test with multiple smooth terms
    model2 = model_fitting.fit_model(
        "y ~ s(x1) + s(x2) + 0", data=test_data, method="gam"
    )
    FITTED_TERMS = {"x1", "x2"}  # explicitly no intercept

    assert model2.fitted_model is not None
    assert set(model2.smooth_terms) == FITTED_TERMS
    assert len(model2.term_names) == len(FITTED_TERMS)

    # Test tidy output
    tidy_df = model2.tidy()
    assert isinstance(tidy_df, pd.DataFrame)
    assert len(tidy_df) == len(FITTED_TERMS)
    assert all(
        row["type"] == "smooth" for _, row in tidy_df.iterrows()
    )  # all terms should be smooth
    assert set(tidy_df["term"]) == FITTED_TERMS

    # All smooth terms: estimate, p_value may be real or NaN; std_error, statistic must be NaN
    assert isinstance(tidy_df["p_value"].iloc[0], float)
    assert all(np.isnan(row["estimate"]) for _, row in tidy_df.iterrows())
    assert all(np.isnan(row["std_error"]) for _, row in tidy_df.iterrows())
    assert all(np.isnan(row["statistic"]) for _, row in tidy_df.iterrows())


def test_gam_glance(test_data):
    """Test GAM glance output format"""
    model = model_fitting.fit_model("y ~ x1 + s(x2)", data=test_data, method="gam")
    glance_df = model.glance()
    assert isinstance(glance_df, pd.DataFrame)
    assert len(glance_df) == 1
    assert "r_squared" in glance_df.columns
    assert "nobs" in glance_df.columns
    assert glance_df["nobs"].iloc[0] == len(test_data)


def test_gam_fit_xy(test_matrices):
    """Test GAM fitting with X, y matrices"""
    X, y = test_matrices
    with pytest.raises(
        NotImplementedError,
        match="Matrix-based fitting is not supported for GAM models",
    ):
        model_fitting.fit_model_xy(
            X, y, method="gam", term_names=["var1", "var2"], smooth_terms=["var1"]
        )


def test_model_registry_valid_methods(test_data):
    """Test that all valid methods work"""
    for method in ["ols", "lm", "linear", "gam", "smooth"]:
        model = model_fitting.fit_model("y ~ x1", data=test_data, method=method)
        assert model.fitted_model is not None


def test_model_registry_invalid_method(test_data):
    """Test that invalid methods raise appropriate errors"""
    with pytest.raises(ValueError, match="Unsupported method"):
        model_fitting.fit_model("y ~ x1", data=test_data, method="invalid_method")


def test_formula_parsing():
    """Test formula parsing functionality"""
    model = model_fitting.GAMModel()

    # Test simple linear formula
    y_var, x_vars, smooth_terms, fit_intercept = model._parse_formula(
        "response ~ pred1 + pred2 + pred3 + 1"
    )
    assert y_var == "response"
    assert x_vars == ["pred1", "pred2", "pred3"]
    assert smooth_terms == []
    assert fit_intercept

    # Test formula with smooth terms
    y_var, x_vars, smooth_terms, fit_intercept = model._parse_formula(
        "y ~ x1 + s(x2) + x3 + s(x4) + 0"
    )
    assert y_var == "y"
    assert x_vars == ["x1", "x2", "x3", "x4"]
    assert smooth_terms == ["x2", "x4"]
    assert not fit_intercept

    # Test formula with only smooth terms
    y_var, x_vars, smooth_terms, fit_intercept = model._parse_formula(
        "outcome ~ s(feature1) + s(feature2)"
    )
    assert y_var == "outcome"
    assert x_vars == ["feature1", "feature2"]
    assert smooth_terms == ["feature1", "feature2"]
    assert fit_intercept


def test_formula_parsing_invalid():
    """Test that invalid formulas raise errors"""
    model = model_fitting.GAMModel()
    with pytest.raises(ValueError, match="Formula must be in format"):
        model._parse_formula("invalid formula")


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
    model = model_fitting.fit_model_xy(X, y, method="ols", term_names=["var1", "var2"])

    aug_df = model.augment()
    assert isinstance(aug_df, pd.DataFrame)
    assert len(aug_df) == len(y)
    assert ".fitted" in aug_df.columns
    assert ".resid" in aug_df.columns
    assert "y" in aug_df.columns
    assert "var1" in aug_df.columns
    assert "var2" in aug_df.columns


def test_residual_stats_calculation():
    """Test utility function for calculating residual statistics"""

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

    stats = model_fitting._calculate_residual_stats(y_true, y_pred)

    assert "r_squared" in stats
    assert "ss_res" in stats
    assert "ss_tot" in stats
    assert "residuals" in stats
    assert len(stats["residuals"]) == len(y_true)
    assert stats["r_squared"] > 0.9  # Should be high with good predictions


def test_logistic_regression(binary_test_data):
    """Test logistic regression with both OLS and GAM"""
    # Fit logistic GAM
    model = model_fitting.fit_model(
        "y ~ x1 + x2", data=binary_test_data, method="gam", family="binomial"
    )
    assert model.fitted_model is not None
    assert model.family == "binomial"

    # Check predictions are probabilities
    preds = model.fitted_model.predict(binary_test_data[["x1", "x2"]].values)
    assert np.all((preds >= 0) & (preds <= 1))

    # Test with smooth terms
    model_smooth = model_fitting.fit_model(
        "y ~ x1 + s(x2)", data=binary_test_data, method="gam", family="binomial"
    )
    assert model_smooth.fitted_model is not None
    assert model_smooth.family == "binomial"
    assert "x2" in model_smooth.smooth_terms


def test_poisson_regression(count_test_data):
    """Test Poisson regression with GAM"""
    model = model_fitting.fit_model(
        "y ~ x1 + s(x2)", data=count_test_data, method="gam", family="poisson"
    )
    assert model.fitted_model is not None
    assert model.family == "poisson"

    # Check predictions are non-negative
    preds = model.fitted_model.predict(count_test_data[["x1", "x2"]].values)
    assert np.all(preds >= 0)


def test_invalid_family():
    """Test that invalid family raises appropriate error"""
    with pytest.raises(ValueError, match="Unsupported family"):
        model_fitting.fit_model(
            "y ~ x1",
            data=pd.DataFrame({"y": [1], "x1": [1]}),
            method="gam",
            family="invalid",
        )


def test_validate_tidy_df():
    """Test tidy DataFrame validation"""
    # Test empty DataFrame
    with pytest.raises(ValueError, match="must contain at least one row"):
        _validate_tidy_df(pd.DataFrame())

    # Test missing columns
    incomplete_df = pd.DataFrame(
        {
            "term": ["x1"],
            "estimate": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_tidy_df(incomplete_df)

    # Test valid DataFrame
    valid_df = pd.DataFrame(
        {
            "term": ["x1"],
            "estimate": [1.0],
            "std_error": [0.1],
            "statistic": [10.0],
            "p_value": [0.05],
            "conf_low": [0.8],
            "conf_high": [1.2],
        }
    )
    _validate_tidy_df(valid_df)  # Should not raise


def test_model_identifiers():
    """Test feature_name and model_name are correctly passed and ordered"""
    # Create test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    term_names = ["x1", "x2"]

    # Test OLS model with both identifiers
    model = model_fitting.OLSModel(feature_name="test_feature", model_name="test_model")
    model.fit_xy(X, y, term_names=term_names)

    # Test tidy output
    tidy_df = model.tidy()
    assert list(tidy_df.columns[:3]) == [
        STATISTICS_DEFS.MODEL_NAME,
        STATISTICS_DEFS.TERM,
        STATISTICS_DEFS.FEATURE_NAME,
    ]
    assert tidy_df[STATISTICS_DEFS.MODEL_NAME].unique() == ["test_model"]
    assert tidy_df[STATISTICS_DEFS.FEATURE_NAME].unique() == ["test_feature"]

    # Test glance output
    glance_df = model.glance()
    assert list(glance_df.columns[:2]) == [
        STATISTICS_DEFS.MODEL_NAME,
        STATISTICS_DEFS.FEATURE_NAME,
    ]
    assert glance_df[STATISTICS_DEFS.MODEL_NAME].unique() == ["test_model"]
    assert glance_df[STATISTICS_DEFS.FEATURE_NAME].unique() == ["test_feature"]

    # Test augment output
    augment_df = model.augment()
    assert list(augment_df.columns[:2]) == [
        STATISTICS_DEFS.MODEL_NAME,
        STATISTICS_DEFS.FEATURE_NAME,
    ]
    assert augment_df[STATISTICS_DEFS.MODEL_NAME].unique() == ["test_model"]
    assert augment_df[STATISTICS_DEFS.FEATURE_NAME].unique() == ["test_feature"]

    # Test with only feature_name
    model_feature_only = model_fitting.OLSModel(feature_name="test_feature")
    model_feature_only.fit_xy(X, y, term_names=term_names)
    tidy_feature_only = model_feature_only.tidy()
    assert STATISTICS_DEFS.MODEL_NAME not in tidy_feature_only.columns
    assert list(tidy_feature_only.columns[:2]) == [
        STATISTICS_DEFS.TERM,
        STATISTICS_DEFS.FEATURE_NAME,
    ]

    # Test with only model_name
    model_name_only = model_fitting.OLSModel(model_name="test_model")
    model_name_only.fit_xy(X, y, term_names=term_names)
    tidy_name_only = model_name_only.tidy()
    assert list(tidy_name_only.columns[:2]) == [
        STATISTICS_DEFS.MODEL_NAME,
        STATISTICS_DEFS.TERM,
    ]

    # Test with formula interface
    data = pd.DataFrame({"x1": [1, 3, 5], "x2": [2, 4, 6], "y": [1, 2, 3]})
    model_formula = model_fitting.OLSModel(
        feature_name="test_feature", model_name="test_model"
    )
    model_formula.fit("y ~ x1 + x2", data)
    tidy_formula = model_formula.tidy()
    assert list(tidy_formula.columns[:3]) == [
        STATISTICS_DEFS.MODEL_NAME,
        STATISTICS_DEFS.TERM,
        STATISTICS_DEFS.FEATURE_NAME,
    ]
