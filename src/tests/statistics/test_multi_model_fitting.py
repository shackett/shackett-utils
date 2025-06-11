import numpy as np
import pandas as pd
from shackett_utils.statistics import multi_model_fitting as mmf
from shackett_utils.statistics.constants import STATISTICS_DEFS, TIDY_DEFS
import pytest
import logging


@pytest.fixture
def test_data():
    """Create test data for model fitting"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create features with known relationships using explicit dtype
    X_features = np.random.randn(n_samples, n_features).astype(np.float64)
    X_model = np.random.randn(n_samples, 2).astype(np.float64)  # Two predictors

    # Create response with known relationships
    y = (
        1.0
        + 2.0 * X_model[:, 0]
        + 0.5 * X_model[:, 1]
        + np.random.normal(0, 0.1, n_samples)
    )
    X_features[:, 0] = y  # First feature has strong relationship

    # Create DataFrame for formula interface with explicit dtype
    data = pd.DataFrame(X_model, columns=["x1", "x2"]).astype(np.float64)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    term_names = ["x1", "x2"]

    # Add intercept to X_model for matrix interface
    X_model_with_intercept = np.column_stack(
        [np.ones(n_samples, dtype=np.float64), X_model]
    )
    term_names_with_intercept = ["intercept"] + term_names

    return {
        "data": data,
        "X_features": X_features,
        "X_model": X_model,
        "X_model_with_intercept": X_model_with_intercept,
        "feature_names": feature_names,
        "term_names": term_names,
        "term_names_with_intercept": term_names_with_intercept,
        "n_samples": n_samples,
    }


@pytest.fixture
def zero_var_data():
    """Create test data with zero variance features"""
    n_samples = 10
    X_features = np.zeros((n_samples, 2), dtype=np.float64)  # All zero variance
    X_model = np.random.randn(n_samples, 2).astype(np.float64)
    X_model_with_intercept = np.column_stack(
        [np.ones(n_samples, dtype=np.float64), X_model]
    )

    return {
        "X_features": X_features,
        "X_model": X_model,
        "X_model_with_intercept": X_model_with_intercept,
        "feature_names": ["f1", "f2"],
        "term_names": ["x1", "x2"],
        "term_names_with_intercept": ["intercept", "x1", "x2"],
        "n_samples": n_samples,
    }


@pytest.fixture
def missing_data():
    """Create test data with missing values"""
    np.random.seed(42)
    n_samples = 20

    # Create data with missing values using explicit dtype
    X_features = np.random.randn(n_samples, 3).astype(np.float64)
    X_model = np.random.randn(n_samples, 2).astype(np.float64)
    data = pd.DataFrame(X_model, columns=["x1", "x2"]).astype(np.float64)

    # Add missing values to different features
    X_features[0:5, 0] = np.nan  # 5 missing at start
    X_features[15:, 1] = np.nan  # 5 missing at end
    # Make middle feature fail by having perfect collinearity
    X_features[::2, 2] = np.nan  # Every other sample is missing
    X_features[1::2, 2] = 1.0  # All non-missing values are identical

    feature_names = ["missing_start", "missing_end", "missing_middle"]
    term_names = ["x1", "x2"]

    # Add intercept to X_model for matrix interface
    X_model_with_intercept = np.column_stack(
        [np.ones(n_samples, dtype=np.float64), X_model]
    )
    term_names_with_intercept = ["intercept"] + term_names

    return {
        "data": data,
        "X_features": X_features,
        "X_model": X_model,
        "X_model_with_intercept": X_model_with_intercept,
        "feature_names": feature_names,
        "term_names": term_names,
        "term_names_with_intercept": term_names_with_intercept,
        "n_samples": n_samples,
    }


@pytest.fixture
def integer_test_data():
    """Create test data with integer predictors"""
    np.random.seed(42)
    n_samples = 100

    # Create integer predictors
    x1 = np.random.randint(0, 10, n_samples)  # integers 0-9
    x2 = np.random.randint(-5, 5, n_samples)  # integers -5 to 4

    # Create response as linear combination plus noise
    y = 2 * x1 - 3 * x2 + np.random.normal(0, 1, n_samples)

    # Create feature matrix with 2 features
    X_features = np.column_stack([y, y + 1])  # Two similar features
    feature_names = ["feature1", "feature2"]

    # Create predictor DataFrame
    data = pd.DataFrame({"x1": x1, "x2": x2})

    return {"X_features": X_features, "data": data, "feature_names": feature_names}


def test_fit_feature_model_matrix(test_data):
    """Test matrix-based OLS fitting for a single feature"""
    results = mmf.fit_feature_model_matrix(
        test_data["X_features"][:, 0],
        test_data["X_model_with_intercept"],
        test_data["feature_names"][0],
        test_data["term_names_with_intercept"],
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(test_data["term_names_with_intercept"])
    assert all(
        col in results.columns
        for col in [
            STATISTICS_DEFS.FEATURE_NAME,
            TIDY_DEFS.TERM,
            TIDY_DEFS.ESTIMATE,
            TIDY_DEFS.STD_ERROR,
            TIDY_DEFS.P_VALUE,
        ]
    )
    assert (
        results.iloc[0][STATISTICS_DEFS.FEATURE_NAME] == test_data["feature_names"][0]
    )
    assert (
        results.iloc[0][STATISTICS_DEFS.TERM]
        == test_data["term_names_with_intercept"][0]
    )


def test_fit_feature_model_formula(test_data):
    """Test formula-based fitting for a single feature"""
    # Test OLS
    results_ols = mmf.fit_feature_model_formula(
        test_data["X_features"][:, 0],
        test_data["data"],
        test_data["feature_names"][0],
        formula="y ~ x1 + x2",
        model_class="ols",
    )

    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0
    assert all(
        col in results_ols.columns
        for col in [
            STATISTICS_DEFS.FEATURE_NAME,
            TIDY_DEFS.TERM,
            TIDY_DEFS.ESTIMATE,
            TIDY_DEFS.STD_ERROR,
            TIDY_DEFS.P_VALUE,
        ]
    )

    # Test GAM
    results_gam = mmf.fit_feature_model_formula(
        test_data["X_features"][:, 0],
        test_data["data"],
        test_data["feature_names"][0],
        formula="y ~ x1 + s(x2)",
        model_class="gam",
    )

    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0
    assert all(
        col in results_gam.columns
        for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM]
    )


def test_fit_parallel_models_matrix(test_data):
    """Test parallel matrix-based OLS fitting"""
    results_df = mmf.fit_parallel_models_matrix(
        test_data["X_features"],
        test_data["X_model_with_intercept"],
        test_data["feature_names"],
        test_data["term_names_with_intercept"],
        n_jobs=2,
        fdr_control=True,
    )

    assert isinstance(results_df, pd.DataFrame)
    # Each feature should have results for each term
    expected_rows = len(test_data["feature_names"]) * len(
        test_data["term_names_with_intercept"]
    )
    assert len(results_df) == expected_rows
    assert all(
        col in results_df.columns
        for col in [
            STATISTICS_DEFS.FEATURE_NAME,
            TIDY_DEFS.TERM,
            TIDY_DEFS.ESTIMATE,
            TIDY_DEFS.P_VALUE,
            STATISTICS_DEFS.Q_VALUE,
        ]
    )


def test_fit_parallel_models_formula(test_data):
    """Test parallel formula-based fitting"""
    # Test OLS
    results_df_ols = mmf.fit_parallel_models_formula(
        test_data["X_features"],
        test_data["data"],
        test_data["feature_names"],
        formula="y ~ x1 + x2",
        model_class="ols",
        n_jobs=2,
        fdr_control=True,
    )

    assert isinstance(results_df_ols, pd.DataFrame)
    assert len(results_df_ols) > 0
    assert all(
        col in results_df_ols.columns
        for col in [
            STATISTICS_DEFS.FEATURE_NAME,
            TIDY_DEFS.TERM,
            TIDY_DEFS.ESTIMATE,
            TIDY_DEFS.P_VALUE,
            STATISTICS_DEFS.Q_VALUE,
        ]
    )

    # Test GAM
    results_df_gam = mmf.fit_parallel_models_formula(
        test_data["X_features"],
        test_data["data"],
        test_data["feature_names"],
        formula="y ~ x1 + s(x2)",
        model_class="gam",
        n_jobs=2,
        fdr_control=True,
    )

    assert isinstance(results_df_gam, pd.DataFrame)
    assert len(results_df_gam) > 0
    assert all(
        col in results_df_gam.columns
        for col in [STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM]
    )


def test_zero_variance_features(zero_var_data):
    """Test handling of zero variance features"""
    # Matrix interface
    results_matrix = mmf.fit_parallel_models_matrix(
        zero_var_data["X_features"],
        zero_var_data["X_model_with_intercept"],
        zero_var_data["feature_names"],
        zero_var_data["term_names_with_intercept"],
        n_jobs=1,
    )
    assert isinstance(results_matrix, pd.DataFrame)
    assert len(results_matrix) == 0

    # Formula interface
    results_formula = mmf.fit_parallel_models_formula(
        zero_var_data["X_features"],
        zero_var_data["X_model"],
        zero_var_data["feature_names"],
        formula="y ~ x1 + x2",
        term_names=zero_var_data["term_names"],
        model_class="ols",
        n_jobs=1,
    )
    assert isinstance(results_formula, pd.DataFrame)
    assert len(results_formula) == 0


def test_missing_values_formula(missing_data, caplog):
    """Test handling of missing values in formula interface"""
    caplog.set_level("DEBUG")

    # Test single feature with missing values
    results = mmf.fit_feature_model_formula(
        missing_data["X_features"][:, 0],
        missing_data["data"],
        missing_data["feature_names"][0],
        formula="y ~ x1 + x2",
        model_class="ols",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "Filtering 5 missing values" in caplog.text

    # Test parallel fitting with different missing value patterns
    results_all = mmf.fit_parallel_models_formula(
        missing_data["X_features"],
        missing_data["data"],
        missing_data["feature_names"],
        formula="y ~ x1 + x2",
        model_class="ols",
        fdr_control=True,
    )

    assert isinstance(results_all, pd.DataFrame)
    assert len(results_all) > 0
    # Should have results for features with sufficient data
    assert (
        len(
            set(results_all[STATISTICS_DEFS.FEATURE_NAME].unique())
            & set(["missing_start", "missing_end"])
        )
        == 2
    )
    # Feature with too many missing values should be skipped
    assert "missing_middle" not in results_all[STATISTICS_DEFS.FEATURE_NAME].unique()


def test_missing_values_matrix(missing_data, caplog):
    """Test handling of missing values in matrix interface"""
    caplog.set_level("DEBUG")

    # Test single feature with missing values
    results = mmf.fit_feature_model_matrix(
        missing_data["X_features"][:, 0],
        missing_data["X_model_with_intercept"],
        missing_data["feature_names"][0],
        missing_data["term_names_with_intercept"],
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "Filtering 5 missing values" in caplog.text

    # Test parallel fitting with different missing value patterns
    results_all = mmf.fit_parallel_models_matrix(
        missing_data["X_features"],
        missing_data["X_model_with_intercept"],
        missing_data["feature_names"],
        missing_data["term_names_with_intercept"],
        fdr_control=True,
    )

    assert isinstance(results_all, pd.DataFrame)
    assert len(results_all) > 0
    # Should have results for features with sufficient data
    assert (
        len(
            set(results_all[STATISTICS_DEFS.FEATURE_NAME].unique())
            & set(["missing_start", "missing_end"])
        )
        == 2
    )
    # Feature with too many missing values should be skipped
    assert "missing_middle" not in results_all[STATISTICS_DEFS.FEATURE_NAME].unique()


def test_insufficient_samples():
    """Test handling of features with insufficient non-missing samples"""
    # Create data with only 1 valid sample
    X_features = np.array([[1.0], [np.nan], [np.nan]])
    X_model = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = pd.DataFrame(X_model, columns=["x1", "x2"])

    # Test formula interface
    results_formula = mmf.fit_feature_model_formula(
        X_features[:, 0],
        data,
        "insufficient_samples",
        formula="y ~ x1 + x2",
        model_class="ols",
    )
    assert len(results_formula) == 0

    # Test matrix interface
    results_matrix = mmf.fit_feature_model_matrix(
        X_features[:, 0], X_model, "insufficient_samples", ["x1", "x2"]
    )
    assert len(results_matrix) == 0


def test_input_validation():
    """Test input validation"""
    X_features = np.random.randn(10, 2)
    X_model = np.random.randn(5, 2)  # Mismatched samples
    data = pd.DataFrame(X_model, columns=["x1", "x2"])
    feature_names = ["f1", "f2"]
    term_names = ["x1", "x2"]

    # Matrix interface
    with pytest.raises(ValueError, match="must have same number of samples"):
        mmf.fit_parallel_models_matrix(X_features, X_model, feature_names, term_names)

    # Formula interface
    with pytest.raises(ValueError, match="must have same number of samples"):
        mmf.fit_parallel_models_formula(
            X_features, data, feature_names, formula="y ~ x1 + x2"
        )

    # Test mismatched feature names
    X_features_small = np.random.randn(5, 1)
    with pytest.raises(
        ValueError, match="Length of feature_names .* must match number of features"
    ):
        mmf.fit_parallel_models_formula(
            X_features_small, data, feature_names, formula="y ~ x1 + x2"
        )

    # Test unsupported model class
    with pytest.raises(ValueError, match="Unsupported model class"):
        mmf.fit_feature_model_formula(
            X_features_small[:, 0],
            data,
            "feature",
            formula="y ~ x1 + x2",
            model_class="unsupported",
        )


def test_progress_bar(test_data, caplog):
    """Test progress bar and logging"""
    caplog.set_level("INFO")

    # Test with progress bar
    _ = mmf.fit_parallel_models_matrix(
        test_data["X_features"],
        test_data["X_model_with_intercept"],
        test_data["feature_names"],
        test_data["term_names_with_intercept"],
        progress_bar=True,
        fdr_control=True,
    )
    assert any(
        "Starting parallel model fitting" in record.message for record in caplog.records
    )
    assert any("Completed model fitting" in record.message for record in caplog.records)

    caplog.clear()

    # Test without progress bar
    _ = mmf.fit_parallel_models_matrix(
        test_data["X_features"],
        test_data["X_model_with_intercept"],
        test_data["feature_names"],
        test_data["term_names_with_intercept"],
        progress_bar=False,
        fdr_control=True,
    )
    assert any(
        "Starting parallel model fitting" in record.message for record in caplog.records
    )
    assert any("Completed model fitting" in record.message for record in caplog.records)


def test_formula_validation():
    """Test formula validation and standardization"""
    # Test valid formulas
    assert mmf._validate_formula("y ~ x1 + x2") == "y ~ x1 + x2"
    assert mmf._validate_formula("~ x1 + x2") == "y ~ x1 + x2"

    # Test invalid formulas
    with pytest.raises(ValueError, match="must contain exactly one '~' character"):
        mmf._validate_formula("x1 + x2")

    with pytest.raises(ValueError, match="must use 'y' as dependent variable"):
        mmf._validate_formula("z ~ x1 + x2")

    with pytest.raises(ValueError, match="must contain exactly one '~' character"):
        mmf._validate_formula("y ~ x1 ~ x2")

    # Test formula with extra whitespace
    assert mmf._validate_formula("  y  ~  x1  +  x2  ") == "y ~ x1  +  x2"

    # Test numeric validation
    data = pd.DataFrame(
        {"y": [1, 2, 3], "x1": [4, 5, 6], "x2": [7, 8, 9], "cat": ["a", "b", "c"]}
    )

    # Valid numeric variables
    assert mmf._validate_formula("y ~ x1 + x2", data) == "y ~ x1 + x2"

    # Invalid categorical variable
    with pytest.raises(ValueError, match="must be numeric"):
        mmf._validate_formula("y ~ x1 + cat", data)

    # Missing variable
    with pytest.raises(ValueError, match="not found in data"):
        mmf._validate_formula("y ~ x1 + x3", data)

    # Test with smooth terms
    assert mmf._validate_formula("y ~ s(x1) + x2", data) == "y ~ s(x1) + x2"

    # Test with smooth terms and invalid variable
    with pytest.raises(ValueError, match="must be numeric"):
        mmf._validate_formula("y ~ s(cat) + x2", data)


def test_fit_parallel_models_formula_validation(test_data):
    """Test formula validation in fit_parallel_models_formula"""
    # Test with formula starting with ~
    results_rhs = mmf.fit_parallel_models_formula(
        test_data["X_features"],
        test_data["data"],
        test_data["feature_names"],
        formula="~ x1 + x2",
        model_class="ols",
        n_jobs=2,
    )
    assert isinstance(results_rhs, pd.DataFrame)
    assert len(results_rhs) > 0

    # Test with multiple ~ characters
    with pytest.raises(ValueError, match="must contain exactly one '~'"):
        mmf.fit_parallel_models_formula(
            test_data["X_features"],
            test_data["data"],
            test_data["feature_names"],
            formula="y ~ x1 ~ x2",
            model_class="ols",
            n_jobs=2,
        )

    # Test with invalid dependent variable
    with pytest.raises(ValueError, match="must use 'y' as dependent variable"):
        mmf.fit_parallel_models_formula(
            test_data["X_features"],
            test_data["data"],
            test_data["feature_names"],
            formula="z ~ x1 + x2",
            model_class="ols",
            n_jobs=2,
        )


def test_integer_predictors(integer_test_data):
    """Test that models work with integer predictors"""
    # Test OLS
    results_df_ols = mmf.fit_parallel_models_formula(
        integer_test_data["X_features"],
        integer_test_data["data"],
        integer_test_data["feature_names"],
        formula="~ x1 + x2",
        model_class="ols",
        n_jobs=1,
        allow_failures=False,
    )

    assert isinstance(results_df_ols, pd.DataFrame)
    assert len(results_df_ols) > 0

    # Test GAM
    results_df_gam = mmf.fit_parallel_models_formula(
        integer_test_data["X_features"],
        integer_test_data["data"],
        integer_test_data["feature_names"],
        formula="~ x1 + s(x2)",
        model_class="gam",
        n_jobs=1,
        allow_failures=False,
    )

    assert isinstance(results_df_gam, pd.DataFrame)
    assert len(results_df_gam) > 0


def test_dtype_handling():
    """Test handling of different data types in model fitting"""
    np.random.seed(42)
    n_samples = 100
    n_features = 2

    # Create test data with various dtypes
    X_features = np.random.randn(n_samples, n_features)

    # Create DataFrame with problematic dtypes
    data = pd.DataFrame(
        {
            "int_col": np.random.randint(0, 10, n_samples),  # integer
            "float_col": np.random.randn(n_samples),  # float
            "object_col": pd.Series(
                np.random.randn(n_samples), dtype="object"
            ),  # object dtype
            "category_col": pd.Categorical(
                np.random.choice(["A", "B"], n_samples)
            ),  # categorical
            "bool_col": np.random.choice([True, False], n_samples),  # boolean
            "string_col": np.random.choice(["X", "Y"], n_samples),  # string
        }
    )

    # Print dtypes to help debug
    print("\nData types in test DataFrame:")
    print(data.dtypes)

    feature_names = ["feature1", "feature2"]

    # Test with different formula combinations
    formulas = [
        "~ int_col + float_col",  # numeric only
        "~ int_col + object_col",  # with object dtype
        "~ float_col + category_col",  # with categorical
        "~ int_col + bool_col",  # with boolean
        "~ float_col + string_col",  # with string
    ]

    for formula in formulas:
        print(f"\nTesting formula: {formula}")
        try:
            mmf.fit_parallel_models_formula(
                X_features=X_features,
                data=data,
                feature_names=feature_names,
                formula=formula,
                model_class="ols",
                n_jobs=1,
                allow_failures=False,
            )
            print(f"Success with formula: {formula}")
        except Exception as e:
            print(f"Error with formula {formula}: {str(e)}")
            print("Data types of variables in formula:")
            for var in formula.split("~")[1].strip().split("+"):
                var = var.strip()
                print(f"  {var}: {data[var].dtype}")


def test_model_class_inference():
    """Test automatic model class inference from formula"""
    from shackett_utils.statistics import multi_model_fitting as mmf

    # Test OLS formulas
    assert mmf._detect_model_class_from_formula("y ~ x1 + x2") == "ols"
    assert mmf._detect_model_class_from_formula("~ x1 + x2") == "ols"
    assert mmf._detect_model_class_from_formula("x1 + x2") == "ols"
    assert mmf._detect_model_class_from_formula("y ~ x1 + x2 + x3 * x4") == "ols"

    # Test GAM formulas
    assert mmf._detect_model_class_from_formula("y ~ s(x1)") == "gam"
    assert mmf._detect_model_class_from_formula("y ~ x1 + s(x2)") == "gam"
    assert mmf._detect_model_class_from_formula("~ s(x1) + s(x2)") == "gam"
    assert mmf._detect_model_class_from_formula("y ~ s(x1) + x2 + s(x3)") == "gam"
    assert (
        mmf._detect_model_class_from_formula("y ~ s(x1, k=5) + x2") == "gam"
    )  # with spline parameters

    # Test edge cases
    assert (
        mmf._detect_model_class_from_formula("y ~ x1 + sin(x2)") == "ols"
    )  # function name containing 's'
    assert (
        mmf._detect_model_class_from_formula("y ~ x1 + stress(x2)") == "ols"
    )  # variable name containing 's'


def test_model_class_inference_in_parallel_fitting(test_data, caplog):
    """Test model class inference in fit_parallel_models_formula"""
    from shackett_utils.statistics import multi_model_fitting as mmf

    # Test auto-detection of OLS
    results_ols = mmf.fit_parallel_models_formula(
        test_data["X_features"],
        test_data["data"],
        test_data["feature_names"],
        formula="y ~ x1 + x2",
        n_jobs=1,
    )
    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0

    # Test auto-detection of GAM
    results_gam = mmf.fit_parallel_models_formula(
        test_data["X_features"],
        test_data["data"],
        test_data["feature_names"],
        formula="y ~ s(x1) + x2",
        n_jobs=1,
    )
    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0

    # Test that OLS with smooth terms raises error
    with pytest.raises(ValueError, match="Cannot fit OLS model with smooth terms"):
        mmf.fit_parallel_models_formula(
            test_data["X_features"],
            test_data["data"],
            test_data["feature_names"],
            formula="y ~ s(x1) + x2",
            model_class="ols",  # Should fail when trying to use OLS with smooth terms
            n_jobs=1,
        )

    # Test that GAM with OLS formula works (just a warning)
    with caplog.at_level(logging.WARNING):
        results_override = mmf.fit_parallel_models_formula(
            test_data["X_features"],
            test_data["data"],
            test_data["feature_names"],
            formula="y ~ x1 + x2",  # Linear terms only
            model_class="gam",  # Override to GAM despite no smooth terms
            n_jobs=1,
        )
        assert isinstance(results_override, pd.DataFrame)
        assert len(results_override) > 0
        assert "Model class mismatch" in caplog.text
