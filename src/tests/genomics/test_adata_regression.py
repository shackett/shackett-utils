import pytest
import numpy as np
import pandas as pd
import anndata as ad
from shackett_utils.genomics import adata_regression
from shackett_utils.statistics.constants import STATISTICS_DEFS, TIDY_DEFS
from shackett_utils.statistics.utils import get_stat_abbreviation


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
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"batch": batch, "condition": condition}))

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


def test_adata_regression_integer_covariates(minimal_adata):
    """Test that AnnData regression works with integer covariates"""
    # Test OLS
    results_ols = adata_regression.adata_model_fitting(
        minimal_adata,
        formula="~ batch + condition",
        model_class="ols",
        n_jobs=1,
        allow_failures=False,
    )

    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0
    assert "feature_name" in results_ols.columns
    assert "term" in results_ols.columns
    assert "estimate" in results_ols.columns

    # Test GAM
    results_gam = adata_regression.adata_model_fitting(
        minimal_adata,
        formula="~ batch + s(condition)",
        model_class="gam",
        n_jobs=1,
        allow_failures=False,
    )

    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0
    assert "feature_name" in results_gam.columns
    assert "term" in results_gam.columns


def test_adata_regression_dtype_handling(minimal_adata):
    """Test that AnnData regression handles different data types correctly"""
    import numpy as np
    import pandas as pd

    # Add columns with different dtypes to obs
    minimal_adata.obs["numerical"] = pd.Series(
        np.random.randn(minimal_adata.n_obs), dtype="float64"
    )
    minimal_adata.obs["integer"] = pd.Series(
        np.random.randint(0, 10, minimal_adata.n_obs), dtype="int64"
    )
    minimal_adata.obs["category"] = pd.Categorical(
        ["A", "B"] * (minimal_adata.n_obs // 2)
    )
    minimal_adata.obs["object"] = pd.Series(
        ["X", "Y"] * (minimal_adata.n_obs // 2), dtype="object"
    )
    minimal_adata.obs["bool"] = pd.Series(
        np.random.choice([True, False], minimal_adata.n_obs)
    )

    # Print dtypes to help debug
    print("\nData types in test DataFrame:")
    print(minimal_adata.obs.dtypes)

    # Test with different formula combinations
    formulas = [
        "~ numerical + integer",  # numeric only
        "~ numerical + category",  # with categorical
        "~ integer + bool",  # with boolean
        "~ numerical + object",  # with object dtype
        "~ s(numerical) + integer",  # GAM with numeric
        "~ s(integer) + numerical",  # GAM with integer
    ]

    for formula in formulas:
        print(f"\nTesting formula: {formula}")
        try:
            if "s(" in formula:
                model_class = "gam"
            else:
                model_class = "ols"

            results = adata_regression.adata_model_fitting(
                minimal_adata,
                formula=formula,
                model_class=model_class,
                n_jobs=1,
                allow_failures=False,
            )
            print(f"Success with formula: {formula}")
            print(f"Results shape: {results.shape}")
            print(f"Results columns: {results.columns}")
        except Exception as e:
            print(f"Error with formula {formula}: {str(e)}")
            print("Data types of variables in formula:")
            for var in formula.split("~")[1].strip().split("+"):
                var = var.strip()
                if "s(" in var:
                    var = var[2:-1]  # Remove s( and )
                print(f"  {var}: {minimal_adata.obs[var].dtype}")


def test_adata_regression_model_class_inference(minimal_adata):
    """Test that model class is correctly inferred from formula"""
    # Test auto-detection of OLS
    results_ols = adata_regression.adata_model_fitting(
        minimal_adata,
        formula="~ batch + condition",  # Linear terms only
        n_jobs=1,
        allow_failures=False,
    )
    assert isinstance(results_ols, pd.DataFrame)
    assert len(results_ols) > 0

    # Test auto-detection of GAM
    results_gam = adata_regression.adata_model_fitting(
        minimal_adata,
        formula="~ batch + s(condition)",  # Contains smooth term
        n_jobs=1,
        allow_failures=False,
    )
    assert isinstance(results_gam, pd.DataFrame)
    assert len(results_gam) > 0

    # Test explicit model class still works
    results_explicit = adata_regression.adata_model_fitting(
        minimal_adata,
        formula="~ batch + condition",
        model_class="gam",  # Explicitly set GAM
        n_jobs=1,
        allow_failures=False,
    )
    assert isinstance(results_explicit, pd.DataFrame)
    assert len(results_explicit) > 0

    # Test that OLS with smooth terms raises error
    with pytest.raises(ValueError, match="Cannot fit OLS model with smooth terms"):
        adata_regression.adata_model_fitting(
            minimal_adata,
            formula="~ batch + s(condition)",
            model_class="ols",  # Try to force OLS with smooth terms
            n_jobs=1,
            allow_failures=False,
        )


def test_add_regression_results_to_anndata(minimal_adata):
    """Test adding regression results to AnnData."""
    # Create sample regression results
    results_df = pd.DataFrame(
        {
            STATISTICS_DEFS.FEATURE_NAME: ["gene_0", "gene_1", "gene_0", "gene_1"],
            TIDY_DEFS.TERM: ["batch", "batch", "condition", "condition"],
            TIDY_DEFS.ESTIMATE: [0.5, -0.3, 0.2, 0.4],
            TIDY_DEFS.STD_ERROR: [0.1, 0.1, 0.1, 0.1],
            STATISTICS_DEFS.P_VALUE: [0.01, 0.02, 0.03, 0.04],
            TIDY_DEFS.LOG10_P_VALUE: [-2, -1.7, -1.5, -1.3],
            STATISTICS_DEFS.Q_VALUE: [0.02, 0.03, 0.04, 0.05],
            TIDY_DEFS.STATISTIC: [2.5, -2.0, 1.5, 2.0],
        }
    )

    # Test with default settings
    adata = minimal_adata.copy()
    adata_regression.add_regression_results_to_anndata(
        adata=adata, results_df=results_df, inplace=True
    )

    # Check that results are stored in uns
    assert "regression_results" in adata.uns

    # Check that all expected columns are present (derive from constants)
    terms = ["batch", "condition"]
    stats_with_abbrev = [
        (TIDY_DEFS.ESTIMATE, get_stat_abbreviation(TIDY_DEFS.ESTIMATE)),
        (STATISTICS_DEFS.P_VALUE, get_stat_abbreviation(STATISTICS_DEFS.P_VALUE)),
        (STATISTICS_DEFS.Q_VALUE, get_stat_abbreviation(STATISTICS_DEFS.Q_VALUE)),
        (TIDY_DEFS.LOG10_P_VALUE, get_stat_abbreviation(TIDY_DEFS.LOG10_P_VALUE)),
        (TIDY_DEFS.STATISTIC, get_stat_abbreviation(TIDY_DEFS.STATISTIC)),
        (TIDY_DEFS.STD_ERROR, get_stat_abbreviation(TIDY_DEFS.STD_ERROR)),
    ]
    expected_cols = {f"{abbrev}_{t}" for _, abbrev in stats_with_abbrev for t in terms}
    assert all(col in adata.var.columns for col in expected_cols)

    # Check specific values
    est_prefix = get_stat_abbreviation(TIDY_DEFS.ESTIMATE)
    assert adata.var.loc["gene_0", f"{est_prefix}_batch"] == 0.5
    assert adata.var.loc["gene_1", f"{est_prefix}_condition"] == 0.4
    assert np.isnan(
        adata.var.loc["gene_2", f"{est_prefix}_batch"]
    )  # Gene not in results

    # Test with specific stats
    adata = minimal_adata.copy()
    selected_stats = [TIDY_DEFS.ESTIMATE, STATISTICS_DEFS.P_VALUE]
    adata_regression.add_regression_results_to_anndata(
        adata=adata, results_df=results_df, stats_to_add=selected_stats, inplace=True
    )

    # Check that only selected stats are present
    expected_cols = {
        f"{get_stat_abbreviation(TIDY_DEFS.ESTIMATE)}_{t}" for t in terms
    } | {f"{get_stat_abbreviation(STATISTICS_DEFS.P_VALUE)}_{t}" for t in terms}
    unexpected_cols = (
        {f"{get_stat_abbreviation(STATISTICS_DEFS.Q_VALUE)}_{t}" for t in terms}
        | {f"{get_stat_abbreviation(TIDY_DEFS.STATISTIC)}_{t}" for t in terms}
        | {f"{get_stat_abbreviation(TIDY_DEFS.STD_ERROR)}_{t}" for t in terms}
    )
    assert all(col in adata.var.columns for col in expected_cols)
    assert not any(col in adata.var.columns for col in unexpected_cols)


def test_add_regression_results_errors():
    """Test error handling in add_regression_results_to_anndata."""
    # Create minimal AnnData
    adata = ad.AnnData(
        X=np.random.randn(10, 5),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )

    # Test empty results
    empty_results = pd.DataFrame(
        columns=[STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE]
    )
    with pytest.raises(ValueError, match="Results DataFrame is empty"):
        adata_regression.add_regression_results_to_anndata(
            adata=adata, results_df=empty_results
        )

    # Test duplicate features for a term
    duplicate_results = pd.DataFrame(
        {
            STATISTICS_DEFS.FEATURE_NAME: ["gene_0", "gene_0"],  # Duplicate
            TIDY_DEFS.TERM: ["batch", "batch"],
            TIDY_DEFS.ESTIMATE: [0.5, 0.6],
            STATISTICS_DEFS.P_VALUE: [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="Found duplicate features for term 'batch'"):
        adata_regression.add_regression_results_to_anndata(
            adata=adata, results_df=duplicate_results
        )

    # Test invalid stats
    valid_results = pd.DataFrame(
        {
            STATISTICS_DEFS.FEATURE_NAME: ["gene_0", "gene_1"],
            TIDY_DEFS.TERM: ["batch", "batch"],
            TIDY_DEFS.ESTIMATE: [0.5, 0.6],
            STATISTICS_DEFS.P_VALUE: [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="Invalid statistics requested"):
        adata_regression.add_regression_results_to_anndata(
            adata=adata, results_df=valid_results, stats_to_add=["invalid_stat"]
        )


def test_add_regression_results_inplace():
    """Test inplace parameter of add_regression_results_to_anndata."""
    # Create minimal AnnData
    adata = ad.AnnData(
        X=np.random.randn(10, 5),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )

    # Create sample results
    results_df = pd.DataFrame(
        {
            STATISTICS_DEFS.FEATURE_NAME: ["gene_0", "gene_1"],
            TIDY_DEFS.TERM: ["batch", "batch"],
            TIDY_DEFS.ESTIMATE: [0.5, 0.6],
            STATISTICS_DEFS.P_VALUE: [0.01, 0.02],
        }
    )

    # Test inplace=True
    adata_copy = adata.copy()
    result = adata_regression.add_regression_results_to_anndata(
        adata=adata_copy, results_df=results_df, inplace=True
    )
    assert result is None
    assert "regression_results" in adata_copy.uns
    assert "est_batch" in adata_copy.var.columns

    # Test inplace=False
    result = adata_regression.add_regression_results_to_anndata(
        adata=adata, results_df=results_df, inplace=False
    )
    assert result is not None
    assert "regression_results" in result.uns
    assert "est_batch" in result.var.columns
    assert "regression_results" not in adata.uns
    assert "est_batch" not in adata.var.columns


def test_add_regression_results_significance_mask(minimal_adata):
    """Test significance mask functionality with fdr_cutoff."""
    # Create sample regression results
    results_df = pd.DataFrame(
        {
            STATISTICS_DEFS.FEATURE_NAME: ["gene_0", "gene_1", "gene_0", "gene_1"],
            TIDY_DEFS.TERM: ["batch", "batch", "condition", "condition"],
            TIDY_DEFS.ESTIMATE: [0.5, -0.3, 0.2, 0.4],
            STATISTICS_DEFS.P_VALUE: [0.01, 0.06, 0.03, 0.04],
            STATISTICS_DEFS.Q_VALUE: [0.02, 0.08, 0.04, 0.05],
        }
    )

    # Test with fdr_cutoff and q-values
    adata = minimal_adata.copy()
    adata_regression.add_regression_results_to_anndata(
        adata=adata, results_df=results_df, fdr_cutoff=0.05, inplace=True
    )

    # Check significance masks are present and correct
    assert "sig_batch" in adata.var.columns
    assert "sig_condition" in adata.var.columns
    assert adata.var.loc["gene_0", "sig_batch"]  # q-value 0.02 < 0.05
    assert not adata.var.loc["gene_1", "sig_batch"]  # q-value 0.08 > 0.05
    assert adata.var.loc["gene_0", "sig_condition"]  # q-value 0.04 < 0.05
    assert not adata.var.loc["gene_1", "sig_condition"]  # q-value 0.05 = 0.05

    # Test without q-values
    results_df_no_q = results_df.drop(columns=[STATISTICS_DEFS.Q_VALUE])
    adata = minimal_adata.copy()
    adata_regression.add_regression_results_to_anndata(
        adata=adata, results_df=results_df_no_q, fdr_cutoff=0.05, inplace=True
    )

    # Check no significance masks are created when q-values are not available
    assert "sig_batch" not in adata.var.columns
    assert "sig_condition" not in adata.var.columns

    # Test without fdr_cutoff
    adata = minimal_adata.copy()
    adata_regression.add_regression_results_to_anndata(
        adata=adata, results_df=results_df, inplace=True
    )

    # Check no significance masks are present when no fdr_cutoff provided
    assert "sig_batch" not in adata.var.columns
    assert "sig_condition" not in adata.var.columns
