import numpy as np
from scipy import stats

from shackett_utils.statistics import hypothesis_testing
from shackett_utils.statistics.constants import HYPOTHESIS_TESTING_DEFS


def test_calculate_tstat_pvalue():
    """Test calculate_tstat_pvalue function"""

    # Test two-tailed t-test
    stat = 2.0
    df = 100
    p_val = hypothesis_testing.calculate_tstat_pvalue(
        stat, df, test_type=HYPOTHESIS_TESTING_DEFS.TWO_TAILED
    )
    expected = 2 * (1 - stats.t.cdf(abs(stat), df))
    assert np.isclose(p_val, expected)

    # Test normal distribution (infinite df)
    p_val_norm = hypothesis_testing.calculate_tstat_pvalue(
        stat, np.inf, test_type=HYPOTHESIS_TESTING_DEFS.TWO_TAILED
    )
    expected_norm = 2 * (1 - stats.norm.cdf(abs(stat)))
    assert np.isclose(p_val_norm, expected_norm)

    # Test array inputs
    stats_array = np.array([1.0, 2.0, 3.0])
    p_vals = hypothesis_testing.calculate_tstat_pvalue(stats_array, df=50)
    for i, s in enumerate(stats_array):
        expected_i = 2 * (1 - stats.t.cdf(abs(s), 50))
        assert np.isclose(p_vals[i], expected_i)


def test_calculate_log_tstat_pvalue():
    """Test calculate_log_tstat_pvalue function"""

    # Test single values
    stat = 2.0
    df = 100

    log_p = hypothesis_testing.calculate_log_tstat_pvalue(stat, df, log_base=10)

    # Compare with manual calculation
    expected_p = 2 * (1 - stats.t.cdf(abs(stat), df))
    expected_log_p = np.log10(expected_p)

    assert np.isclose(log_p, expected_log_p), f"Expected {expected_log_p}, got {log_p}"

    # Test arrays
    stats_array = np.array([1.0, 2.0, 5.0])

    log_p_array = hypothesis_testing.calculate_log_tstat_pvalue(stats_array, df=50)

    # Check each element
    for i, s in enumerate(stats_array):
        expected_p_i = 2 * (1 - stats.t.cdf(abs(s), 50))
        expected_log_p_i = np.log10(expected_p_i)
        assert np.isclose(log_p_array[i], expected_log_p_i), f"Array element {i} failed"

    # Test extreme values that would underflow regular p-values
    extreme_stat = 30.0
    log_p_extreme = hypothesis_testing.calculate_log_tstat_pvalue(extreme_stat, df=1000)

    # Should be a very negative number (very small p-value)
    assert (
        log_p_extreme < -50
    ), f"Expected very negative log p-value, got {log_p_extreme}"

    # Test normal distribution (infinite df)
    log_p_norm = hypothesis_testing.calculate_log_tstat_pvalue(3.0, df=np.inf)
    expected_p_norm = 2 * (1 - stats.norm.cdf(3.0))
    expected_log_p_norm = np.log10(expected_p_norm)

    assert np.isclose(
        log_p_norm, expected_log_p_norm
    ), "Normal distribution test failed"

    # Test natural log base
    log_p_ln = hypothesis_testing.calculate_log_tstat_pvalue(2.0, df=100, log_base=np.e)
    expected_ln_p = np.log(2 * (1 - stats.t.cdf(2.0, 100)))

    assert np.isclose(log_p_ln, expected_ln_p), "Natural log test failed"


def test_quantile_to_pvalue():
    """Test quantile_to_pvalue function"""

    # Test one-tailed upper
    quantile = 0.975  # 97.5th percentile
    p_upper = hypothesis_testing.quantile_to_pvalue(
        quantile, HYPOTHESIS_TESTING_DEFS.ONE_TAILED_UPPER
    )
    assert np.isclose(p_upper, 0.025)  # 1 - 0.975 = 0.025

    # Test one-tailed lower
    p_lower = hypothesis_testing.quantile_to_pvalue(
        quantile, HYPOTHESIS_TESTING_DEFS.ONE_TAILED_LOWER
    )
    assert np.isclose(p_lower, 0.975)  # quantile itself

    # Test two-tailed
    p_two = hypothesis_testing.quantile_to_pvalue(
        quantile, HYPOTHESIS_TESTING_DEFS.TWO_TAILED
    )
    assert np.isclose(p_two, 0.05)  # 2 * min(0.975, 0.025) = 2 * 0.025 = 0.05

    # Test with symmetric quantile (should give same result for two-tailed)
    p_two_sym = hypothesis_testing.quantile_to_pvalue(
        0.025, HYPOTHESIS_TESTING_DEFS.TWO_TAILED
    )
    assert np.isclose(p_two_sym, 0.05)  # 2 * min(0.025, 0.975) = 2 * 0.025 = 0.05

    # Test arrays
    quantiles = np.array([0.9, 0.95, 0.99])
    p_vals = hypothesis_testing.quantile_to_pvalue(
        quantiles, HYPOTHESIS_TESTING_DEFS.ONE_TAILED_UPPER
    )
    expected = np.array([0.1, 0.05, 0.01])
    assert np.allclose(p_vals, expected)

    # Test edge cases
    assert (
        hypothesis_testing.quantile_to_pvalue(0.5, HYPOTHESIS_TESTING_DEFS.TWO_TAILED)
        == 1.0
    )  # median gives p=1
    assert (
        hypothesis_testing.quantile_to_pvalue(
            1.0, HYPOTHESIS_TESTING_DEFS.ONE_TAILED_UPPER
        )
        == 0.0
    )  # extreme quantile
    assert (
        hypothesis_testing.quantile_to_pvalue(
            0.0, HYPOTHESIS_TESTING_DEFS.ONE_TAILED_LOWER
        )
        == 0.0
    )  # extreme quantile
