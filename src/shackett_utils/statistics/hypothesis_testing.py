from typing import Union

import numpy as np
from scipy import stats

from shackett_utils.statistics.constants import HYPOTHESIS_TESTING_DEFS


def calculate_log_tstat_pvalue(
    statistic: Union[float, np.ndarray],
    df: Union[int, float],
    log_base: float = 10.0,
    test_type: str = HYPOTHESIS_TESTING_DEFS.TWO_TAILED,
) -> Union[float, np.ndarray]:
    """
    Calculate log p-values from test statistics to avoid floating point precision issues.

    Parameters
    ----------
    statistic : float or array-like
        Test statistic (e.g., t-statistic, z-statistic)
    df : int or float
        Degrees of freedom for the test statistic
        Use np.inf for z-distribution (normal approximation)
    log_base : float, default 10.0
        Base for logarithm (10 for log10, np.e for natural log)
    test_type : str, default "two-tailed"
        Type of test: "two-tailed", "one-tailed-upper", "one-tailed-lower"

    Returns
    -------
    float or np.ndarray
        Log p-value(s). Returns NaN for invalid inputs.

    Examples
    --------
    >>> # Simple case
    >>> log_p = calculate_log_pvalue(statistic=10.0, df=100)
    >>> print(f"log10(p-value) = {log_p:.2f}")

    >>> # Array case
    >>> stats_array = np.array([2.0, 10.0, 30.0])
    >>> se_array = np.array([0.5, 0.3, 0.1])
    >>> log_p_array = calculate_log_pvalue(stats_array, df=100)
    """

    # Convert inputs to arrays for consistent handling
    statistic = np.asarray(statistic)

    # Validate inputs
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    # Handle invalid statistics (NaN, infinite values)
    valid_mask = np.isfinite(statistic)

    # Initialize output array
    result = np.full_like(statistic, np.nan, dtype=float)

    if not np.any(valid_mask):
        return result if result.ndim > 0 else float(result)

    # Calculate for valid entries only
    valid_statistics = statistic[valid_mask] if statistic.ndim > 0 else statistic

    # Use appropriate distribution
    if np.isinf(df):
        # Normal distribution (z-test)
        if test_type == HYPOTHESIS_TESTING_DEFS.TWO_TAILED:
            log_p_valid = stats.norm.logcdf(-np.abs(valid_statistics)) + np.log(2)
        elif test_type == HYPOTHESIS_TESTING_DEFS.ONE_TAILED_UPPER:
            log_p_valid = stats.norm.logcdf(-valid_statistics)
        elif test_type == HYPOTHESIS_TESTING_DEFS.ONE_TAILED_LOWER:
            log_p_valid = stats.norm.logcdf(valid_statistics)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
    else:
        # t-distribution
        if test_type == HYPOTHESIS_TESTING_DEFS.TWO_TAILED:
            log_p_valid = stats.t.logcdf(-np.abs(valid_statistics), df) + np.log(2)
        elif test_type == HYPOTHESIS_TESTING_DEFS.ONE_TAILED_UPPER:
            log_p_valid = stats.t.logcdf(-valid_statistics, df)
        elif test_type == HYPOTHESIS_TESTING_DEFS.ONE_TAILED_LOWER:
            log_p_valid = stats.t.logcdf(valid_statistics, df)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    # Convert to specified base
    if log_base == np.e:
        log_p_base = log_p_valid
    else:
        log_p_base = log_p_valid / np.log(log_base)

    # Assign results back
    if result.ndim > 0:
        result[valid_mask] = log_p_base
    else:
        result = float(log_p_base)

    return result


def calculate_tstat_pvalue(
    statistic: Union[float, np.ndarray],
    df: Union[int, float],
    test_type: str = HYPOTHESIS_TESTING_DEFS.TWO_TAILED,
) -> Union[float, np.ndarray]:
    """
    Calculate p-values from test statistics using the same logic as when calculating log p-values.

    Parameters
    ----------
    statistic : float or array-like
        Test statistic (e.g., t-statistic, z-statistic)
    df : int or float
        Degrees of freedom for the test statistic
        Use np.inf for z-distribution (normal approximation)
    test_type : str, default "two-tailed"
        Type of test: "two-tailed", "one-tailed-upper", "one-tailed-lower"

    Returns
    -------
    float or np.ndarray
        P-value(s) corresponding to the test statistic(s).
        Returns NaN for invalid inputs.

    Examples
    --------
    >>> # Simple case
    >>> p_val = calculate_tstat_pvalue(statistic=10.0, df=100)
    >>> print(f"p-value = {p_val:.2f}")
    """

    # Convert inputs to arrays for consistent handling
    statistic = np.asarray(statistic)

    # Validate inputs
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    # Handle invalid statistics (NaN, infinite values)
    valid_mask = np.isfinite(statistic)

    # Initialize output array
    result = np.full_like(statistic, np.nan, dtype=float)

    if not np.any(valid_mask):
        return result if result.ndim > 0 else float(result)

    # Calculate for valid entries only
    valid_statistics = statistic[valid_mask] if statistic.ndim > 0 else statistic

    # Get the quantile from the distribution
    if np.isinf(df):
        quantiles = stats.norm.cdf(valid_statistics)
    else:
        quantiles = stats.t.cdf(valid_statistics, df)

    # Convert quantiles to p-values
    p_values = quantile_to_pvalue(quantiles, test_type)

    if result.ndim > 0:
        result[valid_mask] = p_values
    else:
        result = float(p_values)

    return result


def quantile_to_pvalue(
    quantile: Union[float, np.ndarray], test_type: str = "two-tailed"
) -> Union[float, np.ndarray]:
    """
    Convert a statistical quantile to a p-value.

    Parameters
    ----------
    quantile : float or array-like
        Quantile value(s) from a cumulative distribution function (CDF).
        Should be between 0 and 1.
    test_type : str, default "two-tailed"
        Type of test: "two-tailed", "one-tailed-upper", "one-tailed-lower"

    Returns
    -------
    float or np.ndarray
        P-value(s) corresponding to the quantile(s)

    Notes
    -----
    - For one-tailed-upper: p = 1 - quantile
    - For one-tailed-lower: p = quantile
    - For two-tailed: p = 2 * min(quantile, 1 - quantile)

    Examples
    --------
    >>> # If quantile = 0.975 (97.5th percentile)
    >>> quantile_to_pvalue(0.975, "one-tailed-upper")  # 0.025
    >>> quantile_to_pvalue(0.975, "two-tailed")        # 0.05
    """

    quantile = np.asarray(quantile)

    # Validate quantile values
    if np.any((quantile < 0) | (quantile > 1)):
        raise ValueError("Quantile values must be between 0 and 1")

    if test_type == "two-tailed":
        # Two-tailed: use the smaller tail area, then double it
        p_values = 2 * np.minimum(quantile, 1 - quantile)
    elif test_type == "one-tailed-upper":
        # Upper tail: p = 1 - quantile
        p_values = 1 - quantile
    elif test_type == "one-tailed-lower":
        # Lower tail: p = quantile
        p_values = quantile
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    return p_values if p_values.ndim > 0 else float(p_values)
