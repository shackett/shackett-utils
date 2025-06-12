"""Tests for the transform module."""

import numpy as np
import pandas as pd
import pytest

from shackett_utils.statistics.transform import (
    best_normalizing_transform,
    filter_valid_transforms,
    _is_valid_transform,
)


def test_transform_p_values():
    """Test that p-values are valid for appropriate transformations."""
    # Test with positive data - all transforms should work
    s = pd.Series([1, 2, 3, 4, 5])
    results = best_normalizing_transform(s)

    # All transforms should be valid and have p-values > 0
    for name in ["original", "log2", "boxcox", "sqrt", "yeo-johnson", "arcsinh"]:
        assert not np.isnan(results[name]["p"]), f"{name} should have valid p-value"
        assert results[name]["p"] > 0, f"{name} should have p-value > 0"

    # Test with negative data - only some transforms should work
    s = pd.Series([-2, -1, 0, 1, 2])
    results = best_normalizing_transform(s)

    # These should have valid p-values
    for name in ["original", "yeo-johnson", "arcsinh"]:
        assert not np.isnan(results[name]["p"]), f"{name} should have valid p-value"
        assert results[name]["p"] > 0, f"{name} should have p-value > 0"

    # These should have NaN p-values
    for name in ["log2", "boxcox", "sqrt"]:
        assert np.isnan(results[name]["p"]), f"{name} should have NaN p-value"


def test_transform_filtering():
    """Test that filtering removes invalid transformations."""
    # Create data that will produce some invalid transformations
    s = pd.Series([1e-10, 1e-9, 1e9, 1e10])
    results = best_normalizing_transform(s)
    filtered = filter_valid_transforms(results)

    # Check that filtered results have valid values
    for name, result in filtered.items():
        if name == "best":
            continue
        if not np.isnan(result["p"]):
            # Should have valid p-value and transformed values
            assert result["p"] > 0, f"{name} should have p-value > 0"
            assert _is_valid_transform(
                result["transformed"]
            ), f"{name} should have valid values"


def test_edge_cases():
    """Test edge cases produce expected results."""
    # Single value should raise ValueError
    s = pd.Series([1])
    with pytest.raises(
        ValueError, match="Data must have at least 2 unique values for transformation"
    ):
        best_normalizing_transform(s)

    # Constant values should raise ValueError
    s = pd.Series([1, 1, 1])
    with pytest.raises(
        ValueError, match="Data must have at least 2 unique values for transformation"
    ):
        best_normalizing_transform(s)

    # Empty series should raise ValueError
    s = pd.Series([])
    with pytest.raises(
        ValueError, match="Data must have at least 2 unique values for transformation"
    ):
        best_normalizing_transform(s)

    # All NaN series should raise ValueError
    s = pd.Series([np.nan, np.nan])
    with pytest.raises(
        ValueError, match="Data must have at least 2 unique values for transformation"
    ):
        best_normalizing_transform(s)

    # All zeros - sqrt and original should work, log/boxcox should not
    s = pd.Series([0, 0, 0, 0])
    results = best_normalizing_transform(s)
    # These should be NaN due to zero variance
    for name in ["original", "log2", "boxcox", "sqrt", "yeo-johnson", "arcsinh"]:
        assert np.isnan(results[name]["p"]), f"{name} should have NaN p-value"
