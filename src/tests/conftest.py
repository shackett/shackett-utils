"""
Global pytest configuration and fixtures.
"""
import pytest
import warnings

# Configure pytest to ignore specific numpy deprecation warnings
@pytest.fixture(autouse=True)
def ignore_numpy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="pandas.core.dtypes.cast",
            message=".*find_common_type is deprecated.*"
        )
        yield 