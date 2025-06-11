"""Simple lazy import system with manual package->extra mapping."""

import importlib
from functools import lru_cache
from typing import Dict

# Simple mapping of import names to extras
PACKAGE_TO_EXTRA: Dict[str, str] = {
    # Data
    "pandas": "data",
    "numpy": "data",
    # Viz
    "matplotlib": "viz",
    "seaborn": "viz",
    # Genomics
    "adata": "genomics",
    "mudata": "genomics",
    "scanpy": "genomics",
    # statistics
    "statsmodels": "statistics",
}


def get_package(package_name: str):
    """
    Import a package with helpful error message if missing.

    Args:
        package_name: The package to import (e.g., 'pandas', 'bs4')

    Returns:
        The imported package

    Raises:
        ImportError: With install instructions for the relevant extra
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        if package_name not in PACKAGE_TO_EXTRA.keys():
            raise ImportError(
                f"Package {package_name} is not bundled with shackett-utils. Please install it manually."
            )

        extra = PACKAGE_TO_EXTRA[package_name]
        raise ImportError(
            f"Install {extra} extras: pip install shackett-utils[{extra}]"
        )


def create_package_getter(package_name: str):
    """
    Create a cached package getter function.

    Args:
        package_name: The package to import

    Returns:
        A cached function that returns the package
    """

    @lru_cache(maxsize=1)
    def _get_package():
        return get_package(package_name)

    return _get_package


# Pre-created getters for common packages
_get_pandas = create_package_getter("pandas")
_get_numpy = create_package_getter("numpy")
_get_matplotlib_pyplot = create_package_getter("matplotlib.pyplot")
_get_seaborn = create_package_getter("seaborn")
_get_adata = create_package_getter("adata")
_get_mudata = create_package_getter("mudata")
_get_scanpy = create_package_getter("scanpy")
_get_statsmodels = create_package_getter("statsmodels")
