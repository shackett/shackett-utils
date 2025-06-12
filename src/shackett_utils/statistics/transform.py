"""
Utilities for data transformations and normalization.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import kstest, boxcox
from sklearn.preprocessing import PowerTransformer

def _is_valid_transform(transformed_values: np.ndarray) -> bool:
    """
    Check if transformed values are reasonable.
    
    Parameters
    ----------
    transformed_values : np.ndarray
        The transformed values to validate
        
    Returns
    -------
    bool
        True if the transformation produced reasonable values (no extreme or non-finite values),
        False otherwise
    
    Notes
    -----
    A transformation is considered invalid if it:
    - Contains any values with absolute value > 1e10
    - Contains any non-finite values (NaN or ±Inf)
    """
    if not isinstance(transformed_values, np.ndarray):
        transformed_values = np.asarray(transformed_values)
    return not (np.any(np.abs(transformed_values) > 1e10) or 
               np.any(~np.isfinite(transformed_values)))


def filter_valid_transforms(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter transformation results to remove any with invalid values.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from best_normalizing_transform
        
    Returns
    -------
    Dict[str, Any]
        Filtered results with invalid transformations removed
        (their p-values set to NaN)
    """
    filtered = results.copy()
    
    # Check each transformation except 'best'
    for name in results:
        if name == 'best':
            continue
            
        # Get the transformed values using transform_func_map
        if name in transform_func_map and not np.isnan(results[name]["p"]):
            try:
                s = pd.Series(results[name]["transformed"])
                if not _is_valid_transform(s):
                    filtered[name]["p"] = np.nan
                    filtered[name]["stat"] = np.nan
            except Exception:
                filtered[name]["p"] = np.nan
                filtered[name]["stat"] = np.nan
    
    # Recompute best transformation
    valid_transforms = {
        k: v["p"] for k, v in filtered.items() 
        if k != "best" and not np.isnan(v["p"])
    }
    
    if valid_transforms:
        filtered["best"] = max(valid_transforms.items(), key=lambda x: x[1])[0]
    else:
        filtered["best"] = "original"
    
    return filtered


def best_normalizing_transform(series: pd.Series) -> Dict[str, Any]:
    """
    Find the best normalizing transformation for a series of values.
    
    Tests multiple transformations and selects the one that produces the most
    normally distributed results according to the Kolmogorov-Smirnov test.
    
    Parameters
    ----------
    series : pd.Series
        The data to transform
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing results for each transformation:
        - Keys are transformation names: 'original', 'log2', 'boxcox', 'sqrt',
          'yeo-johnson', 'arcsinh'
        - Values are dictionaries containing:
            - 'stat': KS test statistic
            - 'p': KS test p-value
        - Additional key 'best' contains the name of the best transformation
            
    Raises
    ------
    ValueError
        If the input series has less than 2 values or contains all identical non-zero values
    """
    s = series.dropna()
    
    # Check if data is too small
    if len(s) < 2:
        raise ValueError("Data must have at least 2 unique values for transformation.")
        
    # Check if data is constant and non-zero (allow zeros since they can be transformed)
    if np.all(s == s.iloc[0]) and s.iloc[0] != 0:
        raise ValueError("Data must have at least 2 unique values for transformation.")
        
    results = {}

    # Original
    stat, p = kstest((s - s.mean()) / s.std(), "norm")
    results["original"] = {"stat": stat, "p": p, "transformed": s}

    # log2 (only for positive values)
    if (s > 0).all():
        s_log2 = np.log2(s)
        stat, p = kstest((s_log2 - s_log2.mean()) / s_log2.std(), "norm")
        results["log2"] = {"stat": stat, "p": p, "transformed": s_log2}
        # Box-Cox
        try:
            s_boxcox, _ = boxcox(s)
            stat, p = kstest((s_boxcox - s_boxcox.mean()) / s_boxcox.std(), "norm")
            results["boxcox"] = {"stat": stat, "p": p, "transformed": s_boxcox}
        except ValueError:  # Handle case where boxcox fails
            results["boxcox"] = {"stat": np.nan, "p": np.nan, "transformed": None}
    else:
        results["log2"] = {"stat": np.nan, "p": np.nan, "transformed": None}
        results["boxcox"] = {"stat": np.nan, "p": np.nan, "transformed": None}

    # sqrt (only for non-negative values)
    if (s >= 0).all():
        s_sqrt = np.sqrt(s)
        stat, p = kstest((s_sqrt - s_sqrt.mean()) / s_sqrt.std(), "norm")
        results["sqrt"] = {"stat": stat, "p": p, "transformed": s_sqrt}
    else:
        results["sqrt"] = {"stat": np.nan, "p": np.nan, "transformed": None}

    # Yeo-Johnson (can handle negatives)
    try:
        pt = PowerTransformer(method="yeo-johnson")
        s_yeojohnson = pt.fit_transform(s.values.reshape(-1, 1)).flatten()
        stat, p = kstest(
            (s_yeojohnson - s_yeojohnson.mean()) / s_yeojohnson.std(), "norm"
        )
        results["yeo-johnson"] = {"stat": stat, "p": p, "transformed": s_yeojohnson}
    except Exception:
        results["yeo-johnson"] = {"stat": np.nan, "p": np.nan, "transformed": None}

    # Arcsinh (can handle negatives)
    s_arcsinh = np.arcsinh(s)
    stat, p = kstest((s_arcsinh - s_arcsinh.mean()) / s_arcsinh.std(), "norm")
    results["arcsinh"] = {"stat": stat, "p": p, "transformed": s_arcsinh}

    # Find the best (highest p-value)
    best = max(
        results, key=lambda k: results[k]["p"] if not np.isnan(results[k]["p"]) else -1
    )
    results["best"] = best

    return results


transform_func_map = {
    "original": lambda x: x,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "boxcox": lambda x: pd.Series(
        boxcox(x.dropna())[0], index=x.dropna().index
    ).reindex(x.index),
    "yeo-johnson": lambda x: pd.Series(
        PowerTransformer(method="yeo-johnson")
        .fit_transform(x.dropna().values.reshape(-1, 1))
        .flatten(),
        index=x.dropna().index,
    ).reindex(x.index),
    "arcsinh": np.arcsinh,
} 