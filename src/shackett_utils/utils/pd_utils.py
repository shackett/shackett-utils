import numpy as np
import pandas as pd
from typing import Optional


def format_numeric_columns(
    df: pd.DataFrame, 
    format_spec: str = "{:.3f}", 
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Format all numeric columns in a DataFrame with a specified format.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    format_spec : str, default="{:.3f}"
        Format specification string (e.g., "{:.3f}", "{:.2%}", "{:,.0f}")
    inplace : bool, default=False
        If True, modify the original DataFrame. If False, return a copy.
    
    Returns
    -------
    pd.DataFrame or None
        Formatted DataFrame if inplace=False, None if inplace=True
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1.23456, 2.34567, 3.45678],
    ...     'B': [0.123, 0.456, 0.789],
    ...     'C': ['text1', 'text2', 'text3']
    ... })
    >>> formatted_df = format_numeric_columns(df, "{:.2f}")
    >>> print(formatted_df)
           A     B      C
    0  1.23  0.12  text1
    1  2.35  0.46  text2
    2  3.46  0.79  text3
    """
    if inplace:
        result_df = df
    else:
        result_df = df.copy()
    
    # Get a list of all numeric columns
    numeric_cols = result_df.select_dtypes(include=np.number).columns
    
    # Loop through the numeric columns and apply the formatting
    for col in numeric_cols:
        result_df[col] = result_df[col].map(format_spec.format)
    
    if inplace:
        return None
    else:
        return result_df