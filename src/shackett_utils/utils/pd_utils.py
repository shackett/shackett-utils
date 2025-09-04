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


def format_character_columns(
    df: pd.DataFrame,
    wrap_length: int = 30,
    truncate_length: int = 120,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Format character columns by wrapping long strings and truncating very long ones.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    wrap_length : int, default=30
        Maximum length before wrapping to new lines
    truncate_length : int, default=120
        Maximum length before truncating with "..."
    inplace : bool, default=False
        If True, modify the original DataFrame. If False, return a copy.
    
    Returns
    -------
    pd.DataFrame or None
        Formatted DataFrame if inplace=False, None if inplace=True
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': ['Short text', 'This is a longer text that will be wrapped', 'Very long text...'],
    ...     'B': [1, 2, 3]
    ... })
    >>> formatted_df = format_character_columns(df, wrap_length=20, truncate_length=50)
    """
    if inplace:
        result_df = df
    else:
        result_df = df.copy()
    
    def format_string(s):
        if pd.isna(s) or not isinstance(s, str):
            return s
        
        # Truncate if too long
        if len(s) > truncate_length:
            s = s[:truncate_length-3] + "..."
        
        # Wrap if longer than wrap_length
        if len(s) > wrap_length:
            # Simple word wrapping - break at spaces when possible
            words = s.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= wrap_length:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        # Word itself is longer than wrap_length, break it
                        lines.append(word[:wrap_length])
                        current_line = word[wrap_length:]
            
            if current_line:
                lines.append(current_line)
            
            return "\n".join(lines)
        
        return s
    
    # Get character/object columns (strings)
    char_cols = result_df.select_dtypes(include=['object', 'string']).columns
    
    # Apply formatting to each character column
    for col in char_cols:
        result_df[col] = result_df[col].apply(format_string)
    
    if inplace:
        return None
    else:
        return result_df