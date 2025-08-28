import pytest
import pandas as pd
import numpy as np
from shackett_utils.utils.pd_utils import format_numeric_columns


class TestFormatNumericColumns:
    """Test cases for format_numeric_columns function."""

    def test_basic_formatting(self):
        """Test basic numeric formatting with default 3 decimal places."""
        df = pd.DataFrame({
            'A': [1.23456, 2.34567, 3.45678],
            'B': [0.123, 0.456, 0.789],
            'C': ['text1', 'text2', 'text3']
        })
        
        result = format_numeric_columns(df)
        
        # Check that numeric columns are formatted
        assert result['A'].iloc[0] == '1.235'
        assert result['B'].iloc[0] == '0.123'
        # Check that non-numeric columns are unchanged
        assert result['C'].iloc[0] == 'text1'

    def test_scientific_notation(self):
        """Test that very small values are converted to scientific notation."""
        df = pd.DataFrame({
            'A': [1.23e-10, 2.34e-8, 3.45e-6],
            'B': [0.000001, 0.0000001, 0.00000001],
            'C': ['text1', 'text2', 'text3']
        })
        
        result = format_numeric_columns(df, "{:.2e}")
        
        # Check that small values are formatted in scientific notation
        assert result['A'].iloc[0] == '1.23e-10'
        assert result['A'].iloc[1] == '2.34e-08'
        assert result['A'].iloc[2] == '3.45e-06'
        assert result['B'].iloc[0] == '1.00e-06'
        assert result['B'].iloc[1] == '1.00e-07'
        assert result['B'].iloc[2] == '1.00e-08'

    def test_inplace_modification(self):
        """Test inplace modification of the original DataFrame."""
        df = pd.DataFrame({
            'A': [1.23456, 2.34567, 3.45678],
            'B': ['text1', 'text2', 'text3']
        })
        
        original_id = id(df)
        result = format_numeric_columns(df, inplace=True)
        
        # Check that result is None (inplace operation)
        assert result is None
        # Check that the original DataFrame was modified
        assert id(df) == original_id
        assert df['A'].iloc[0] == '1.235'

    def test_mixed_data_types(self):
        """Test DataFrame with various data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'string_col': ['a', 'b', 'c'],
            'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        
        result = format_numeric_columns(df, "{:.1f}")
        
        # Check that only numeric columns are formatted
        assert result['int_col'].iloc[0] == '1.0'
        assert result['float_col'].iloc[0] == '1.1'
        # Check that non-numeric columns are unchanged
        assert result['string_col'].iloc[0] == 'a'
        assert result['datetime_col'].iloc[0] == pd.Timestamp('2023-01-01')
