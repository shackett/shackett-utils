import pytest
import pandas as pd
import numpy as np
from shackett_utils.blog.html_utils import export_tabulator_payload


class TestExportTabulatorPayload:
    """Test cases for export_tabulator_payload function."""

    def test_simple_dataframe(self):
        """Test basic DataFrame with simple data."""
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Age': [25, 30, 35, 28],
            'Score': [85.5, 92.3, 78.1, 96.7],
            'Grade': ['B', 'A', 'C', 'A']
        }
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df)
        
        # Check structure
        assert 'table' in result
        assert 'columns' in result
        assert 'options' in result
        
        # Check table data
        assert len(result['table']) == 4
        assert result['table'][0]['Name'] == 'Alice'
        assert result['table'][0]['Age'] == 25
        assert result['table'][0]['Score'] == 85.5
        assert result['table'][0]['Grade'] == 'B'
        
        # Check columns
        assert len(result['columns']) == 4
        assert result['columns'][0]['title'] == 'Name'
        assert result['columns'][0]['field'] == 'Name'
        
        # Check options
        assert result['options']['layout'] == 'fitColumns'
        assert result['options']['responsiveLayout'] == 'collapse'

    def test_dataframe_without_index(self):
        """Test DataFrame without including index."""
        data = {
            'Product': ['Widget A', 'Widget B', 'Widget C'],
            'Price': [19.99, 24.99, 14.99], 
            'Stock': [100, 50, 200]
        }
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df, include_index=False)
        
        # Check that index is not included
        assert len(result['table']) == 3
        assert 'index' not in result['table'][0]
        assert result['table'][0]['Product'] == 'Widget A'
        assert result['table'][0]['Price'] == 19.99

    def test_dataframe_with_custom_index(self):
        """Test DataFrame with custom index."""
        data = {
            'Product': ['Widget A', 'Widget B', 'Widget C'],
            'Price': [19.99, 24.99, 14.99]
        }
        df = pd.DataFrame(data, index=['A1', 'B2', 'C3'])
        
        result = export_tabulator_payload(df, include_index=True)
        
        # Check that index is included in data
        assert len(result['table']) == 3
        assert result['table'][0]['index'] == 'A1'
        assert result['table'][0]['Product'] == 'Widget A'
        
        # Check that index column is included in column definitions
        assert len(result['columns']) == 3  # index + Product + Price
        assert result['columns'][0]['title'] == 'index'
        assert result['columns'][0]['field'] == 'index'
        assert result['columns'][1]['title'] == 'Product'
        assert result['columns'][2]['title'] == 'Price'

    def test_dataframe_with_named_index(self):
        """Test DataFrame with named index."""
        data = {
            'Product': ['Widget A', 'Widget B', 'Widget C'],
            'Price': [19.99, 24.99, 14.99]
        }
        df = pd.DataFrame(data, index=['A1', 'B2', 'C3'])
        df.index.name = 'ID'
        
        result = export_tabulator_payload(df, include_index=True)
        
        # Check that named index is included in data
        assert len(result['table']) == 3
        assert result['table'][0]['ID'] == 'A1'
        assert result['table'][0]['Product'] == 'Widget A'
        
        # Check that named index column is included in column definitions
        assert len(result['columns']) == 3  # ID + Product + Price
        assert result['columns'][0]['title'] == 'ID'
        assert result['columns'][0]['field'] == 'ID'
        assert result['columns'][1]['title'] == 'Product'
        assert result['columns'][2]['title'] == 'Price'

    def test_multiindex_rows(self):
        """Test DataFrame with MultiIndex rows."""
        data = {
            'Q1': [100, 150, 120, 180],
            'Q2': [110, 160, 130, 190],
            'Q3': [120, 170, 125, 195],
            'Q4': [130, 180, 135, 200]
        }
        
        # Create MultiIndex for rows
        row_index = pd.MultiIndex.from_tuples([
            ('North', 'ProductA'), ('North', 'ProductB'), 
            ('South', 'ProductA'), ('South', 'ProductB')
        ], names=['Region', 'Product'])
        
        df = pd.DataFrame(data, index=row_index)
        
        result = export_tabulator_payload(df, include_index=True)
        
        # Check that MultiIndex is flattened in data
        assert len(result['table']) == 4
        assert result['table'][0]['index'] == 'North / ProductA'
        assert result['table'][0]['Q1'] == 100
        assert result['table'][1]['index'] == 'North / ProductB'
        assert result['table'][1]['Q1'] == 150
        
        # Check that index column is included in column definitions
        assert len(result['columns']) == 5  # index + Q1 + Q2 + Q3 + Q4
        assert result['columns'][0]['title'] == 'index'
        assert result['columns'][0]['field'] == 'index'
        assert result['columns'][1]['title'] == 'Q1'
        assert result['columns'][2]['title'] == 'Q2'

    def test_dataframe_with_index_excluded(self):
        """Test DataFrame with custom index but include_index=False."""
        data = {
            'Product': ['Widget A', 'Widget B', 'Widget C'],
            'Price': [19.99, 24.99, 14.99]
        }
        df = pd.DataFrame(data, index=['A1', 'B2', 'C3'])
        
        result = export_tabulator_payload(df, include_index=False)
        
        # Check that index is not included in data
        assert len(result['table']) == 3
        assert 'index' not in result['table'][0]
        assert result['table'][0]['Product'] == 'Widget A'
        
        # Check that index column is not included in column definitions
        assert len(result['columns']) == 2  # Only Product + Price
        assert result['columns'][0]['title'] == 'Product'
        assert result['columns'][1]['title'] == 'Price'

    def test_multiindex_columns(self):
        """Test DataFrame with MultiIndex columns."""
        data = {
            'Q1': [100, 150, 120, 180],
            'Q2': [110, 160, 130, 190],
            'Q3': [120, 170, 125, 195],
            'Q4': [130, 180, 135, 200]
        }
        
        df = pd.DataFrame(data)
        
        # Create MultiIndex for columns
        col_index = pd.MultiIndex.from_tuples([
            ('Sales', 'Q1'), ('Sales', 'Q2'), 
            ('Revenue', 'Q3'), ('Revenue', 'Q4')
        ], names=['Category', 'Quarter'])
        
        df.columns = col_index
        
        result = export_tabulator_payload(df)
        
        # Check that MultiIndex columns are flattened
        assert len(result['columns']) == 2  # Two parent columns
        assert result['columns'][0]['title'] == 'Sales'
        assert len(result['columns'][0]['columns']) == 2
        assert result['columns'][0]['columns'][0]['title'] == 'Q1'
        assert result['columns'][0]['columns'][0]['field'] == 'Sales_Q1'
        
        # Check table data has flattened column names
        assert 'Sales_Q1' in result['table'][0]
        assert 'Sales_Q2' in result['table'][0]
        assert 'Revenue_Q3' in result['table'][0]
        assert 'Revenue_Q4' in result['table'][0]

    def test_multiindex_columns_always_flattened(self):
        """Test DataFrame with MultiIndex columns - always flattened for HTML compatibility."""
        data = {
            'Q1': [100, 150, 120, 180],
            'Q2': [110, 160, 130, 190],
            'Q3': [120, 170, 125, 195],
            'Q4': [130, 180, 135, 200]
        }
        
        df = pd.DataFrame(data)
        
        # Create MultiIndex for columns
        col_index = pd.MultiIndex.from_tuples([
            ('Sales', 'Q1'), ('Sales', 'Q2'), 
            ('Revenue', 'Q3'), ('Revenue', 'Q4')
        ], names=['Category', 'Quarter'])
        
        df.columns = col_index
        
        result = export_tabulator_payload(df)
        
        # Check that MultiIndex columns are always flattened
        assert len(result['columns']) == 2  # Two parent columns
        assert result['columns'][0]['title'] == 'Sales'
        assert len(result['columns'][0]['columns']) == 2
        assert result['columns'][0]['columns'][0]['title'] == 'Q1'
        assert result['columns'][0]['columns'][0]['field'] == 'Sales_Q1'

    def test_both_multiindex(self):
        """Test DataFrame with both row and column MultiIndex."""
        # Create complex data
        complex_data = np.random.randint(10, 100, (4, 4))
        row_idx = pd.MultiIndex.from_product([['Group1', 'Group2'], ['ItemA', 'ItemB']], 
                                           names=['Group', 'Item'])
        col_idx = pd.MultiIndex.from_product([['Metric1', 'Metric2'], ['Val1', 'Val2']], 
                                           names=['Metric', 'Value'])
        
        df = pd.DataFrame(complex_data, index=row_idx, columns=col_idx)
        
        result = export_tabulator_payload(df, include_index=True)
        
        # Check structure
        assert len(result['table']) == 4
        assert len(result['columns']) == 3  # index + two parent columns (Metric1, Metric2)
        
        # Check that index column is first
        assert result['columns'][0]['title'] == 'index'
        assert result['columns'][0]['field'] == 'index'
        
        # Check that MultiIndex columns are properly structured
        assert result['columns'][1]['title'] == 'Metric1'
        assert len(result['columns'][1]['columns']) == 2
        assert result['columns'][2]['title'] == 'Metric2'
        assert len(result['columns'][2]['columns']) == 2
        
        # Check flattened row index
        assert result['table'][0]['index'] == 'Group1 / ItemA'
        assert result['table'][1]['index'] == 'Group1 / ItemB'
        assert result['table'][2]['index'] == 'Group2 / ItemA'
        assert result['table'][3]['index'] == 'Group2 / ItemB'
        
        # Check flattened column names
        assert 'Metric1_Val1' in result['table'][0]
        assert 'Metric1_Val2' in result['table'][0]
        assert 'Metric2_Val1' in result['table'][0]
        assert 'Metric2_Val2' in result['table'][0]

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pd.DataFrame()
        
        result = export_tabulator_payload(df)
        
        assert 'table' in result
        assert 'columns' in result
        assert 'options' in result
        assert len(result['table']) == 0
        assert result['columns'] == []  # Empty DataFrame with include_columns=True returns empty list

    def test_empty_dataframe_no_columns(self):
        """Test empty DataFrame without including columns."""
        df = pd.DataFrame()
        
        result = export_tabulator_payload(df, include_columns=False)
        
        assert 'table' in result
        assert 'columns' in result
        assert 'options' in result
        assert len(result['table']) == 0
        assert result['columns'] is None  # When include_columns=False, columns should be None

    def test_single_row_dataframe(self):
        """Test DataFrame with single row."""
        data = {'Name': ['Alice'], 'Age': [25]}
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df)
        
        assert len(result['table']) == 1
        assert result['table'][0]['Name'] == 'Alice'
        assert result['table'][0]['Age'] == 25

    def test_single_column_dataframe(self):
        """Test DataFrame with single column."""
        data = {'Name': ['Alice', 'Bob', 'Charlie']}
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df)
        
        assert len(result['table']) == 3
        assert len(result['columns']) == 1
        assert result['columns'][0]['title'] == 'Name'
        assert result['columns'][0]['field'] == 'Name'

    def test_dataframe_with_numeric_index(self):
        """Test DataFrame with numeric index."""
        data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
        df = pd.DataFrame(data, index=[100, 200])
        
        result = export_tabulator_payload(df, include_index=True)
        
        assert len(result['table']) == 2
        assert result['table'][0]['index'] == 100
        assert result['table'][1]['index'] == 200

    def test_dataframe_with_range_index(self):
        """Test DataFrame with default RangeIndex."""
        data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
        df = pd.DataFrame(data)  # Default RangeIndex
        
        result = export_tabulator_payload(df, include_index=True)
        
        # RangeIndex should not be included
        assert len(result['table']) == 2
        assert 'index' not in result['table'][0]
        assert result['table'][0]['Name'] == 'Alice'

    def test_custom_layout(self):
        """Test DataFrame with custom layout."""
        data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df, layout='fitData')
        
        assert result['options']['layout'] == 'fitData'
        assert result['options']['responsiveLayout'] == 'collapse'

    def test_without_columns(self):
        """Test DataFrame without including columns."""
        data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df, include_columns=False)
        
        assert result['columns'] is None
        assert len(result['table']) == 2

    def test_mixed_data_types(self):
        """Test DataFrame with mixed data types."""
        data = {
            'String': ['Alice', 'Bob'],
            'Integer': [25, 30],
            'Float': [25.5, 30.7],
            'Boolean': [True, False],
            'None': [None, 'Value']
        }
        df = pd.DataFrame(data)
        
        result = export_tabulator_payload(df)
        
        assert len(result['table']) == 2
        assert result['table'][0]['String'] == 'Alice'
        assert result['table'][0]['Integer'] == 25
        assert result['table'][0]['Float'] == 25.5
        assert result['table'][0]['Boolean'] is True
        assert result['table'][0]['None'] is None

    def test_multiindex_with_nan_values(self):
        """Test MultiIndex with NaN values."""
        data = {
            'Q1': [100, 150, 120, 180],
            'Q2': [110, 160, 130, 190]
        }
        
        # Create MultiIndex with NaN values
        row_index = pd.MultiIndex.from_tuples([
            ('North', 'ProductA'), ('North', np.nan), 
            (np.nan, 'ProductA'), ('South', 'ProductB')
        ], names=['Region', 'Product'])
        
        df = pd.DataFrame(data, index=row_index)
        
        result = export_tabulator_payload(df, include_index=True)
        
        # Check that NaN values are handled properly
        assert len(result['table']) == 4
        assert result['table'][0]['index'] == 'North / ProductA'
        assert 'nan' in result['table'][1]['index']  # NaN becomes 'nan' in string conversion
        assert 'nan' in result['table'][2]['index']
        assert result['table'][3]['index'] == 'South / ProductB'
