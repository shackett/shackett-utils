import json
from typing import Optional, List, Union, Dict

import pandas as pd
from IPython.display import display, HTML


def display_tabulator(
    df: pd.DataFrame,
    layout: str = "fitColumns",
    include_index: bool = True,
    include_columns: bool = True,
    caption: Optional[str] = None,
    width: Optional[str] = None,
    wrap_columns: Optional[Union[List[str], str, bool]] = None,
    column_widths: Optional[Union[Dict[str, Union[int, str]], List[Union[int, str]]]] = None,
) -> None:
    """
    Display a DataFrame as an interactive Tabulator table with optional text wrapping.

    Parameters
    ---------- 
    df : pd.DataFrame
        The input data.
    layout : str, default="fitColumns"
        Layout strategy for Tabulator. Options are:
        - fitData: resize the tables columns to fit the data held in each column, unless you specify a width or minWidth in the column constructor
        - fitDataFill: functions in the same way as the fitData mode, but ensures that rows are always at least the full width of the table.
        - fitDataStretch: functions in the same way as the fitDataFill mode, but instead of stretching the empty row to fill the table it stretches the last visible column.
        - fitDataTable: will set the column widths in the same way as the fitData mode, but it will also then resize the width of the table to match the total width of the columns
        - fitColumns: resize columns so they fit perfectly in the available table width.
    include_index : bool, default=True
        Whether to include the index in the output.
    include_columns : bool, default=True
        Whether to include column headers.
    caption : Optional[str], default=None
        Optional caption for the table.
    width : Optional[str], default=None
        CSS width for the table container.
    wrap_columns : Optional[Union[List[str], str, bool]], default=None
        Columns to enable text wrapping for:
        - True: Enable wrapping for all columns
        - False/None: No text wrapping
        - str: Single column name to wrap
        - List[str]: List of column names to wrap
    column_widths : Optional[Union[Dict[str, Union[int, str]], List[Union[int, str]]]], default=None
        Column width specifications:
        - Dict: Map column names to widths (e.g., {'col1': 100, 'col2': '20%'})
        - List: Widths in column order (e.g., [100, 200, '30%'])
        - Widths can be integers (pixels) or strings (percentages/CSS units)
        
    Examples
    --------
    >>> # Regular table (no wrapping)
    >>> display_tabulator(df)
    
    >>> # Table with text wrapping on all columns
    >>> display_tabulator(df, wrap_columns=True)
    
    >>> # Table with wrapping on specific columns
    >>> display_tabulator(df, wrap_columns=['description', 'comments'])
    
    >>> # Table with wrapping on a single column
    >>> display_tabulator(df, wrap_columns='long_text_column')
    """
    payload = export_tabulator_payload(
        df,
        layout=layout,
        include_index=include_index,
        include_columns=include_columns,
        wrap_columns=wrap_columns,
        column_widths=column_widths,
    )

    if caption:
        display(HTML(f"""
        <figcaption style='font-weight:bold; margin-bottom:0.5em'>
            {caption}
        </figcaption>
        """))

    # Control width through CSS on the container div
    container_style = f"width: {width}; display: inline-block;" if width else ""

    display(HTML(f"""
    <div class="data-table" style="{container_style}"
        data-table='{json.dumps(payload["table"])}'
        data-columns='{json.dumps(payload["columns"] or [])}'
        data-options='{json.dumps(payload["options"])}'>
    </div>
    """))

    return None


def export_tabulator_payload(
    df: pd.DataFrame,
    layout: str = 'fitColumns',
    include_index: bool = True,
    include_columns: bool = True,
    wrap_columns: Optional[Union[List[str], str, bool]] = None,
    column_widths: Optional[Union[Dict[str, Union[int, str]], List[Union[int, str]]]] = None,
) -> dict:
    """
    Convert a DataFrame into a Tabulator-compatible JSON payload.

    Parameters
    ----------
    df : pd.DataFrame
        The input data.
    Layout strategy for Tabulator. Options are:
        - fitData: resize the tables columns to fit the data held in each column, unless you specify a width or minWidth in the column constructor
        - fitDataFill: functions in the same way as the fitData mode, but ensures that rows are always at least the full width of the table.
        - fitDataStretch: functions in the same way as the fitDataFill mode, but instead of stretching the empty row to fill the table it stretches the last visible column.
        - fitDataTable: will set the column widths in the same way as the fitData mode, but it will also then resize the width of the table to match the total width of the columns
        - fitColumns: resize columns so they fit perfectly in the available table width.
    include_index : bool, default=True
        Whether to include the index in the output.
    include_columns : bool, default=True
        Whether to include column headers.
    wrap_columns : Optional[Union[List[str], str, bool]], default=None
        Columns to enable text wrapping for:
        - True: Enable wrapping for all columns
        - False/None: No text wrapping
        - str: Single column name to wrap
        - List[str]: List of column names to wrap
    column_widths : Optional[Union[Dict[str, Union[int, str]], List[Union[int, str]]]], default=None
        Column width specifications:
        - Dict: Map column names to widths (e.g., {'col1': 100, 'col2': '20%'})
        - List: Widths in column order (e.g., [100, 200, '30%'])
        - Widths can be integers (pixels) or strings (percentages/CSS units)
    
    Returns
    -------
    dict
        A dictionary with keys: "table", "columns", "options".
        - table: a list of dictionaries, each representing a row in the table
        - columns: a list of dictionaries, each representing a column in the table. These define the columns in the table and how they should be displayed.
        - options: a dictionary Tabulator table options such as the layout of the table.
    """
    table_data = df.copy()
    column_defs = None

    # --- Handle MultiIndex Columns ---
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        table_data.columns = flat_cols

        grouped = {}
        for (parent, child), flat_name in zip(df.columns, flat_cols):
            grouped.setdefault(parent, []).append({
                "title": str(child),
                "field": flat_name
            })
        column_defs = [{"title": str(parent), "columns": children} for parent, children in grouped.items()]
    elif include_columns:
        column_defs = [{"title": str(col), "field": str(col)} for col in table_data.columns]

    # --- Handle Index ---
    index_columns = []
    if include_index:
        if isinstance(table_data.index, pd.MultiIndex):
            table_data.index = table_data.index.map(lambda x: " / ".join(map(str, x)))
        if table_data.index.name or not table_data.index.equals(pd.RangeIndex(len(table_data))):
            if table_data.index.name:
                index_columns = [table_data.index.name]
            else:
                index_columns = ['index']
            table_data = table_data.reset_index()

    # --- Update column definitions to include index columns ---
    if include_columns and index_columns:
        if column_defs is None:
            column_defs = []
        
        for col_name in index_columns:
            new_col_def = {"title": str(col_name), "field": str(col_name)}
            # Apply wrapping to index columns if requested
            if _should_wrap_column(col_name, wrap_columns, [col_name]):
                new_col_def['formatter'] = 'textarea'
                new_col_def['variableHeight'] = True
            column_defs.insert(0, new_col_def)

    table_records = table_data.to_dict(orient="records")

    # --- Handle text wrapping ---
    if wrap_columns is not None and column_defs is not None:
        _apply_text_wrapping(column_defs, wrap_columns, table_data.columns.tolist())

    # --- Handle column widths ---
    if column_widths is not None and column_defs is not None:
        _apply_column_widths(column_defs, column_widths, table_data.columns.tolist())

    # --- Options ---
    options = {
        "layout": layout,
        "responsiveLayout": "collapse"
    }

    return {
        "table": table_records,
        "columns": column_defs,
        "options": options
    }

# private utils

def _apply_column_widths(column_defs: List[dict], column_widths: Union[Dict[str, Union[int, str]], List[Union[int, str]]], available_columns: List[str]) -> None:
    """Apply column width configuration to column definitions."""
    if isinstance(column_widths, dict):
        # Validate that all specified columns exist
        missing_columns = set(column_widths.keys()) - set(available_columns)
        if missing_columns:
            raise ValueError(
                f"Column width specified for non-existent columns: {sorted(missing_columns)}. "
                f"Available columns are: {sorted(available_columns)}"
            )
        
        # Dictionary mapping column names to widths
        for col_def in column_defs:
            if isinstance(col_def, dict):
                if 'field' in col_def and col_def['field'] in column_widths:
                    col_def['width'] = column_widths[col_def['field']]
                elif 'columns' in col_def:
                    # Handle grouped columns (MultiIndex)
                    for sub_col in col_def['columns']:
                        if sub_col['field'] in column_widths:
                            sub_col['width'] = column_widths[sub_col['field']]
    elif isinstance(column_widths, list):
        # Validate list length
        if len(column_widths) > len(available_columns):
            raise ValueError(
                f"Too many column widths specified: got {len(column_widths)} widths "
                f"but only {len(available_columns)} columns available. "
                f"Available columns are: {sorted(available_columns)}"
            )
        
        # List of widths in column order
        col_index = 0
        for col_def in column_defs:
            if isinstance(col_def, dict):
                if 'field' in col_def:
                    # Regular column
                    if col_index < len(column_widths):
                        col_def['width'] = column_widths[col_index]
                        col_index += 1
                elif 'columns' in col_def:
                    # Handle grouped columns (MultiIndex)
                    for sub_col in col_def['columns']:
                        if col_index < len(column_widths):
                            sub_col['width'] = column_widths[col_index]
                            col_index += 1


def _apply_text_wrapping(column_defs: List[dict], wrap_columns: Union[List[str], str, bool], available_columns: List[str]) -> None:
    """Apply text wrapping configuration to column definitions."""
    # Validate column names if using dict-like specification
    if isinstance(wrap_columns, str):
        if wrap_columns not in available_columns:
            raise ValueError(
                f"Text wrapping specified for non-existent column: '{wrap_columns}'. "
                f"Available columns are: {sorted(available_columns)}"
            )
    elif isinstance(wrap_columns, list):
        missing_columns = set(wrap_columns) - set(available_columns)
        if missing_columns:
            raise ValueError(
                f"Text wrapping specified for non-existent columns: {sorted(missing_columns)}. "
                f"Available columns are: {sorted(available_columns)}"
            )
    
    for col_def in column_defs:
        if isinstance(col_def, dict):
            if 'field' in col_def:
                # Regular column
                if _should_wrap_column(col_def['field'], wrap_columns, available_columns):
                    col_def['formatter'] = 'textarea'
                    col_def['variableHeight'] = True
            elif 'columns' in col_def:
                # Handle grouped columns (MultiIndex)
                for sub_col in col_def['columns']:
                    if _should_wrap_column(sub_col['field'], wrap_columns, available_columns):
                        sub_col['formatter'] = 'textarea'
                        sub_col['variableHeight'] = True


def _should_wrap_column(column_name: str, wrap_columns: Union[List[str], str, bool], available_columns: List[str]) -> bool:
    """Check if a column should have text wrapping enabled."""
    if wrap_columns is True:
        return True
    elif wrap_columns is False or wrap_columns is None:
        return False
    elif isinstance(wrap_columns, str):
        return column_name == wrap_columns and column_name in available_columns
    elif isinstance(wrap_columns, list):
        return column_name in wrap_columns and column_name in available_columns
    else:
        return False
