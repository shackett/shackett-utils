import json
from typing import Optional

import pandas as pd
from IPython.display import display, HTML

def export_tabulator_payload(
    df: pd.DataFrame,
    layout: str = 'fitColumns',
    flatten_multiindex: bool = True,
    include_index: bool = True,
    include_columns: bool = True,
) -> dict:
    """
    Convert a DataFrame into a Tabulator-compatible JSON payload.

    Parameters
    ----------
    df : pd.DataFrame
        The input data.
    layout : str, default='fitColumns'
        Layout strategy for Tabulator.
    flatten_multiindex : bool, default=True
        Flatten MultiIndex columns into compound strings.
    include_index : bool, default=True
        Whether to include the index in the output.
    include_columns : bool = True
        Whether to include column headers.
    
    Returns
    -------
    dict
        A dictionary with keys: "table", "columns", "options".
    """
    table_data = df.copy()
    column_defs = None

    # --- Handle MultiIndex Columns ---
    if isinstance(df.columns, pd.MultiIndex) and flatten_multiindex:
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

    # --- Options ---
    options = {
        "layout": layout,
        "responsiveLayout": "collapse"
    }

    # --- Handle Index (including flattening MultiIndex index) ---
    index_columns = []
    if include_index:
        if isinstance(table_data.index, pd.MultiIndex):
            table_data.index = table_data.index.map(lambda x: " / ".join(map(str, x)))
        if table_data.index.name or not table_data.index.equals(pd.RangeIndex(len(table_data))):
            # Store index column names before resetting
            if table_data.index.name:
                index_columns = [table_data.index.name]
            else:
                index_columns = ['index']
            table_data = table_data.reset_index()

    # --- Update column definitions to include index columns ---
    if include_columns and index_columns:
        if column_defs is None:
            # If no column definitions exist yet, create them
            column_defs = []
        
        # Add index column definitions at the beginning
        for col_name in index_columns:
            column_defs.insert(0, {"title": str(col_name), "field": str(col_name)})

    table_records = table_data.to_dict(orient="records")

    return {
        "table": table_records,
        "columns": column_defs,
        "options": options
    }

def display_tabulator(
    df: pd.DataFrame,
    layout: str = "fitColumns",
    flatten_multiindex: bool = True,
    include_index: bool = True,
    include_columns: bool = True,
    caption: Optional[str] = None
) -> None:
    """
    Display a DataFrame as an interactive Tabulator table in Quarto.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to display.
    layout : str, default='fitColumns'
        Layout mode for Tabulator ('fitColumns', 'fitData', 'fitDataStretch', etc.).
    flatten_multiindex : bool, default=True
        If True, flattens MultiIndex columns using underscores.
    include_index : bool, default=True
        If True, includes the index column in the table.
    include_columns : bool, default=True
        If False, omits header columns (not typical for Tabulator).
    caption : str, optional
        Optional caption text to display above the table.
    """
    payload = export_tabulator_payload(
        df,
        layout=layout,
        flatten_multiindex=flatten_multiindex,
        include_index=include_index,
        include_columns=include_columns,
    )

    if caption:
        display(HTML(f"""
        <figcaption style='font-weight:bold; margin-bottom:0.5em'>
            {caption}
        </figcaption>
        """))

    display(HTML(f"""
    <div class="data-table"
        data-table='{json.dumps(payload["table"])}'
        data-columns='{json.dumps(payload["columns"] or [])}'
        data-options='{json.dumps(payload["options"])}'>
    </div>
    """))