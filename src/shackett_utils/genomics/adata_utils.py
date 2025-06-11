"""
Utility functions for working with AnnData objects
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from typing import Tuple, Union, List


def get_adata_features_and_data(
    adata: AnnData, layer: Union[str, None] = None
) -> Tuple[Union[List[str], pd.Index], np.ndarray]:
    """
    Extract feature names and data matrix from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to extract data from
    layer : str or None
        The layer to extract data from. Can be:
        - None: use adata.X
        - A key in adata.layers
        - A key in adata.obsm

    Returns
    -------
    feature_names : List[str] or pd.Index
        Names of the features in the data matrix
    data_matrix : np.ndarray
        The extracted data as a dense numpy array

    Raises
    ------
    ValueError
        If the specified layer is not found or data type is unsupported
    """
    # Get feature names based on data source
    if layer is None:
        feature_names = adata.var_names
    elif layer in adata.layers:
        feature_names = adata.var_names
    elif layer in adata.obsm:
        # For obsm matrices, we need to generate feature names
        if hasattr(adata.obsm[layer], "columns"):
            # If it's a DataFrame with column names
            feature_names = adata.obsm[layer].columns
        else:
            # Otherwise, generate numeric feature names
            n_features = adata.obsm[layer].shape[1]
            feature_names = [f"{layer}_feature_{i}" for i in range(n_features)]
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers or adata.obsm")

    # Get data to regress
    if layer is None:
        if isinstance(adata.X, np.ndarray):
            data_matrix = adata.X
        else:
            data_matrix = adata.X.toarray()
    elif layer in adata.layers:
        if isinstance(adata.layers[layer], np.ndarray):
            data_matrix = adata.layers[layer]
        else:
            data_matrix = adata.layers[layer].toarray()
    elif layer in adata.obsm:
        if isinstance(adata.obsm[layer], pd.DataFrame):
            data_matrix = adata.obsm[layer].values
        elif isinstance(adata.obsm[layer], np.ndarray):
            data_matrix = adata.obsm[layer]
        elif sp.issparse(adata.obsm[layer]):
            data_matrix = adata.obsm[layer].toarray()
        else:
            raise ValueError(f"Unsupported data type for obsm['{layer}']")

    return feature_names, data_matrix
