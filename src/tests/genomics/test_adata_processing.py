import numpy as np
import pandas as pd
import pytest
import anndata as ad

from shackett_utils.genomics.adata_processing import (
    CENTERING_DEFS,
    center_rows_and_columns,
)


@pytest.fixture
def small_adata():
    np.random.seed(0)
    n_cells, n_genes = 20, 5
    X = np.random.normal(loc=3.0, scale=1.0, size=(n_cells, n_genes))
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )


def test_copy_layer_with_x_does_not_modify_x(small_adata):
    """copy_layer=True with layer=None must leave .X untouched and write a new layer."""
    original_x = small_adata.X.copy()

    center_rows_and_columns(small_adata, layer=None, copy_layer=True)

    # .X must be unchanged
    np.testing.assert_array_equal(small_adata.X, original_x)

    # Default new layer name should exist and contain centered data
    new_layer = CENTERING_DEFS.DEFAULT_NAME_NOLAYER
    assert new_layer in small_adata.layers
    centered = small_adata.layers[new_layer]
    assert centered.shape == original_x.shape
    assert not np.allclose(centered, original_x)
    assert abs(centered.mean(axis=0)).max() < 1e-6
    assert abs(centered.mean(axis=1)).max() < 1e-6


def test_copy_layer_with_x_respects_custom_name(small_adata):
    original_x = small_adata.X.copy()
    center_rows_and_columns(
        small_adata, layer=None, copy_layer=True, new_layer_name="my_centered"
    )

    np.testing.assert_array_equal(small_adata.X, original_x)
    assert "my_centered" in small_adata.layers
    assert CENTERING_DEFS.DEFAULT_NAME_NOLAYER not in small_adata.layers
