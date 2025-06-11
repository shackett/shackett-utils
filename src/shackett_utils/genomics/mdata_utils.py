"""
MuData Utilities

This module provides utility functions for working with MuData objects.
"""

from typing import List, Optional

import anndata as ad
import mudata as md


def create_minimal_mudata(
    original_mdata: md.MuData,
    include_layers: Optional[List[str]] = None,
    include_obsm: bool = True,
    include_varm: bool = False,
) -> md.MuData:
    """
    Create a minimal copy of a MuData object with only essential components
    to avoid serialization issues.

    Parameters
    ----------
    original_mdata : md.MuData
        Original MuData object to copy
    include_layers : Optional[List[str]]
        List of layer names to include in the copy, by default None
    include_obsm : bool
        Whether to include obsm matrices, by default True
    include_varm : bool
        Whether to include varm matrices, by default False

    Returns
    -------
    md.MuData
        Minimal MuData object

    Examples
    --------
    >>> # Create a minimal copy with just the log-normalized layer
    >>> minimal_mdata = create_minimal_mudata(
    ...     mdata,
    ...     include_layers=["log_normalized"],
    ...     include_obsm=True
    ... )
    """
    # Create dictionary to hold modalities
    modalities = {}

    # Copy over each modality with minimal data
    for mod_name, mod in original_mdata.mod.items():
        # Create basic AnnData with just X, obs, and var
        new_mod = ad.AnnData(X=mod.X.copy(), obs=mod.obs.copy(), var=mod.var.copy())

        # Add selected layers if present
        if include_layers and hasattr(mod, "layers"):
            for layer_name in include_layers:
                if layer_name in mod.layers:
                    new_mod.layers[layer_name] = mod.layers[layer_name].copy()

        # Add obsm matrices if requested
        if include_obsm and hasattr(mod, "obsm"):
            for obsm_key in mod.obsm.keys():
                new_mod.obsm[obsm_key] = mod.obsm[obsm_key].copy()

        # Add varm matrices if requested
        if include_varm and hasattr(mod, "varm"):
            for varm_key in mod.varm.keys():
                new_mod.varm[varm_key] = mod.varm[varm_key].copy()

        # Add to modalities dictionary
        modalities[mod_name] = new_mod

    # Create MuData object with dictionary of modalities
    minimal_mdata = md.MuData(modalities)

    # Copy global obsm if present and requested
    if include_obsm and hasattr(original_mdata, "obsm"):
        for obsm_key in original_mdata.obsm.keys():
            minimal_mdata.obsm[obsm_key] = original_mdata.obsm[obsm_key].copy()

    return minimal_mdata
