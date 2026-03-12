"""
Extract, transform, and load the Forny 2023 dataset.

This module provides ETL pipelines for the multi-omics dataset from Forny et al. 2023,
combining transcriptomics, proteomics, and phenotype data into a MuData object.

Public Functions
----------------
forny_etl_and_normalize
    Load supplemental data, build MuData, and normalize in-place.
"""

import os
from typing import Any, Dict

import pandas as pd
from anndata import AnnData
from mudata import MuData
from sklearn.impute import KNNImputer

import shackett_utils.genomics.adata_processing as processing
from shackett_utils.applications import forny_imputation
from shackett_utils.applications.constants import (
    FORNY_PROTEOMICS_FILE_NAMES_URL,
    FORNY_SUPPLEMENTAL_DATA_FILES,
)
from shackett_utils.statistics import transform


def forny_etl_and_normalize(supplemental_data_dir: str) -> MuData:
    """
    Load the Forny 2023 dataset and return a normalized MuData object.

    Parameters
    ----------
    supplemental_data_dir : str
        Path to the directory containing the supplemental Excel files from the
        Forny paper (MOESM3 and MOESM4).

    Returns
    -------
    MuData
        MuData object with transcriptomics and proteomics modalities, phenotypes
        in obs, and log2-centered normalized layers.
    """
    supplemental_data_paths = _get_supplemental_data_paths(supplemental_data_dir)
    supplemental_data = _load_supplemental_data(supplemental_data_paths)
    phenotypes_df = _load_phenotypes(supplemental_data["phenotypes"])
    mdata = _load_forny_mudata(supplemental_data, phenotypes_df)
    _normalize_forny_mudata(mdata)

    return mdata


def _get_proteomics_run_order(
    proteomics_file_names_url: str = FORNY_PROTEOMICS_FILE_NAMES_URL,
) -> pd.Series:
    """
    Parse proteomics run order from PRIDE repository file names.

    For samples with multiple runs, returns the max run order (typically the
    highest-quality rerun).

    Parameters
    ----------
    proteomics_file_names_url : str, optional
        URL or path to the tab-separated sample annotation file.

    Returns
    -------
    pd.Series
        Series mapping patient_id to proteomics_runorder (max per patient).
    """
    proteomics_file_names = pd.read_csv(proteomics_file_names_url, sep="\t")
    proteomics_file_names["proteomics_runorder"] = [
        _runorder_from_filename(fn) for fn in proteomics_file_names["File Name"]
    ]
    proteomics_file_names["patient_id"] = proteomics_file_names[
        "Run Label"
    ].str.replace("_", "")
    # some samples have multiple files in the PRIDE proteomics repository but they each sample is just a single observation in the
    # final dataset. Here we'll take the max run order for each patient. These likely reflect reruns due to technical failures
    # so the max run order sample is usually the one with the best technical quality.
    proteomics_file_names = proteomics_file_names.groupby("patient_id")[
        "proteomics_runorder"
    ].max()

    return proteomics_file_names


def _get_supplemental_data_paths(
    supplemental_data_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve paths to supplemental data files and validate they exist.

    Parameters
    ----------
    supplemental_data_dir : str
        Base directory containing the supplemental Excel files.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dict mapping data type to {"path": str, "sheet": str | int}.

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    supplemental_data_paths = {
        x: {"path": os.path.join(supplemental_data_dir, y["file"]), "sheet": y["sheet"]}
        for x, y in FORNY_SUPPLEMENTAL_DATA_FILES.items()
    }

    missing_files = [
        x for x in supplemental_data_paths.values() if not os.path.isfile(x["path"])
    ]
    if missing_files:
        raise FileNotFoundError(
            f"The following files were not found: {missing_files}. Please download the files from the Forny paper and place them in the {supplemental_data_dir} directory."
        )

    return supplemental_data_paths


def _load_forny_mudata(
    supplemental_data: Dict[str, pd.DataFrame],
    phenotypes_df: pd.DataFrame,
) -> MuData:
    """
    Build MuData from transcriptomics, proteomics, and phenotype DataFrames.

    Parameters
    ----------
    supplemental_data : Dict[str, pd.DataFrame]
        Dict with "transcriptomics", "proteomics", and "phenotypes" DataFrames.
    phenotypes_df : pd.DataFrame
        Processed phenotypes (imputed, transformed) indexed by patient_id.

    Returns
    -------
    MuData
        MuData with transcriptomics and proteomics modalities.
    """
    transcr_adata = AnnData(
        X=supplemental_data["transcriptomics"].T,
        # some samples are missing
        obs=phenotypes_df.loc[
            phenotypes_df.index.isin(supplemental_data["transcriptomics"].columns)
        ],
    )

    protein_metadata_vars = supplemental_data["proteomics"].columns[
        supplemental_data["proteomics"].columns.str.startswith("PG")
    ]

    proteomics_adata = AnnData(
        # drop protein metadata vars and transpose
        X=supplemental_data["proteomics"].drop(protein_metadata_vars, axis=1).T,
        # some samples are missing
        obs=phenotypes_df.loc[
            phenotypes_df.index.isin(supplemental_data["proteomics"].columns)
        ],
        var=supplemental_data["proteomics"][protein_metadata_vars],
    )

    return MuData({"transcriptomics": transcr_adata, "proteomics": proteomics_adata})


def _load_phenotypes(
    phenotypes_df: pd.DataFrame,
    max_missing: int = 180,
) -> pd.DataFrame:
    """
    Process phenotypes: select continuous measures, impute, transform, and merge.

    Parameters
    ----------
    phenotypes_df : pd.DataFrame
        Raw phenotypes from the supplemental Excel file (indexed by patient_id).
    max_missing : int, optional
        Maximum number of missing values allowed for a continuous measure to be
        included. Default is 180.

    Returns
    -------
    pd.DataFrame
        Processed phenotypes with binary variables, imputed/transformed continuous
        variables, and proteomics_runorder.
    """
    proteomics_file_names = _get_proteomics_run_order()

    # select continuous phenotypes without too many missing values
    continuous_df = forny_imputation._select_continuous_measures(
        phenotypes_df, max_missing=max_missing
    )

    # identify transformations which improve variable normality
    normalizing_transforms = {}
    for col in continuous_df.columns:
        normalizing_transforms[col] = transform.best_normalizing_transform(
            continuous_df[col]
        )["best"]

    func_transform_dict = {
        col: transform.transform_func_map[trans]
        for col, trans in normalizing_transforms.items()
    }

    # Apply transformation
    transformed_df = forny_imputation.transform_columns(
        continuous_df, func_transform_dict
    )

    # Create the imputer
    imputer = KNNImputer(n_neighbors=5, weights="uniform")

    # Fit and transform the data
    imputed_array = imputer.fit_transform(transformed_df)
    imputed_df = pd.DataFrame(
        imputed_array, columns=transformed_df.columns, index=transformed_df.index
    )

    binary_phenotypes = phenotypes_df[
        [
            col
            for col in phenotypes_df.columns
            if phenotypes_df[col].dropna().nunique() == 2
        ]
    ]

    # create a table of phenotypes
    return (
        pd.concat([binary_phenotypes, imputed_df], axis=1)
        # add proteomics run order since this is a batch effect in the proteomics data
        .join(proteomics_file_names, how="left")
    )


def _load_supplemental_data(
    supplemental_data_paths: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """
    Load and format supplemental Excel files into DataFrames.

    Parameters
    ----------
    supplemental_data_paths : Dict[str, Dict[str, Any]]
        Dict mapping data type to {"path": str, "sheet": str | int}.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dict with "transcriptomics", "proteomics", and "phenotypes" DataFrames,
        each with appropriate index set.
    """
    supplemental_data = {
        x: pd.read_excel(y["path"], sheet_name=y["sheet"])
        for x, y in supplemental_data_paths.items()
    }

    # formatting
    supplemental_data["transcriptomics"] = (
        supplemental_data["transcriptomics"]
        .rename({"Unnamed: 0": "ensembl_gene"}, axis=1)
        .set_index("ensembl_gene")
    )
    supplemental_data["proteomics"] = (
        supplemental_data["proteomics"]
        .rename({"PG.ProteinAccessions": "uniprot"}, axis=1)
        .set_index("uniprot")
    )
    supplemental_data["phenotypes"] = (
        supplemental_data["phenotypes"]
        .rename({"Unnamed: 0": "patient_id"}, axis=1)
        .set_index("patient_id")
    )

    return supplemental_data


def _normalize_forny_mudata(
    mdata: MuData,
    read_cutoff: int = 400,
) -> None:
    """
    Filter and normalize the Forny MuData in-place.

    Applies count filtering, log2 transform, and row/column centering as per
    the Forny paper methods.

    Parameters
    ----------
    mdata : MuData
        MuData to normalize (modified in-place).
    read_cutoff : int, optional
        Minimum total counts for transcriptomics features to retain.
        Default is 400.
    """
    # filter to drop features with low counts
    processing.filter_features_by_counts(
        mdata["transcriptomics"], min_counts=read_cutoff
    )

    # add a pseudocount before logging
    processing.log2_transform(mdata["transcriptomics"], pseudocount=1)
    # proteomics has a minimum value of 1 so no pseudocounts are needed before logging
    processing.log2_transform(mdata["proteomics"], pseudocount=0)

    # row and column center as per Forny paper
    processing.center_rows_and_columns_mudata(
        mdata, layer="log2", new_layer_name="log2_centered"
    )

    return None


def _runorder_from_filename(fn: str) -> int:
    """
    Extract run order integer from a filename.

    Expects filename format ending in "_<runorder>" (e.g. "sample_2" -> 2).

    Parameters
    ----------
    fn : str
        Filename containing run order as the last underscore-separated segment.

    Returns
    -------
    int
        The run order value.

    Examples
    --------
    >>> _runorder_from_filename("asdf_2")
    2
    """
    return int(fn.split("_")[-1])
