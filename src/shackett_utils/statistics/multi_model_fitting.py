from joblib import Parallel, delayed
import logging
import re
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multitest

from .model_fitting import StatisticalModel, OLSModel, GAMModel, _validate_tidy_df
from .constants import REQUIRED_TIDY_VARS, STATISTICS_DEFS, MULTITEST_GROUPING_VARS, FDR_METHODS_DEFS, FDR_METHODS, TIDY_DEFS

logger = logging.getLogger(__name__)

def fit_feature_model_formula(
    y: np.ndarray,
    data: pd.DataFrame,
    feature_name: str,
    formula: str,
    model_class: str = 'ols',
    model_name: Optional[str] = None,
    allow_failures: bool = False,
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit a model for a single feature using formula interface.
    
    Parameters
    ----------
    y : np.ndarray
        Response vector (feature values to model)
    data : pd.DataFrame
        DataFrame containing predictor variables
    feature_name : str
        Name of the feature being modeled (for identification in output)
    formula : str
        Model formula (e.g. 'y ~ x1 + s(x2)')
    model_class : str
        Type of model to fit ('ols', 'gam', etc.)
    model_name : str, optional
        Name of the model for identification. Default is None.
    allow_failures : bool
        If True, handle errors gracefully and return empty DataFrame.
        If False, raise exceptions for debugging. Default is False.
    **model_kwargs
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model statistics for each coefficient
        
    Raises
    ------
    Exception
        If allow_failures is False and an error occurs during model fitting
    """
    # Filter out missing values
    valid_mask = ~np.isnan(y)
    n_valid = np.sum(valid_mask)
    if n_valid < len(y):
        logger.debug(f"Filtering {len(y) - n_valid} missing values for feature {feature_name}")
        y = y[valid_mask]
        data = data.iloc[valid_mask]
    
    # Check for zero variance
    if np.var(y) == 0:
        logger.warning(f"Skipping feature {feature_name} due to zero variance")
        return pd.DataFrame()
    
    try:
        # Create model instance
        if model_class.lower() == 'ols':
            model = OLSModel(feature_name=feature_name, model_name=model_name)
        elif model_class.lower() == 'gam':
            model = GAMModel(feature_name=feature_name, model_name=model_name)
        else:
            raise ValueError(f"Unsupported model class: {model_class}")
        
        # Create temporary DataFrame with response
        model_data = data.copy()
        # Ensure y is float64 before adding to DataFrame
        model_data['y'] = y
        
        # Fit the model
        model.fit(formula, data=model_data, **model_kwargs)
        
        return model.tidy()
            
    except Exception as e:
        if allow_failures:
            return _handle_model_error(e, feature_name)
        else:
            raise

def fit_feature_model_matrix(
    y: np.ndarray,
    X: np.ndarray,
    feature_name: str,
    term_names: List[str],
    model_name: Optional[str] = None,
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit an OLS model for a single feature using matrix interface.
    
    Parameters
    ----------
    y : np.ndarray
        Feature values to model (response variable)
    X : np.ndarray
        Model matrix (predictors), should already include intercept if needed
    feature_name : str
        Name of the feature being modeled
    term_names : List[str]
        Names for the coefficients/predictors
    model_name : str, optional
        Name of the model for identification. Default is None.
    **model_kwargs
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model statistics for each coefficient
    """
    # Filter out missing values
    valid_mask = ~np.isnan(y)
    n_valid = np.sum(valid_mask)
    if n_valid < len(y):
        logger.debug(f"Filtering {len(y) - n_valid} missing values for feature {feature_name}")
        y = y[valid_mask]
        X = X[valid_mask]
    
    # Check for zero variance
    if np.var(y) == 0:
        logger.warning(f"Skipping feature {feature_name} due to zero variance")
        return pd.DataFrame()
    
    try:
        # Validate input dimensions
        if len(y) != X.shape[0]:
            raise ValueError("Response vector and model matrix must have same number of samples")
        if len(term_names) != X.shape[1]:
            raise ValueError("Number of coefficient names must match number of model matrix columns")
        
        model = OLSModel(feature_name=feature_name, model_name=model_name)
        model.fit_xy(X, y, term_names=term_names, **model_kwargs)
        
        return model.tidy()
            
    except Exception as e:
        return _handle_model_error(e, feature_name)

def _validate_tidy_df(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame meets the requirements for a tidy results table.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
        
    Raises
    ------
    ValueError
        If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError("Tidy DataFrame must contain at least one row")
    
    missing_cols = [col for col in REQUIRED_TIDY_VARS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Tidy DataFrame missing required columns: {missing_cols}")

def _apply_fdr_correction(df: pd.DataFrame, mask: pd.Series, fdr_method: str) -> None:
    """
    Apply FDR correction to a subset of p-values and assign q-values in-place.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing p-values to correct
    mask : pd.Series
        Boolean mask indicating which rows to correct
    fdr_method : str
        FDR method to use
    """
    p_values = df.loc[mask, STATISTICS_DEFS.P_VALUE].values
    if len(p_values) > 0:  # Only perform correction if we have p-values
        df.loc[mask, STATISTICS_DEFS.Q_VALUE] = multitest.multipletests(
            p_values, 
            method=fdr_method
        )[1]
    else:
        df.loc[mask, STATISTICS_DEFS.Q_VALUE] = np.nan

def control_fdr(results_df: pd.DataFrame, fdr_method: str = FDR_METHODS_DEFS.FDR_BH) -> pd.DataFrame:
    """
    Apply FDR control to p-values, grouping by term and model_name (if present).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with model results. Must be a valid tidy DataFrame with required columns.
    fdr_method : str
        FDR method to use. Must be one of the methods in FDR_METHODS.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with FDR-corrected p-values
        
    Raises
    ------
    ValueError
        If results_df is empty or missing required columns
    """
    # Validate input DataFrame
    _validate_tidy_df(results_df)
    
    if fdr_method not in FDR_METHODS:
        raise ValueError(f"Invalid FDR method: {fdr_method}. Must be one of: {FDR_METHODS}")
    
    # Determine grouping columns based on available columns
    group_cols = [col for col in MULTITEST_GROUPING_VARS if col in results_df.columns]
    
    if len(group_cols) > 1:
        logger.info(f"Applying FDR control within groups: {', '.join(group_cols)}")
        # Apply correction within each group
        for _, group in results_df.groupby(group_cols):
            mask = results_df.index.isin(group.index)
            _apply_fdr_correction(results_df, mask, fdr_method)
    else:
        if group_cols:
            logger.info(f"Applying FDR control within {group_cols[0]} groups")
            # Use single column without list to avoid pandas warning
            for _, group in results_df.groupby(group_cols[0]):
                mask = results_df.index.isin(group.index)
                _apply_fdr_correction(results_df, mask, fdr_method)
        else:
            logger.info("Applying FDR control globally (no grouping columns found)")
            # Apply correction to all p-values at once
            _apply_fdr_correction(results_df, pd.Series(True, index=results_df.index), fdr_method)
    
    return results_df

def _validate_formula(formula: str, data: Optional[pd.DataFrame] = None) -> str:
    """
    Validate and standardize model formula to ensure 'y' is the dependent variable
    and all variables are numeric.
    
    Parameters
    ----------
    formula : str
        Model formula to validate. Must be either:
        - Full formula with response: 'y ~ x1 + x2'
        - Formula starting with ~: '~ x1 + x2'
    data : Optional[pd.DataFrame]
        DataFrame containing the variables referenced in the formula.
        If provided, checks that all variables are numeric.
        
    Returns
    -------
    str
        Standardized formula with 'y' as dependent variable
        
    Raises
    ------
    ValueError
        If formula is invalid or contains non-numeric variables
    """
    formula = formula.strip()
    
    # Validate formula has exactly one ~
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError("Formula must contain exactly one '~' character")
    
    # Handle formulas starting with ~
    lhs = parts[0].strip()
    rhs = parts[1].strip()
    if not lhs:
        formula = f"y ~ {rhs}"
    elif lhs != 'y':
        raise ValueError(f"Formula must use 'y' as dependent variable, got '{lhs}'")
    else:
        formula = f"y ~ {rhs}"
    
    # If data is provided, check that all variables are numeric
    if data is not None:
        # Remove s() from terms like s(x)
        rhs_clean = re.sub(r's\((.*?)\)', r'\1', rhs)
        # Split on operators and get unique variables
        variables = set()
        variables.update(var.strip() for var in re.split(r'[+\-*/]', rhs_clean) if var.strip())
        
        # Check each variable
        for var in variables:
            if var not in data.columns:
                raise ValueError(f"Variable '{var}' not found in data")
            if not np.issubdtype(data[var].dtype, np.number):
                raise ValueError(f"Variable '{var}' must be numeric, got dtype {data[var].dtype}")
    
    return formula

def _detect_model_class_from_formula(formula: str) -> str:
    """
    Detect whether a formula requires GAM or OLS based on presence of smooth terms.
    
    Parameters
    ----------
    formula : str
        Model formula (e.g. 'y ~ x1 + s(x2)')
        
    Returns
    -------
    str
        'gam' if formula contains smooth terms s(), 'ols' otherwise
    """
    # Check if formula contains any smooth terms s()
    # Match 's(' only when it's at the start of a term or after a '+' or '~'
    has_smooth_terms = bool(re.search(r'(^|\s|~|\+)\s*s\([^)]+\)', formula))
    model_class = 'gam' if has_smooth_terms else 'ols'
    logger.debug(f"Detected model_class='{model_class}' for formula='{formula}'")
    return model_class

def fit_parallel_models_formula(
    X_features: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    formula: str,
    model_class: Optional[str] = None,
    model_name: Optional[str] = None,
    n_jobs: int = 1,
    allow_failures: bool = False,
    fdr_control: bool = True,
    fdr_method: str = FDR_METHODS_DEFS.FDR_BH,
    batch_size: int = 100,
    progress_bar: bool = True,
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit models in parallel for multiple features using formula interface.
    
    Parameters
    ----------
    X_features : np.ndarray
        Feature matrix (n_samples, n_features)
    data : pd.DataFrame
        DataFrame containing predictor variables
    feature_names : List[str]
        Names of features being modeled
    formula : str
        Model formula (e.g. 'y ~ x1 + s(x2)')
    model_class : str, optional
        Type of model to fit ('ols', 'gam', etc.). If None, will be detected from formula.
    model_name : str, optional
        Name of the model for identification
    n_jobs : int
        Number of parallel jobs. Default is 1.
    allow_failures : bool
        If True, handle errors gracefully. Default is False.
    fdr_control : bool
        Whether to perform FDR control. Default is True.
    fdr_method : str
        FDR control method. Default is 'bh'.
    batch_size : int
        Number of features to process in each batch. Default is 100.
    progress_bar : bool
        Whether to display a progress bar. Default is True.
    **model_kwargs
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with model statistics for all features
        
    Raises
    ------
    ValueError
        If trying to fit OLS model with smooth terms in formula
    """
    # Input validation
    if len(feature_names) != X_features.shape[1]:
        raise ValueError(f"Length of feature_names ({len(feature_names)}) must match number of features in X_features ({X_features.shape[1]})")
    if X_features.shape[0] != len(data):
        raise ValueError("X_features and data must have same number of samples")
    
    inferred_model_class = _detect_model_class_from_formula(formula)
    # Auto-detect model class if not provided   
    if model_class is None:
        model_class = inferred_model_class
        logger.info(f"Auto-detected model_class='{model_class}' for formula='{formula}'")
    else:
        if model_class != inferred_model_class:
            if model_class == 'ols' and inferred_model_class == 'gam':
                raise ValueError(
                    f"Cannot fit OLS model with smooth terms. Formula '{formula}' contains smooth terms "
                    f"but model_class='ols' was specified. Either:\n"
                    f"1. Use model_class='gam' to fit with smooth terms\n"
                    f"2. Remove smooth terms from formula"
                )
            logger.warning(
                f"Model class mismatch: provided='{model_class}' inferred='{inferred_model_class}'. "
                f"Using provided model class='{model_class}'"
            )

    # Validate and standardize formula
    formula = _validate_formula(formula)

    # Verbose setting for progress bar
    verbose = 10 if progress_bar else 0
    
    # Run parallel processing
    logger.info(f"Starting parallel model fitting with {n_jobs} cores...")
    results_nested = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)(
        delayed(fit_feature_model_formula)(
            X_features[:, i],
            data,
            feature_names[i],
            formula=formula,
            model_class=model_class,
            model_name=model_name,
            allow_failures=allow_failures,
            **model_kwargs
        )
        for i in range(X_features.shape[1])
    )
    
    # Filter out empty DataFrames and combine results
    results_nested = [df for df in results_nested if not df.empty]
    if not results_nested:
        logger.warning("No valid model results were generated. Check your data and parameters.")
        return pd.DataFrame(columns=[STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.STD_ERROR, TIDY_DEFS.P_VALUE])
    
    # Combine results and apply FDR control
    results_df = pd.concat(results_nested, ignore_index=True)
    if fdr_control:
        results_df = control_fdr(results_df, fdr_method=fdr_method)
    
    logger.info(f"Completed model fitting for {len(results_df)} feature-term pairs.")
    return results_df

def fit_parallel_models_matrix(
    X_features: np.ndarray,
    X_model: np.ndarray,
    feature_names: List[str],
    term_names: List[str],
    n_jobs: int = 1,
    batch_size: int = 100,
    model_name: Optional[str] = None,
    progress_bar: bool = True,
    fdr_control: bool = True,
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit OLS models for multiple features in parallel using matrix interface.
    
    Parameters
    ----------
    X_features : np.ndarray
        Feature matrix where each column is a feature to model (n_samples x n_features)
    X_model : np.ndarray
        Design matrix (n_samples x n_terms)
    feature_names : List[str]
        Names of the features
    term_names : List[str]
        Names for the coefficients/predictors
    n_jobs : int
        Number of parallel jobs. Default is 1.
    batch_size : int
        Number of features to process in each batch.
    model_name : str, optional
        Name of the model for identification in output. Default is None.
    progress_bar : bool
        Whether to display a progress bar.
    fdr_control : bool
        Whether to apply FDR control to the results. Default is True.
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        Combined results from all models
    """
    # Input validation
    if X_features.shape[0] != X_model.shape[0]:
        raise ValueError("X_features and X_model must have same number of samples")
    if len(term_names) != X_model.shape[1]:
        raise ValueError(f"Length of term_names ({len(term_names)}) must match number of terms (columns of X_model: {X_model.shape[1]})")
    if len(feature_names) != X_features.shape[1]:
        raise ValueError(f"Length of feature_names ({len(feature_names)}) must match number of features (columns of X_features: {X_features.shape[1]})")

    # Verbose setting for progress bar
    verbose = 10 if progress_bar else 0
    
    # Run parallel processing
    logger.info(f"Starting parallel model fitting with {n_jobs} cores...")
    results_nested = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)(
        delayed(fit_feature_model_matrix)(
            X_features[:, i],
            X_model,
            feature_names[i],
            term_names,
            model_name=model_name,
            **model_kwargs
        ) for i in range(X_features.shape[1])
    )
    
    # Filter out empty DataFrames and combine results
    results_nested = [df for df in results_nested if not df.empty]
    if not results_nested:
        logger.warning("No valid model results were generated. Check your data and parameters.")
        return pd.DataFrame(columns=[STATISTICS_DEFS.FEATURE_NAME, TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.STD_ERROR, TIDY_DEFS.P_VALUE])
    
    # Combine results and apply FDR control
    results_df = pd.concat(results_nested, ignore_index=True)
    if fdr_control:
        results_df = control_fdr(results_df)
    
    logger.info(f"Completed model fitting for {len(results_df)} feature-term pairs.")
    return results_df 

def _handle_model_error(e: Exception, feature_name: str) -> pd.DataFrame:
    """
    Handle model fitting errors consistently.
    
    Parameters
    ----------
    e : Exception
        The caught exception
    feature_name : str
        Name of the feature being modeled
        
    Returns
    -------
    pd.DataFrame
        Empty DataFrame for failed model fits
    """
    if isinstance(e, ValueError) and "Unsupported model class" in str(e):
        raise e
    # Handle perfect collinearity and other numerical errors
    if any(msg in str(e).lower() for msg in ['singular', 'collinear', 'invertible']):
        logger.warning(f"Skipping feature {feature_name} due to perfect collinearity or numerical issues")
    else:
        logger.warning(f"Error in feature {feature_name}: {str(e)}")
    return pd.DataFrame()