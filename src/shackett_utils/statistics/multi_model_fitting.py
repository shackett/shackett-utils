from typing import List, Dict, Any, Optional, Callable, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import statsmodels.stats.multitest as multitest
import logging
from .model_fitting import StatisticalModel, OLSModel, GAMModel

logger = logging.getLogger(__name__)

def fit_feature_model_formula(
    y: np.ndarray,
    X: np.ndarray,
    feature_name: str,
    formula: str,
    term_names: List[str],
    model_class: str = 'gam',
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit a model for a single feature using formula interface.
    
    Parameters
    ----------
    y : np.ndarray
        Feature values to model (response variable)
    X : np.ndarray
        Model matrix (predictors)
    feature_name : str
        Name of the feature being modeled
    formula : str
        Model formula (e.g. 'y ~ x1 + s(x2)')
    term_names : List[str]
        Names for the coefficients/predictors
    model_class : str
        Type of model to fit ('ols', 'gam', etc.)
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model statistics for each coefficient
    """
    if np.var(y) == 0:
        logger.debug(f"Skipping feature {feature_name} due to zero variance")
        return pd.DataFrame()
    
    try:
        # Create model instance
        if model_class.lower() == 'ols':
            model = OLSModel(feature_name=feature_name)
        elif model_class.lower() == 'gam':
            model = GAMModel(feature_name=feature_name)
        else:
            raise ValueError(f"Unsupported model class: {model_class}")
        
        # Create temporary DataFrame for formula interface
        df = pd.DataFrame(X, columns=term_names)
        df['y'] = y
        model.fit(formula, data=df, **model_kwargs)
        
        return model.tidy()
            
    except ValueError as e:
        if "Unsupported model class" in str(e):
            raise
        logger.warning(f"Error in feature {feature_name}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error in feature {feature_name}: {str(e)}")
        return pd.DataFrame()

def fit_feature_model_matrix(
    y: np.ndarray,
    X: np.ndarray,
    feature_name: str,
    term_names: List[str],
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
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model statistics for each coefficient
    """
    if np.var(y) == 0:
        logger.debug(f"Skipping feature {feature_name} due to zero variance")
        return pd.DataFrame()
    
    try:
        # Validate input dimensions
        if len(y) != X.shape[0]:
            raise ValueError("Response vector and model matrix must have same number of samples")
        if len(term_names) != X.shape[1]:
            raise ValueError("Number of coefficient names must match number of model matrix columns")
        
        model = OLSModel(feature_name=feature_name)
        model.fit_xy(X, y, term_names=term_names, **model_kwargs)
        
        return model.tidy()
            
    except Exception as e:
        logger.warning(f"Error in feature {feature_name}: {str(e)}")
        return pd.DataFrame()

def fit_parallel_models_formula(
    X_features: np.ndarray,
    X_model: np.ndarray,
    feature_names: List[str],
    formula: str,
    term_names: List[str],
    model_class: str = 'gam',
    n_jobs: int = -1,
    batch_size: int = 100,
    progress_bar: bool = True,
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit models in parallel for multiple features using formula interface.
    
    Parameters
    ----------
    X_features : np.ndarray
        Feature matrix where each column is a feature to model (n_samples x n_features)
    X_model : np.ndarray
        Model matrix for predictors (n_samples x n_predictors)
    feature_names : List[str]
        Names of the features
    formula : str
        Model formula (e.g. 'y ~ x1 + s(x2)')
    term_names : List[str]
        Names for the coefficients/predictors (without intercept)
    model_class : str
        Type of model to fit ('ols', 'gam', etc.)
    n_jobs : int
        Number of parallel jobs. -1 means using all processors.
    batch_size : int
        Number of features to process in each batch.
    progress_bar : bool
        Whether to display a progress bar.
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model statistics for each feature-coefficient pair
    """
    # Input validation
    if X_features.shape[0] != X_model.shape[0]:
        raise ValueError("X_features and X_model must have same number of samples")
    if len(term_names) != X_model.shape[1]:
        raise ValueError(f"Length of term_names ({len(term_names)}) must match number of predictors (columns of X_model: {X_model.shape[1]})")
    if len(feature_names) != X_features.shape[1]:
        raise ValueError(f"Length of feature_names ({len(feature_names)}) must match number of features (columns of X_features: {X_features.shape[1]})")

    # Verbose setting for progress bar
    verbose = 10 if progress_bar else 0
    
    # Run parallel processing
    logger.info(f"Starting parallel model fitting with {n_jobs} cores...")
    results_nested = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)(
        delayed(fit_feature_model_formula)(
            X_features[:, i],
            X_model,
            feature_names[i],
            formula=formula,
            term_names=term_names,
            model_class=model_class,
            **model_kwargs
        ) for i in range(X_features.shape[1])
    )
    
    results = _process_parallel_results(results_nested)
    logger.info(f"Completed model fitting for {len(results)} feature-term pairs.")
    return results

def fit_parallel_models_matrix(
    X_features: np.ndarray,
    X_model: np.ndarray,
    feature_names: List[str],
    term_names: List[str],
    n_jobs: int = -1,
    batch_size: int = 100,
    progress_bar: bool = True,
    **model_kwargs
) -> pd.DataFrame:
    """
    Fit OLS models in parallel for multiple features using matrix interface.
    
    Parameters
    ----------
    X_features : np.ndarray
        Feature matrix where each column is a feature to model (n_samples x n_features)
    X_model : np.ndarray
        Model matrix for predictors (n_samples x n_predictors), should include intercept
    feature_names : List[str]
        Names of the features
    term_names : List[str]
        Names for the coefficients/predictors
    n_jobs : int
        Number of parallel jobs. -1 means using all processors.
    batch_size : int
        Number of features to process in each batch.
    progress_bar : bool
        Whether to display a progress bar.
    **model_kwargs : 
        Additional arguments passed to model fitting
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model statistics for each feature-coefficient pair
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
            **model_kwargs
        ) for i in range(X_features.shape[1])
    )
    
    results = _process_parallel_results(results_nested)
    logger.info(f"Completed model fitting for {len(results)} feature-term pairs.")
    return results

def _process_parallel_results(results_nested: List[pd.DataFrame]) -> pd.DataFrame:
    """Process results from parallel model fitting."""
    # Handle empty results
    if not results_nested:
        logger.warning("No model results were generated. Check your data and parameters.")
        return pd.DataFrame()
    
    # Combine results
    results_df = pd.concat(results_nested, ignore_index=True)
    
    # Add FDR correction if p-values present
    if not results_df.empty and "p_value" in results_df.columns:
        # Group by term and calculate FDR-corrected p-values
        for term in results_df["term"].unique():
            mask = results_df["term"] == term
            p_values = results_df.loc[mask, "p_value"].values
            if len(p_values) > 0:  # Only perform correction if we have p-values
                results_df.loc[mask, "fdr_bh"] = multitest.multipletests(
                    p_values, 
                    method="fdr_bh"
                )[1]
            else:
                results_df.loc[mask, "fdr_bh"] = np.nan
    
    return results_df 