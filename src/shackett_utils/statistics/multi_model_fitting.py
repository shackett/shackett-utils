"""
Functions for fitting models and applying FDR control to model results.

Public Functions
----------------
add_lfdr(results_df, grouping_vars=MULTITEST_GROUPING_VARS, require_groups=False, p_value_var=STATISTICS_DEFS.P_VALUE, lfdr_var=STATISTICS_DEFS.LFDR)
    Estimate local FDR values for p-values, optionally within groups.
control_fdr(results_df, fdr_method=FDR_METHODS_DEFS.QVALUE, grouping_vars=MULTITEST_GROUPING_VARS, require_groups=False, p_value_var=STATISTICS_DEFS.P_VALUE, q_value_var=STATISTICS_DEFS.Q_VALUE)
    Apply FDR control to p-values, optionally within groups.
fit_feature_model_formula(y, data, feature_name, formula, model_class=REGRESSION_TYPES.OLS, model_name=None, outputs=[STATISTICAL_SUMMARIES.TIDY], allow_failures=False, **model_kwargs)
    Fit a model for a single feature using formula interface.
fit_feature_model_matrix(y, X, feature_name, term_names, model_name=None, **model_kwargs)
    Fit an OLS model for a single feature using matrix interface.
fit_parallel_models_formula(X_features, data, feature_names, formula, model_class=None, model_name=None, outputs=[STATISTICAL_SUMMARIES.TIDY], n_jobs=1, allow_failures=False, fdr_control=True, fdr_method=FDR_METHODS_DEFS.BH, batch_size=100, progress_bar=True, **model_kwargs)
    Fit models in parallel for multiple features using formula interface.
fit_parallel_models_matrix(X_features, X_model, feature_names, term_names, n_jobs=1, batch_size=100, model_name=None, progress_bar=True, fdr_control=True, **model_kwargs)
    Fit OLS models for multiple features in parallel using matrix interface.
"""

from joblib import Parallel, delayed
import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multitest

from shackett_utils.statistics.model_fitting import (
    OLSModel,
    GAMModel,
)
from shackett_utils.statistics.constants import (
    FDR_METHODS,
    FDR_METHODS_DEFS,
    MULTITEST_GROUPING_VARS,
    REGRESSION_TYPES,
    STATISTICAL_SUMMARIES,
    STATISTICS_DEFS,
    TIDY_DEFS,
)
from shackett_utils.statistics.qvalue import estimate_lfdr, estimate_qvalues
from shackett_utils.statistics.utils import validate_regression_output_types

logger = logging.getLogger(__name__)


def add_lfdr(
    results_df: pd.DataFrame,
    grouping_vars: Optional[List[str]] = MULTITEST_GROUPING_VARS,
    require_groups: bool = False,
    p_value_var: str = STATISTICS_DEFS.P_VALUE,
    lfdr_var: str = STATISTICS_DEFS.LFDR,
) -> pd.DataFrame:
    """
    Add and estimate local FDR values for p-values, optionally within groups.

    Local FDR estimates the posterior probability that each individual
    test is a true null, in contrast to q-values which control the
    global FDR at a threshold. Estimation uses Storey's probit-KDE
    method via ``estimate_lfdr()``.

    Grouping follows the same logic as ``control_fdr()``: correction
    is applied independently within each group, so pi0 and the marginal
    p-value density are estimated separately per group. This is
    appropriate when groups correspond to distinct experiments or
    modalities with different signal compositions.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with model results containing p-values.
    grouping_vars : Optional[List[str]]
        Columns to group by. Columns absent from results_df are
        silently ignored unless require_groups=True.
    require_groups : bool
        If True, raise if any grouping variable is absent from results_df.
    p_value_var : str
        Column name for p-values.
    lfdr_var : str
        Column name for local FDR values (written in-place).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with lfdr_var column added or updated.

    Raises
    ------
    ValueError
        If require_groups=True and a grouping variable is missing.
    """
    group_cols = _resolve_groups(results_df, grouping_vars, require_groups)

    for mask, label in _iter_groups(results_df, group_cols):
        logger.info(f"Estimating local FDR within {label}")
        p_values = results_df.loc[mask, p_value_var].values
        if len(p_values) == 0:
            results_df.loc[mask, lfdr_var] = np.nan
        else:
            results_df.loc[mask, lfdr_var] = estimate_lfdr(p_values)

    return results_df


def control_fdr(
    results_df: pd.DataFrame,
    fdr_method: str = FDR_METHODS_DEFS.QVALUE,
    grouping_vars: Optional[List[str]] = MULTITEST_GROUPING_VARS,
    require_groups: bool = False,
    p_value_var: str = STATISTICS_DEFS.P_VALUE,
    q_value_var: str = STATISTICS_DEFS.Q_VALUE,
) -> pd.DataFrame:
    """
    Apply FDR control to p-values, optionally within groups.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with model results containing p-values.
    fdr_method : str
        FDR correction method. Must be one of FDR_METHODS.
    grouping_vars : Optional[List[str]]
        Columns to group by for FDR control. Columns absent from
        results_df are silently ignored unless require_groups=True.
    require_groups : bool
        If True, raise if any grouping variable is absent from results_df.
    p_value_var : str
        Column name for p-values.
    q_value_var : str
        Column name for q-values (written in-place).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with q_value_var column added or updated.

    Raises
    ------
    ValueError
        If fdr_method is not in FDR_METHODS, results_df is empty,
        or require_groups=True and a grouping variable is missing.
    """
    if fdr_method not in FDR_METHODS:
        raise ValueError(
            f"Invalid FDR method: {fdr_method!r}. Must be one of: {FDR_METHODS}"
        )

    group_cols = _resolve_groups(results_df, grouping_vars, require_groups)

    for mask, label in _iter_groups(results_df, group_cols):
        logger.info(f"Applying FDR control ({fdr_method}) within {label}")
        _apply_fdr_correction(results_df, mask, fdr_method, p_value_var, q_value_var)

    return results_df


def fit_feature_model_formula(
    y: np.ndarray,
    data: pd.DataFrame,
    feature_name: str,
    formula: str,
    model_class: str = REGRESSION_TYPES.OLS,
    model_name: Optional[str] = None,
    outputs: List[str] = [STATISTICAL_SUMMARIES.TIDY],
    allow_failures: bool = False,
    **model_kwargs,
) -> Dict[str, pd.DataFrame]:
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
    outputs : list of str
        Which summaries to return. Any combination of 'tidy', 'glance', 'augment'.
        Default is [STATISTICAL_SUMMARIES.TIDY].
    allow_failures : bool
        If True, handle errors gracefully and return dict of empty DataFrames.
        If False, raise exceptions for debugging. Default is False.
    **model_kwargs
        Additional arguments passed to model fitting

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary keyed by output type containing the requested summaries.
    """
    validate_regression_output_types(outputs)

    # Filter out missing values
    valid_mask = ~np.isnan(y)
    n_valid = np.sum(valid_mask)
    if n_valid < len(y):
        logger.debug(
            f"Filtering {len(y) - n_valid} missing values for feature {feature_name}"
        )
        y = y[valid_mask]
        data = data.iloc[valid_mask]

    # Check for zero variance
    if np.var(y) == 0:
        logger.warning(f"Skipping feature {feature_name} due to zero variance")
        return {key: pd.DataFrame() for key in outputs}

    try:
        if model_class.lower() == REGRESSION_TYPES.OLS:
            model = OLSModel(feature_name=feature_name, model_name=model_name)
        elif model_class.lower() == REGRESSION_TYPES.GAM:
            model = GAMModel(feature_name=feature_name, model_name=model_name)
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        model_data = data.copy()
        model_data["y"] = y
        model.fit(formula, data=model_data, **model_kwargs)

        result = {}
        for key in outputs:
            if key == STATISTICAL_SUMMARIES.TIDY:
                result[key] = model.tidy()
            elif key == STATISTICAL_SUMMARIES.GLANCE:
                result[key] = model.glance()
            elif key == STATISTICAL_SUMMARIES.AUGMENT:
                result[key] = model.augment()
            else:
                raise ValueError(
                    f"Unknown output type: '{key}'. Use 'tidy', 'glance', or 'augment'."
                )
        return result

    except Exception as e:
        if allow_failures:
            return _handle_model_error(e, feature_name, outputs)
        else:
            raise


def fit_feature_model_matrix(
    y: np.ndarray,
    X: np.ndarray,
    feature_name: str,
    term_names: List[str],
    model_name: Optional[str] = None,
    **model_kwargs,
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
        logger.debug(
            f"Filtering {len(y) - n_valid} missing values for feature {feature_name}"
        )
        y = y[valid_mask]
        X = X[valid_mask]

    # Check for zero variance
    if np.var(y) == 0:
        logger.warning(f"Skipping feature {feature_name} due to zero variance")
        return pd.DataFrame()

    try:
        # Validate input dimensions
        if len(y) != X.shape[0]:
            raise ValueError(
                "Response vector and model matrix must have same number of samples"
            )
        if len(term_names) != X.shape[1]:
            raise ValueError(
                "Number of coefficient names must match number of model matrix columns"
            )

        model = OLSModel(feature_name=feature_name, model_name=model_name)
        model.fit_xy(X, y, term_names=term_names, **model_kwargs)

        return model.tidy()

    except Exception as e:
        return _handle_model_error(
            e, feature_name, outputs=[STATISTICAL_SUMMARIES.TIDY]
        )[STATISTICAL_SUMMARIES.TIDY]


def fit_parallel_models_formula(
    X_features: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    formula: str,
    model_class: Optional[str] = None,
    model_name: Optional[str] = None,
    outputs: List[str] = [STATISTICAL_SUMMARIES.TIDY],
    n_jobs: int = 1,
    allow_failures: bool = False,
    fdr_control: bool = True,
    fdr_method: str = FDR_METHODS_DEFS.QVALUE,
    batch_size: int = 100,
    progress_bar: bool = True,
    **model_kwargs,
) -> Dict[str, pd.DataFrame]:
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
    outputs : list of str
        Which summaries to return. Any combination of 'tidy', 'glance', 'augment'.
        Default is [STATISTICAL_SUMMARIES.TIDY].
    n_jobs : int
        Number of parallel jobs. Default is 1.
    allow_failures : bool
        If True, handle errors gracefully. Default is False.
    fdr_control : bool
        Whether to perform FDR control on tidy output. Default is True.
    fdr_method : str
        FDR control method. Default is 'qvalue'.
        Must be one of FDR_METHODS.
    batch_size : int
        Number of features to process in each batch. Default is 100.
    progress_bar : bool
        Whether to display a progress bar. Default is True.
    **model_kwargs
        Additional arguments passed to model fitting

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary keyed by output type containing combined results for all features.
    """
    # Input validation
    if len(feature_names) != X_features.shape[1]:
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) must match number of features in X_features ({X_features.shape[1]})"
        )
    if X_features.shape[0] != len(data):
        raise ValueError("X_features and data must have same number of samples")

    inferred_model_class = _detect_model_class_from_formula(formula)
    if model_class is None:
        model_class = inferred_model_class
        logger.info(
            f"Auto-detected model_class='{model_class}' for formula='{formula}'"
        )
    else:
        if model_class != inferred_model_class:
            if (
                model_class == REGRESSION_TYPES.OLS
                and inferred_model_class == REGRESSION_TYPES.GAM
            ):
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

    formula = _validate_formula(formula)
    verbose = 10 if progress_bar else 0

    logger.info(f"Starting parallel model fitting with {n_jobs} cores...")
    results_nested = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)(
        delayed(fit_feature_model_formula)(
            X_features[:, i],
            data,
            feature_names[i],
            formula=formula,
            model_class=model_class,
            model_name=model_name,
            outputs=outputs,
            allow_failures=allow_failures,
            **model_kwargs,
        )
        for i in range(X_features.shape[1])
    )

    # Concat per output key, skipping empty DataFrames
    empty_df = pd.DataFrame(
        columns=[
            STATISTICS_DEFS.FEATURE_NAME,
            TIDY_DEFS.TERM,
            TIDY_DEFS.ESTIMATE,
            TIDY_DEFS.STD_ERROR,
            TIDY_DEFS.P_VALUE,
        ]
    )
    combined = {}
    for key in outputs:
        frames = [r[key] for r in results_nested if not r[key].empty]
        if not frames:
            logger.warning(f"No valid results for output '{key}'.")
            combined[key] = empty_df
        else:
            # Preserve index for augment (obs identifiers) so pivot works correctly
            ignore_index = key != STATISTICAL_SUMMARIES.AUGMENT
            combined[key] = pd.concat(frames, ignore_index=ignore_index)

    # FDR control applies only to tidy output
    if (
        fdr_control
        and STATISTICAL_SUMMARIES.TIDY in combined
        and not combined[STATISTICAL_SUMMARIES.TIDY].empty
    ):
        combined[STATISTICAL_SUMMARIES.TIDY] = control_fdr(
            combined[STATISTICAL_SUMMARIES.TIDY], fdr_method=fdr_method
        )

    logger.info(f"Completed model fitting for {X_features.shape[1]} features.")
    return combined


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
    fdr_method: str = FDR_METHODS_DEFS.QVALUE,
    **model_kwargs,
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
    fdr_method : str
        FDR control method. Default is 'qvalue'.
        Must be one of FDR_METHODS.
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
        raise ValueError(
            f"Length of term_names ({len(term_names)}) must match number of terms (columns of X_model: {X_model.shape[1]})"
        )
    if len(feature_names) != X_features.shape[1]:
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) must match number of features (columns of X_features: {X_features.shape[1]})"
        )

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
            **model_kwargs,
        )
        for i in range(X_features.shape[1])
    )

    # Filter out empty DataFrames and combine results
    results_nested = [df for df in results_nested if not df.empty]
    if not results_nested:
        logger.warning(
            "No valid model results were generated. Check your data and parameters."
        )
        return pd.DataFrame(
            columns=[
                STATISTICS_DEFS.FEATURE_NAME,
                TIDY_DEFS.TERM,
                TIDY_DEFS.ESTIMATE,
                TIDY_DEFS.STD_ERROR,
                TIDY_DEFS.P_VALUE,
            ]
        )

    # Combine results and apply FDR control
    results_df = pd.concat(results_nested, ignore_index=True)
    if fdr_control:
        results_df = control_fdr(results_df, fdr_method=fdr_method)

    logger.info(f"Completed model fitting for {len(results_df)} feature-term pairs.")
    return results_df


def _apply_fdr_correction(
    df: pd.DataFrame,
    mask: pd.Series,
    fdr_method: str,
    p_value_var: str = STATISTICS_DEFS.P_VALUE,
    q_value_var: str = STATISTICS_DEFS.Q_VALUE,
) -> None:
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
    p_value_var : str
        Column name for p-values
    q_value_var : str
        Column name for q-values
    """
    p_values = df.loc[mask, p_value_var].values
    if len(p_values) > 0:  # Only perform correction if we have p-values

        if fdr_method == FDR_METHODS_DEFS.BH:
            df.loc[mask, q_value_var] = multitest.multipletests(
                p_values, method=fdr_method
            )[1]
        elif fdr_method == FDR_METHODS_DEFS.QVALUE:
            df.loc[mask, q_value_var] = estimate_qvalues(p_values)
    else:
        df.loc[mask, q_value_var] = np.nan


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
    has_smooth_terms = bool(re.search(r"(^|\s|~|\+)\s*s\([^)]+\)", formula))
    model_class = REGRESSION_TYPES.GAM if has_smooth_terms else REGRESSION_TYPES.OLS
    logger.debug(f"Detected model_class='{model_class}' for formula='{formula}'")
    return model_class


def _handle_model_error(
    e: Exception, feature_name: str, outputs: List[str]
) -> Dict[str, pd.DataFrame]:
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
    if any(msg in str(e).lower() for msg in ["singular", "collinear", "invertible"]):
        logger.warning(
            f"Skipping feature {feature_name} due to perfect collinearity or numerical issues"
        )
    else:
        logger.warning(f"Error in feature {feature_name}: {str(e)}")
    return {key: pd.DataFrame() for key in outputs}


def _iter_groups(
    results_df: pd.DataFrame,
    group_cols: List[str],
):
    """
    Yield (mask, group_label) for each group, or a single all-True mask
    if no grouping columns are provided.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame to iterate over.
    group_cols : List[str]
        Columns to group by.

    Yields
    ------
    mask : pd.Series
        Boolean mask for the current group.
    label : str
        Human-readable label for logging.
    """
    if not group_cols:
        yield pd.Series(True, index=results_df.index), "all rows"
        return

    by = group_cols[0] if len(group_cols) == 1 else group_cols
    for name, group in results_df.groupby(by):
        label = f"{group_cols}={name}" if len(group_cols) > 1 else f"{by}={name}"
        yield results_df.index.isin(group.index), label


def _resolve_groups(
    results_df: pd.DataFrame,
    grouping_vars: Optional[List[str]],
    require_groups: bool,
) -> List[str]:
    """
    Resolve which grouping columns are present in the DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame to check for grouping columns.
    grouping_vars : Optional[List[str]]
        Requested grouping columns.
    require_groups : bool
        If True, raise if any grouping variables are missing.

    Returns
    -------
    List[str]
        Grouping columns present in results_df.
    """
    if not grouping_vars:
        return []

    group_cols = [col for col in grouping_vars if col in results_df.columns]

    if require_groups:
        missing = [col for col in grouping_vars if col not in results_df.columns]
        if missing:
            raise ValueError(f"Grouping variables {missing} not found in results_df")

    return group_cols


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
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError("Formula must contain exactly one '~' character")

    # Handle formulas starting with ~
    lhs = parts[0].strip()
    rhs = parts[1].strip()
    if not lhs:
        formula = f"y ~ {rhs}"
    elif lhs != "y":
        raise ValueError(f"Formula must use 'y' as dependent variable, got '{lhs}'")
    else:
        formula = f"y ~ {rhs}"

    # If data is provided, check that all variables are numeric
    if data is not None:
        # Remove s() from terms like s(x)
        rhs_clean = re.sub(r"s\((.*?)\)", r"\1", rhs)
        # Split on operators and get unique variables
        variables = set()
        variables.update(
            var.strip() for var in re.split(r"[+\-*/]", rhs_clean) if var.strip()
        )

        # Check each variable
        for var in variables:
            if var not in data.columns:
                raise ValueError(f"Variable '{var}' not found in data")
            if not np.issubdtype(data[var].dtype, np.number):
                raise ValueError(
                    f"Variable '{var}' must be numeric, got dtype {data[var].dtype}"
                )

    return formula
