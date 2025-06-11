import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import LinearGAM
from abc import ABC, abstractmethod
from typing import Optional, Dict
from .constants import REQUIRED_TIDY_VARS, STATISTICS_DEFS, TIDY_DEFS

logger = logging.getLogger(__name__)


def _validate_xy_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate model matrix and response vector inputs"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got {y.ndim} dimensions")
    if X.shape[0] != len(y):
        raise ValueError(f"X and y must have same length: {X.shape[0]} vs {len(y)}")


def _calculate_residual_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate residual-based statistics"""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "r_squared": r_squared,
        "residuals": residuals,
    }


class StatisticalModel(ABC):
    """Abstract base class for statistical models."""

    def __init__(
        self, feature_name: Optional[str] = None, model_name: Optional[str] = None
    ):
        """
        Initialize model.

        Parameters
        ----------
        feature_name : str, optional
            Name of the feature being modeled. Default is None.
        model_name : str, optional
            Name of the model for identification. Default is None.
        """
        self.feature_name = feature_name
        self.model_name = model_name
        self.fitted_model = None
        self.formula = None
        self.data = None
        self.feature_name = feature_name  # Name of the feature being modeled
        self.model_name = model_name  # Name of the model type/variant
        self.term_names = None
        self._X = None
        self._y = None

    def _add_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature and model identifiers to output dataframes if present.
        Ensures consistent column ordering with identifiers first.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to add identifiers to

        Returns
        -------
        pd.DataFrame
            DataFrame with identifiers added and columns reordered
        """
        # Add identifiers if present
        if self.model_name is not None:
            df[STATISTICS_DEFS.MODEL_NAME] = self.model_name
        if self.feature_name is not None:
            df[STATISTICS_DEFS.FEATURE_NAME] = self.feature_name

        # Define column order
        identifier_cols = []
        if STATISTICS_DEFS.MODEL_NAME in df.columns:
            identifier_cols.append(STATISTICS_DEFS.MODEL_NAME)
        if STATISTICS_DEFS.TERM in df.columns:
            identifier_cols.append(STATISTICS_DEFS.TERM)
        if STATISTICS_DEFS.FEATURE_NAME in df.columns:
            identifier_cols.append(STATISTICS_DEFS.FEATURE_NAME)

        # Get remaining columns in their current order
        other_cols = [col for col in df.columns if col not in identifier_cols]

        # Reorder columns
        return df[identifier_cols + other_cols]

    @abstractmethod
    def fit(self, formula: str, data: pd.DataFrame, **kwargs) -> "StatisticalModel":
        """Fit the model using formula and data"""
        pass

    @abstractmethod
    def fit_xy(
        self, X: np.ndarray, y: np.ndarray, term_names: Optional[list] = None, **kwargs
    ) -> "StatisticalModel":
        """Fit the model using model matrix X and response y"""
        pass

    @abstractmethod
    def tidy(self) -> pd.DataFrame:
        """Return coefficient-level information (like broom::tidy)"""
        pass

    @abstractmethod
    def glance(self) -> pd.DataFrame:
        """Return model-level statistics (like broom::glance)"""
        pass

    def augment(self) -> pd.DataFrame:
        """Return original data with fitted values and residuals"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        if self.data is not None:
            # Formula-based fitting
            result = self.data.copy()
            result[".fitted"] = self.fitted_model.fittedvalues
            result[".resid"] = self.fitted_model.resid
        else:
            # Matrix-based fitting
            result = pd.DataFrame(self._X, columns=self.term_names)
            result["y"] = self._y
            result[".fitted"] = self.fitted_model.fittedvalues
            result[".resid"] = self.fitted_model.resid

        # Add additional diagnostics if available
        if hasattr(self.fitted_model, "get_influence"):
            influence = self.fitted_model.get_influence()
            result[".std_resid"] = influence.resid_studentized_internal
            result[".hat"] = influence.hat_matrix_diag
            result[".cooksd"] = influence.cooks_distance[0]

        result = self._add_identifiers(result)

        return result


class OLSModel(StatisticalModel):
    """OLS model wrapper using statsmodels"""

    def fit(self, formula: str, data: pd.DataFrame, **kwargs) -> "OLSModel":
        """Fit OLS model using formula

        Parameters
        ----------
        formula : str
            Model formula (e.g. 'y ~ x1 + x2')
            Must include explicit dependent variable.
        data : pd.DataFrame
            Data containing the variables
        **kwargs : dict
            Additional arguments passed to OLS.fit()
        """
        self.formula = formula
        self.data = data

        # Validate formula has dependent variable
        parts = formula.split("~")
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError("Formula must include explicit dependent variable")

        logger.debug(f"Fitting OLS with formula '{formula}'")
        self.fitted_model = smf.ols(formula, data=data).fit(**kwargs)
        return self

    def fit_xy(
        self, X: np.ndarray, y: np.ndarray, term_names: Optional[list] = None, **kwargs
    ) -> "OLSModel":
        """Fit OLS model using model matrix and response"""
        _validate_xy_inputs(X, y)

        self._X = X
        self._y = y
        self.term_names = term_names or [f"x{i}" for i in range(X.shape[1])]

        self.fitted_model = sm.OLS(y, X).fit(**kwargs)
        return self

    def tidy(self) -> pd.DataFrame:
        """Return coefficient information"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        # Create tidy dataframe with proper index handling
        params = self.fitted_model.params
        conf_int = self.fitted_model.conf_int()

        if isinstance(params, pd.Series):
            terms = params.index.tolist()
        else:
            terms = self.term_names

        # Ensure all numeric values are float64
        tidy_df = pd.DataFrame(
            {
                "term": terms,
                "estimate": np.asarray(params, dtype=np.float64),
                "std_error": np.asarray(self.fitted_model.bse, dtype=np.float64),
                "statistic": np.asarray(self.fitted_model.tvalues, dtype=np.float64),
                "p_value": np.asarray(self.fitted_model.pvalues, dtype=np.float64),
                "conf_low": np.asarray(
                    (
                        conf_int[:, 0]
                        if isinstance(conf_int, np.ndarray)
                        else conf_int.iloc[:, 0]
                    ),
                    dtype=np.float64,
                ),
                "conf_high": np.asarray(
                    (
                        conf_int[:, 1]
                        if isinstance(conf_int, np.ndarray)
                        else conf_int.iloc[:, 1]
                    ),
                    dtype=np.float64,
                ),
            }
        )

        # Add feature and model name if present
        tidy_df = self._add_identifiers(tidy_df)

        return tidy_df.reset_index(drop=True)

    def glance(self) -> pd.DataFrame:
        """Return model-level statistics"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        model = self.fitted_model

        glance_df = pd.DataFrame(
            {
                "r_squared": [model.rsquared],
                "adj_r_squared": [model.rsquared_adj],
                "sigma": [np.sqrt(model.mse_resid)],
                "statistic": [model.fvalue],
                "p_value": [model.f_pvalue],
                "df": [model.df_model],
                "df_residual": [model.df_resid],
                "nobs": [int(model.nobs)],
                "aic": [model.aic],
                "bic": [model.bic],
                "log_likelihood": [model.llf],
            }
        )

        glance_df = self._add_identifiers(glance_df)

        return glance_df


class GAMModel(StatisticalModel):
    """GAM model wrapper using pygam"""

    def __init__(
        self, feature_name: Optional[str] = None, model_name: Optional[str] = None
    ):
        super().__init__(feature_name=feature_name, model_name=model_name)
        self.smooth_terms = []
        self.family = None

    def _parse_formula(self, formula: str) -> tuple:
        """Parse formula string to extract dependent and independent variables

        Parameters
        ----------
        formula : str
            Model formula (e.g. 'y ~ x1 + x2' or 'response ~ x1 + s(x2)')
            Must include explicit dependent variable.

        Returns
        -------
        tuple
            (y_var, x_vars, smooth_terms)
            - y_var: name of response variable from formula
            - x_vars: list of predictor variable names
            - smooth_terms: list of variables that should be smoothed
        """
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError("Formula must be in format 'y ~ x1 + x2 + ...'")

        # Get response variable
        y_var = parts[0].strip()
        if not y_var:
            raise ValueError("Formula must include explicit dependent variable")

        # Parse predictor terms
        x_part = parts[1].strip()
        raw_terms = [term.strip() for term in x_part.split("+")]

        x_vars = []
        smooth_terms = []

        for term in raw_terms:
            # Check if it's a smooth term like s(variable_name)
            if term.startswith("s(") and term.endswith(")"):
                # Extract variable name from s(variable_name)
                var_name = term[2:-1].strip()
                x_vars.append(var_name)
                smooth_terms.append(var_name)
            else:
                # Regular linear term
                x_vars.append(term)

        logger.debug(
            f"Parsed formula '{formula}': response='{y_var}', predictors={x_vars}, smooth_terms={smooth_terms}"
        )
        return y_var, x_vars, smooth_terms

    def _build_gam_terms(self, x_vars: list, smooth_terms: list) -> list:
        """Build GAM terms list for model fitting"""
        from pygam.terms import TermList, l, s

        terms = []
        for i, var in enumerate(x_vars):
            if var in smooth_terms:
                logger.debug(f"Adding smooth term s({i}) for variable '{var}'")
                terms.append(s(i))
            else:
                logger.debug(f"Adding linear term l({i}) for variable '{var}'")
                terms.append(l(i))
        return TermList(*terms)

    def fit(
        self, formula: str, data: pd.DataFrame, family: str = "gaussian", **kwargs
    ) -> "GAMModel":
        """
        Fit GAM using formula interface

        Parameters
        ----------
        formula : str
            Model formula (e.g. 'y ~ x1 + s(x2)')
        data : pd.DataFrame
            Data containing the variables
        family : str
            Distribution family for GAM ('gaussian', 'binomial', etc.)
        **kwargs :
            Additional arguments passed to GAM fitting

        Returns
        -------
        self : GAMModel
            Fitted model
        """
        from pygam import LogisticGAM, PoissonGAM, GammaGAM

        # Map family names to GAM classes
        family_map = {
            "gaussian": LinearGAM,
            "binomial": LogisticGAM,
            "poisson": PoissonGAM,
            "gamma": GammaGAM,
        }

        if family not in family_map:
            raise ValueError(
                f"Unsupported family: {family}. Available: {list(family_map.keys())}"
            )

        # Make a copy to avoid modifying original
        self.data = data.copy()
        self.formula = formula
        self.family = family

        # Parse formula
        y_var, x_vars, smooth_terms = self._parse_formula(formula)
        self.term_names = x_vars
        self.smooth_terms = smooth_terms

        # Convert all numeric columns to float64
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce").astype(
                    np.float64
                )
            elif isinstance(self.data[col].dtype, pd.CategoricalDtype):
                # For categorical variables, return the codes
                self.data[col] = self.data[col].cat.codes
            else:
                # For other types (e.g. string), try to convert to categorical first
                try:
                    self.data[col] = pd.Categorical(self.data[col]).codes
                except Exception as e:
                    raise ValueError(
                        f"Could not convert variable {col} to numeric: {str(e)}"
                    )

        # Extract X and y
        y = self.data[y_var].values
        X = self.data[x_vars].values

        logger.debug(
            f"Fitting GAM with X shape {X.shape}, y shape {y.shape}, X dtypes {[X[:, i].dtype for i in range(X.shape[1])]}"
        )

        # Build GAM terms
        terms = self._build_gam_terms(x_vars, smooth_terms)

        try:
            # Use the appropriate GAM class for the family
            GAMClass = family_map[family]
            self.fitted_model = GAMClass(terms, **kwargs).fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Failed to fit GAM model: {str(e)}")

        return self

    def fit_xy(
        self, X: np.ndarray, y: np.ndarray, term_names: Optional[list] = None, **kwargs
    ) -> "GAMModel":
        """Matrix-based fitting is not supported for GAM models.

        GAMs require a formula-based interface to specify smooth terms. Use the fit() method with
        a formula string instead.
        """
        raise NotImplementedError(
            "Matrix-based fitting is not supported for GAM models. "
            "Use the fit() method with a formula string instead, e.g.: "
            "'y ~ x1 + s(x2)' where s() indicates smooth terms."
        )

    def tidy(self) -> pd.DataFrame:
        """Return coefficient/term information for GAM"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        model = self.fitted_model
        terms_info = []

        # For each feature, get information
        for i, feature_name in enumerate(self.term_names):
            is_smooth = feature_name in self.smooth_terms

            if is_smooth:
                # For smooth terms, get effective degrees of freedom
                try:
                    edf = (
                        model.statistics_["edof_per_coef"][i]
                        if hasattr(model, "statistics_")
                        else None
                    )
                    estimate = None  # Smooth terms don't have single coefficients
                    p_value = (
                        model.statistics_["p_values"][i]
                        if hasattr(model, "statistics_")
                        else None
                    )
                except (KeyError, IndexError, AttributeError):
                    edf = None
                    estimate = None
                    p_value = None

                terms_info.append(
                    {
                        "term": f"s({feature_name})",
                        "type": "smooth",
                        "edf": np.float64(edf) if edf is not None else np.nan,
                        "estimate": (
                            np.float64(estimate) if estimate is not None else np.nan
                        ),
                        "p_value": (
                            np.float64(p_value) if p_value is not None else np.nan
                        ),
                        "std_error": np.nan,  # Smooth terms don't have standard errors in the same way
                    }
                )
            else:
                # For linear terms, get coefficient and p-value
                try:
                    coef = model.coef_[i] if hasattr(model, "coef_") else None
                    p_value = (
                        model.statistics_["p_values"][i]
                        if hasattr(model, "statistics_")
                        else None
                    )
                except (IndexError, AttributeError, KeyError):
                    coef = None
                    p_value = None

                terms_info.append(
                    {
                        "term": feature_name,
                        "type": "linear",
                        "estimate": np.float64(coef) if coef is not None else np.nan,
                        "edf": np.float64(1.0),  # Linear terms have 1 degree of freedom
                        "p_value": (
                            np.float64(p_value) if p_value is not None else np.nan
                        ),
                        "std_error": np.nan,  # GAM doesn't provide standard errors in the same way as OLS
                    }
                )

        tidy_df = pd.DataFrame(terms_info)

        # Add feature name if present
        tidy_df = self._add_identifiers(tidy_df)

        return tidy_df

    def glance(self) -> pd.DataFrame:
        """Return model-level statistics for GAM"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        model = self.fitted_model

        # Calculate basic statistics
        try:
            if self.data is not None:
                # Formula-based fitting
                predictions = model.predict(self.data[self.term_names].values)
                y_var = self._parse_formula(self.formula)[0]
                y_actual = self.data[y_var].values
            else:
                # Matrix-based fitting
                predictions = model.predict(self._X)
                y_actual = self._y

            # Calculate R-squared and other stats
            stats = _calculate_residual_stats(y_actual, predictions)

            # Get statistics from model if available
            try:
                deviance = (
                    model.statistics_["deviance"]
                    if hasattr(model, "statistics_")
                    else stats["ss_res"]
                )
                aic = (
                    model.statistics_["AIC"] if hasattr(model, "statistics_") else None
                )
                aicc = (
                    model.statistics_["AICc"] if hasattr(model, "statistics_") else None
                )
                edf = (
                    model.statistics_["edof"] if hasattr(model, "statistics_") else None
                )
            except (KeyError, AttributeError):
                deviance = stats["ss_res"]
                aic = None
                aicc = None
                edf = None

            glance_df = pd.DataFrame(
                {
                    "deviance": [deviance],
                    "r_squared": [stats["r_squared"]],
                    "aic": [aic],
                    "aicc": [aicc],
                    "edf": [edf],
                    "nobs": [len(y_actual)],
                }
            )

        except Exception:
            # Fallback with minimal info if statistics calculation fails
            if self.data is not None:
                y_var = self._parse_formula(self.formula)[0]
                y_actual = self.data[y_var].values
            else:
                y_actual = self._y

            glance_df = pd.DataFrame(
                {
                    "deviance": [None],
                    "r_squared": [None],
                    "aic": [None],
                    "aicc": [None],
                    "edf": [None],
                    "nobs": [len(y_actual)],
                }
            )

            glance_df = self._add_identifiers(glance_df)

        return glance_df


# Model registry - simple dictionary lookup
MODEL_REGISTRY = {
    "ols": OLSModel,
    "lm": OLSModel,
    "linear": OLSModel,
    "gam": GAMModel,
    "smooth": GAMModel,
}


def fit_model(
    formula: str, data: pd.DataFrame, method: str = "ols", **kwargs
) -> StatisticalModel:
    """
    Convenient function to fit statistical models with broom-like interface using formula

    Parameters:
    -----------
    formula : str
        Model formula. Examples:
        - OLS: 'y ~ x1 + x2'
        - GAM: 'y ~ x1 + s(x2)' (s() indicates smooth terms)
    data : pd.DataFrame
        Data containing the variables
    method : str
        Model type ('ols', 'lm', 'linear', 'gam', 'smooth')
    **kwargs : additional arguments passed to model fitting
        For GAM: can include family='binomial' for logistic regression, etc.

    Returns:
    --------
    StatisticalModel : Fitted model with tidy() and glance() methods

    Examples:
    ---------
    # OLS model
    model = fit_model('mpg ~ hp + wt', data=mtcars, method='ols')

    # Logistic GAM
    model = fit_model('am ~ hp + s(wt)', data=mtcars, method='gam', family='binomial')

    # Poisson regression
    model = fit_model('count ~ temp + s(time)', data=df, method='gam', family='poisson')
    """

    method = method.lower()
    if method not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported method: {method}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[method]()
    return model.fit(formula, data, **kwargs)


def fit_model_xy(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "ols",
    term_names: Optional[list] = None,
    **kwargs,
) -> StatisticalModel:
    """
    Convenient function to fit statistical models using model matrix and response

    Parameters:
    -----------
    X : np.ndarray
        Model matrix (n_samples, n_features)
    y : np.ndarray
        Response vector (n_samples,)
    method : str
        Model type ('ols', 'lm', 'linear', 'gam', 'smooth')
    term_names : list, optional
        Names for the features. If None, uses ['x0', 'x1', ...]
    **kwargs : additional arguments passed to model fitting
        For GAM: can include smooth_terms (list of feature names to smooth)

    Returns:
    --------
    StatisticalModel : Fitted model with tidy() and glance() methods

    Examples:
    ---------
    # OLS model
    model = fit_model_xy(X, y, method='ols', term_names=['hp', 'wt'])
    print(model.tidy())

    # GAM model with some smooth terms
    model = fit_model_xy(X, y, method='gam', term_names=['hp', 'wt'],
                         smooth_terms=['wt'])
    print(model.tidy())
    """

    method = method.lower()
    if method not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported method: {method}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[method]()
    return model.fit_xy(X, y, term_names=term_names, **kwargs)


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
        If DataFrame does not meet requirements
    """
    if df.empty:
        raise ValueError("Tidy DataFrame must contain at least one row")

    missing_cols = [col for col in REQUIRED_TIDY_VARS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Tidy DataFrame missing required columns: {missing_cols}")

    # Check that p-values are between 0 and 1
    if (df[TIDY_DEFS.P_VALUE] < 0).any() or (df[TIDY_DEFS.P_VALUE] > 1).any():
        raise ValueError(f"Column {TIDY_DEFS.P_VALUE} must be between 0 and 1")
