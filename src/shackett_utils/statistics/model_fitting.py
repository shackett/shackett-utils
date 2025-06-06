import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import LinearGAM, s
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import warnings

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
    

def _calculate_residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate residual-based statistics"""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'ss_res': ss_res,
        'ss_tot': ss_tot, 
        'r_squared': r_squared,
        'residuals': residuals
    }

class StatisticalModel(ABC):
    """Abstract base class for statistical models with broom-like interface"""
    
    def __init__(self, feature_name: Optional[str] = None, model_name: Optional[str] = None):
        self.fitted_model = None
        self.formula = None
        self.data = None
        self.feature_name = feature_name  # Name of the feature being modeled
        self.model_name = model_name  # Name of the model type/variant
        self.term_names = None
        self._X = None
        self._y = None
    
    def _add_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature and model identifiers to output dataframes if present"""
        if self.feature_name is not None:
            df = df.assign(feature=self.feature_name)
        if self.model_name is not None:
            df = df.assign(model=self.model_name)
        return df
        
    @abstractmethod
    def fit(self, formula: str, data: pd.DataFrame, **kwargs) -> 'StatisticalModel':
        """Fit the model using formula and data"""
        pass
    
    @abstractmethod
    def fit_xy(self, X: np.ndarray, y: np.ndarray, term_names: Optional[list] = None, **kwargs) -> 'StatisticalModel':
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
            result['.fitted'] = self.fitted_model.fittedvalues
            result['.resid'] = self.fitted_model.resid
        else:
            # Matrix-based fitting
            result = pd.DataFrame(
                self._X, 
                columns=self.term_names[1:] if self.term_names[0] == 'const' else self.term_names
            )
            result['y'] = self._y
            result['.fitted'] = self.fitted_model.fittedvalues
            result['.resid'] = self.fitted_model.resid
        
        # Add additional diagnostics if available
        if hasattr(self.fitted_model, 'get_influence'):
            influence = self.fitted_model.get_influence()
            result['.std_resid'] = influence.resid_studentized_internal
            result['.hat'] = influence.hat_matrix_diag
            result['.cooksd'] = influence.cooks_distance[0]
        
        result = self._add_identifiers(result)

        return result

class OLSModel(StatisticalModel):
    """OLS model wrapper"""
    
    def fit(self, formula: str, data: pd.DataFrame, **kwargs) -> 'OLSModel':
        """Fit OLS model using formula"""
        self.formula = formula
        self.data = data
        self.fitted_model = smf.ols(formula, data=data).fit(**kwargs)
        return self
    
    def fit_xy(self, X: np.ndarray, y: np.ndarray, term_names: Optional[list] = None, **kwargs) -> 'OLSModel':
        """Fit OLS model using model matrix and response"""
        _validate_xy_inputs(X, y)
        
        self._X = X
        self._y = y
        self.term_names = term_names or [f'x{i}' for i in range(X.shape[1])]
        
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
        
        tidy_df = pd.DataFrame({
            'term': terms,
            'estimate': params,
            'std_error': self.fitted_model.bse,
            't_statistic': self.fitted_model.tvalues,
            'p_value': self.fitted_model.pvalues,
            'conf_low': conf_int[:, 0] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 0],
            'conf_high': conf_int[:, 1] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 1]
        })
        
        # Add feature and model name if present
        tidy_df = self._add_identifiers(tidy_df)
        
        return tidy_df.reset_index(drop=True)
    
    def glance(self) -> pd.DataFrame:
        """Return model-level statistics"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        model = self.fitted_model
        
        glance_df = pd.DataFrame({
            'r_squared': [model.rsquared],
            'adj_r_squared': [model.rsquared_adj],
            'sigma': [np.sqrt(model.mse_resid)],
            'statistic': [model.fvalue],
            'p_value': [model.f_pvalue],
            'df': [model.df_model],
            'df_residual': [model.df_resid],
            'nobs': [int(model.nobs)],
            'aic': [model.aic],
            'bic': [model.bic],
            'log_likelihood': [model.llf]
        })
        
        glance_df = self._add_identifiers(glance_df)
        
        return glance_df

class GAMModel(StatisticalModel):
    """GAM model wrapper using pygam"""
    
    def __init__(self, feature_name: Optional[str] = None, model_name: Optional[str] = None):
        super().__init__(feature_name=feature_name, model_name=model_name)
        self.smooth_terms = []
        self.family = None
    
    def _parse_formula(self, formula: str) -> tuple:
        """Parse formula string to extract dependent and independent variables"""
        parts = formula.split('~')
        if len(parts) != 2:
            raise ValueError("Formula must be in format 'y ~ x1 + x2 + ...'")
        
        y_var = parts[0].strip()
        x_part = parts[1].strip()
        
        # Parse terms - handle both linear terms and smooth terms like s(x1)
        raw_terms = [term.strip() for term in x_part.split('+')]
        
        x_vars = []
        smooth_terms = []
        
        for term in raw_terms:
            # Check if it's a smooth term like s(variable_name)
            if term.startswith('s(') and term.endswith(')'):
                # Extract variable name from s(variable_name)
                var_name = term[2:-1].strip()
                x_vars.append(var_name)
                smooth_terms.append(var_name)
            else:
                # Regular linear term
                x_vars.append(term)
        
        return y_var, x_vars, smooth_terms
    
    def _build_gam_terms(self, x_vars: list, smooth_terms: list) -> list:
        """Build GAM terms list for model fitting"""
        from pygam.terms import TermList, l, s
        
        terms = []
        for i, var in enumerate(x_vars):
            if var in smooth_terms:
                terms.append(s(i))
            else:
                terms.append(l(i))
        return TermList(*terms)
    
    def fit(self, formula: str, data: pd.DataFrame, family: str = 'gaussian', **kwargs) -> 'GAMModel':
        """Fit GAM model using formula
        
        Parameters
        ----------
        formula : str
            Model formula (e.g. 'y ~ x1 + s(x2)')
        data : pd.DataFrame
            Data containing the variables
        family : str
            Distribution family for the response variable:
            - 'gaussian' : Normal distribution (default)
            - 'binomial' : Logistic regression
            - 'poisson' : Poisson regression
            - 'gamma' : Gamma regression
        **kwargs : additional arguments passed to LinearGAM
        """
        from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM
        
        # Map family names to GAM classes
        family_map = {
            'gaussian': LinearGAM,
            'binomial': LogisticGAM,
            'poisson': PoissonGAM,
            'gamma': GammaGAM
        }
        
        if family not in family_map:
            raise ValueError(f"Unsupported family: {family}. Available: {list(family_map.keys())}")
        
        self.formula = formula
        self.data = data
        self.family = family
        
        # Parse formula
        y_var, x_vars, smooth_terms = self._parse_formula(formula)
        self.term_names = x_vars
        self.smooth_terms = smooth_terms
        
        # Extract X and y
        y = data[y_var].values
        X = data[x_vars].values
        
        # Build GAM terms
        terms = self._build_gam_terms(x_vars, smooth_terms)
        
        try:
            # Use the appropriate GAM class for the family
            GAMClass = family_map[family]
            self.fitted_model = GAMClass(terms, **kwargs).fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Failed to fit GAM model: {str(e)}")
        
        return self
    
    def fit_xy(self, X: np.ndarray, y: np.ndarray, term_names: Optional[list] = None, **kwargs) -> 'GAMModel':
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
                    edf = model.statistics_['edof_per_coef'][i] if hasattr(model, 'statistics_') else None
                    p_value = model.statistics_['p_values'][i] if hasattr(model, 'statistics_') else None
                except (KeyError, IndexError, AttributeError):
                    edf = None
                    p_value = None
                
                terms_info.append({
                    'term': f's({feature_name})',
                    'type': 'smooth',
                    'edf': edf,
                    'p_value': p_value
                })
            else:
                # For linear terms, try to get coefficient
                try:
                    coef = model.coef_[i] if hasattr(model, 'coef_') else None
                except (IndexError, AttributeError):
                    coef = None
                
                terms_info.append({
                    'term': feature_name,
                    'type': 'linear',
                    'estimate': coef,
                    'edf': 1.0
                })
        
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
                deviance = model.statistics_['deviance'] if hasattr(model, 'statistics_') else stats['ss_res']
                aic = model.statistics_['AIC'] if hasattr(model, 'statistics_') else None
                aicc = model.statistics_['AICc'] if hasattr(model, 'statistics_') else None
                edf = model.statistics_['edof'] if hasattr(model, 'statistics_') else None
            except (KeyError, AttributeError):
                deviance = stats['ss_res']
                aic = None
                aicc = None
                edf = None
            
            glance_df = pd.DataFrame({
                'deviance': [deviance],
                'r_squared': [stats['r_squared']],
                'aic': [aic],
                'aicc': [aicc],
                'edf': [edf],
                'nobs': [len(y_actual)]
            })
            
        except Exception as e:
            # Fallback with minimal info if statistics calculation fails
            if self.data is not None:
                y_var = self._parse_formula(self.formula)[0]
                y_actual = self.data[y_var].values
            else:
                y_actual = self._y
                
            glance_df = pd.DataFrame({
                'deviance': [None],
                'r_squared': [None],
                'aic': [None],
                'aicc': [None],
                'edf': [None],
                'nobs': [len(y_actual)]
            })

            glance_df = self._add_identifiers(glance_df)
        
        return glance_df

# Model registry - simple dictionary lookup
MODEL_REGISTRY = {
    'ols': OLSModel,
    'lm': OLSModel,
    'linear': OLSModel,
    'gam': GAMModel,
    'smooth': GAMModel
}

def fit_model(formula: str, data: pd.DataFrame, method: str = 'ols', **kwargs) -> StatisticalModel:
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
        raise ValueError(f"Unsupported method: {method}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model = MODEL_REGISTRY[method]()
    return model.fit(formula, data, **kwargs)

def fit_model_xy(X: np.ndarray, y: np.ndarray, method: str = 'ols', 
                 term_names: Optional[list] = None, **kwargs) -> StatisticalModel:
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
        raise ValueError(f"Unsupported method: {method}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model = MODEL_REGISTRY[method]()
    return model.fit_xy(X, y, term_names=term_names, **kwargs)