from types import SimpleNamespace

STATISTICS_DEFS = SimpleNamespace(
    FDR_METHOD = "fdr_method",
    FEATURE_NAME = "feature_name",
    FEATURE_NAMES = "feature_names",
    MODEL_NAME = "model_name",
    TERM = "term",
    P_VALUE = "p_value",
    Q_VALUE = "q_value",    
)

TIDY_DEFS = SimpleNamespace(
    TERM = STATISTICS_DEFS.TERM,
    ESTIMATE = "estimate",
    STD_ERROR = "std_error",
    STATISTIC = "statistic",
    P_VALUE = STATISTICS_DEFS.P_VALUE,
    CONF_LOW = "conf_low",
    CONF_HIGH = "conf_high",
)

REQUIRED_TIDY_VARS = [
    TIDY_DEFS.TERM,
    TIDY_DEFS.ESTIMATE,
    TIDY_DEFS.P_VALUE
]

GLANCE_DEFS = SimpleNamespace(
    R_SQUARED = "r_squared",
    ADJUSTED_R_SQUARED = "adj_r_squared",
    SIGMA = "sigma",
    STATISTIC = "statistic",
    P_VALUE = STATISTICS_DEFS.P_VALUE,
    DF = "df",
    DF_RESIDUAL = "df_residual",
    NOBS = "nobs",
    AIC = "aic",
    BIC = "bic",
    LOG_LIKELIHOOD = "log_likelihood",
    DEVIANCE = "deviance",
    AICC = "aicc",
    EDF = "edf",
)

REQUIRED_GLANCE_VARS = [
    GLANCE_DEFS.R_SQUARED,
    GLANCE_DEFS.AIC,
    GLANCE_DEFS.NOBS
]

AUGMENT_DEFS = SimpleNamespace(
    FITTED = ".fitted",
    RESIDUAL = ".resid",
    STD_RESID = ".std_resid",
    HAT = ".hat",
    COOKSD = ".cooksd",
)

REQUIRED_AUGMENT_VARS = [
    AUGMENT_DEFS.FITTED,
    AUGMENT_DEFS.RESIDUAL,
]

MULTITEST_GROUPING_VARS = [
    STATISTICS_DEFS.TERM,
    STATISTICS_DEFS.MODEL_NAME,
]

FDR_METHODS_DEFS = SimpleNamespace(
    FDR_BH = "fdr_bh"
)

FDR_METHODS = [
    FDR_METHODS_DEFS.FDR_BH,
]