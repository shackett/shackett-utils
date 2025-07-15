from types import SimpleNamespace

STATISTICS_DEFS = SimpleNamespace(
    FDR_METHOD="fdr_method",
    FEATURE_NAME="feature_name",
    FEATURE_NAMES="feature_names",
    MODEL_NAME="model_name",
    TERM="term",
    P_VALUE="p_value",
    Q_VALUE="q_value",
    SIGNIFICANCE="significance",
)

TIDY_DEFS = SimpleNamespace(
    TERM=STATISTICS_DEFS.TERM,
    ESTIMATE="estimate",
    STD_ERROR="std_error",
    STATISTIC="statistic",
    P_VALUE=STATISTICS_DEFS.P_VALUE,
    CONF_LOW="conf_low",
    CONF_HIGH="conf_high",
)

POSSIBLE_TIDY_VARS = [
    TIDY_DEFS.TERM,
    TIDY_DEFS.ESTIMATE,
    TIDY_DEFS.STD_ERROR,
    TIDY_DEFS.STATISTIC,
    TIDY_DEFS.P_VALUE,
    TIDY_DEFS.CONF_LOW,
    TIDY_DEFS.CONF_HIGH,
]

REQUIRED_TIDY_VARS = [TIDY_DEFS.TERM, TIDY_DEFS.ESTIMATE, TIDY_DEFS.P_VALUE]

# Map statistics to their column prefixes
STATISTICS_ABBREVIATIONS = {
    TIDY_DEFS.ESTIMATE: "est",
    TIDY_DEFS.STD_ERROR: "stderr",
    STATISTICS_DEFS.P_VALUE: "p",
    STATISTICS_DEFS.Q_VALUE: "q",
    TIDY_DEFS.STATISTIC: "stat",
    STATISTICS_DEFS.SIGNIFICANCE: "sig",
}

GLANCE_DEFS = SimpleNamespace(
    R_SQUARED="r_squared",
    ADJUSTED_R_SQUARED="adj_r_squared",
    SIGMA="sigma",
    STATISTIC="statistic",
    P_VALUE=STATISTICS_DEFS.P_VALUE,
    DF="df",
    DF_RESIDUAL="df_residual",
    NOBS="nobs",
    AIC="aic",
    BIC="bic",
    LOG_LIKELIHOOD="log_likelihood",
    DEVIANCE="deviance",
    AICC="aicc",
    EDF="edf",
)

REQUIRED_GLANCE_VARS = [GLANCE_DEFS.R_SQUARED, GLANCE_DEFS.AIC, GLANCE_DEFS.NOBS]

AUGMENT_DEFS = SimpleNamespace(
    FITTED=".fitted",
    RESIDUAL=".resid",
    STD_RESID=".std_resid",
    HAT=".hat",
    COOKSD=".cooksd",
)

REQUIRED_AUGMENT_VARS = [
    AUGMENT_DEFS.FITTED,
    AUGMENT_DEFS.RESIDUAL,
]

# hyothesis testing

HYPOTHESIS_TESTING_DEFS = SimpleNamespace(
    TWO_TAILED="two-tailed",
    ONE_TAILED_UPPER="one-tailed-upper",
    ONE_TAILED_LOWER="one-tailed-lower",
)

HYPOTHESIS_TESTING_TYPES = []


# FDR control

MULTITEST_GROUPING_VARS = [
    STATISTICS_DEFS.TERM,
    STATISTICS_DEFS.MODEL_NAME,
]

FDR_METHODS_DEFS = SimpleNamespace(BH="fdr_bh")

FDR_METHODS = [
    FDR_METHODS_DEFS.BH,
]
