"""Constants for genomics utilities."""

from typing import List
from types import SimpleNamespace

from shackett_utils.statistics.constants import TIDY_DEFS, STATISTICS_DEFS

# Available regression statistics that can be spread to adata.var
REGRESSION_DEFAULT_STATS: List[str] = [
    TIDY_DEFS.ESTIMATE,  # Effect size estimates
    STATISTICS_DEFS.P_VALUE,  # Raw p-values
    STATISTICS_DEFS.Q_VALUE,  # FDR-corrected p-values
    TIDY_DEFS.STATISTIC,  # T-statistics
    TIDY_DEFS.STD_ERROR,  # Standard errors
]

MOFA_DEFS = SimpleNamespace(FACTOR_NAME="factor_name", X_MOFA="X_mofa", LFS="LFs")

VARIANCE_METRICS_DEFS = SimpleNamespace(
    TOTAL_VARIANCE="total_variance",
    MODALITY_VARIANCE="modality_variance",
    RAW_TSS="raw_tss",
    RAW_ESS="raw_ess",
)

K_SELECTION_CRITERIA = SimpleNamespace(
    ELBOW="elbow",
    THRESHOLD="threshold",
    BALANCED="balanced",
    USER_DEFINED="user_defined",
)

AUTO_K_SELECTION_CRITERIA = [
    K_SELECTION_CRITERIA.ELBOW,
    K_SELECTION_CRITERIA.THRESHOLD,
    K_SELECTION_CRITERIA.BALANCED,
]

# Key for storing MOFA regression results in uns (deprecated)
FACTOR_REGRESSION_STR = "mofa_regression_{}"

# Pattern for factor names (e.g. Factor_1, Factor_2, etc.)
FACTOR_NAME_PATTERN = "Factor_{}"

# Key for storing unified factor regression results
FACTOR_REGRESSIONS_KEY = "factor_regressions"
