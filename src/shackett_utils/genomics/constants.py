"""Constants for genomics utilities."""

from typing import List

from shackett_utils.statistics.constants import TIDY_DEFS, STATISTICS_DEFS

# Available regression statistics that can be spread to adata.var
REGRESSION_DEFAULT_STATS: List[str] = [
    TIDY_DEFS.ESTIMATE,  # Effect size estimates
    STATISTICS_DEFS.P_VALUE,    # Raw p-values
    STATISTICS_DEFS.Q_VALUE,    # FDR-corrected p-values
    TIDY_DEFS.STATISTIC,   # T-statistics
    TIDY_DEFS.STD_ERROR   # Standard errors
] 