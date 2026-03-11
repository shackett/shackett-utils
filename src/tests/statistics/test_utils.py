import pytest

from shackett_utils.statistics.utils import (
    get_stat_abbreviation,
    validate_regression_output_types,
)
from shackett_utils.statistics.constants import (
    STATISTICAL_SUMMARIES,
    STATISTICS_DEFS,
    TIDY_DEFS,
)


def test_get_stat_abbreviation():
    # Known keys
    assert get_stat_abbreviation(TIDY_DEFS.ESTIMATE) == "est"
    assert get_stat_abbreviation(TIDY_DEFS.STD_ERROR) == "stderr"
    assert get_stat_abbreviation(STATISTICS_DEFS.P_VALUE) == "p"
    assert get_stat_abbreviation(STATISTICS_DEFS.Q_VALUE) == "q"
    assert get_stat_abbreviation(TIDY_DEFS.LOG10_P_VALUE) == "log10p"
    assert get_stat_abbreviation(TIDY_DEFS.STATISTIC) == "stat"
    assert get_stat_abbreviation(STATISTICS_DEFS.SIGNIFICANCE) == "sig"
    # Fallback for unknown key
    assert get_stat_abbreviation("custom_stat") == "custom"


def test_validate_outputs():
    """Test validate_outputs checks 1+ entries and valid summary types."""
    # Valid: single entry
    validate_regression_output_types([STATISTICAL_SUMMARIES.TIDY])

    # Valid: multiple entries
    validate_regression_output_types(
        [STATISTICAL_SUMMARIES.TIDY, STATISTICAL_SUMMARIES.GLANCE]
    )
    validate_regression_output_types(
        [
            STATISTICAL_SUMMARIES.TIDY,
            STATISTICAL_SUMMARIES.GLANCE,
            STATISTICAL_SUMMARIES.AUGMENT,
        ]
    )

    # Invalid: empty list
    with pytest.raises(ValueError, match="at least one entry"):
        validate_regression_output_types([])

    # Invalid: unknown output type
    with pytest.raises(ValueError, match="Invalid output"):
        validate_regression_output_types(["tidy", "unknown"])
