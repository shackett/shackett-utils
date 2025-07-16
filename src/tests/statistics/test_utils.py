from shackett_utils.statistics.utils import get_stat_abbreviation

from shackett_utils.statistics.constants import TIDY_DEFS, STATISTICS_DEFS


def test_get_stat_abbreviation():
    # Known keys
    assert get_stat_abbreviation(TIDY_DEFS.ESTIMATE) == "est"
    assert get_stat_abbreviation(TIDY_DEFS.STD_ERROR) == "stderr"
    assert get_stat_abbreviation(STATISTICS_DEFS.P_VALUE) == "p"
    assert get_stat_abbreviation(STATISTICS_DEFS.Q_VALUE) == "q"
    assert get_stat_abbreviation(TIDY_DEFS.LOG10_P_VALUE) == "log10_p"
    assert get_stat_abbreviation(TIDY_DEFS.STATISTIC) == "stat"
    assert get_stat_abbreviation(STATISTICS_DEFS.SIGNIFICANCE) == "sig"
    # Fallback for unknown key
    assert get_stat_abbreviation("custom_stat") == "custom"
