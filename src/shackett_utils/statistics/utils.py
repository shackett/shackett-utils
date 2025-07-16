from shackett_utils.statistics.constants import STATISTICS_ABBREVIATIONS


def get_stat_abbreviation(stat: str) -> str:
    """Return the abbreviation for a statistic, or a default based on the stat name."""
    return STATISTICS_ABBREVIATIONS.get(stat, stat.split("_")[0].lower())
