"""Utilities for the statistics subpackage."""

from typing import List

from shackett_utils.statistics.constants import (
    STATISTICS_ABBREVIATIONS,
    STATISTICAL_SUMMARIES,
)

VALID_REGRESSION_OUTPUT_TYPES = frozenset(
    {
        STATISTICAL_SUMMARIES.TIDY,
        STATISTICAL_SUMMARIES.GLANCE,
        STATISTICAL_SUMMARIES.AUGMENT,
    }
)


def get_stat_abbreviation(stat: str) -> str:
    """Return the abbreviation for a statistic, or a default based on the stat name."""
    return STATISTICS_ABBREVIATIONS.get(stat, stat.split("_")[0].lower())


def validate_regression_output_types(outputs: List[str]) -> None:
    """
    Validate that outputs is a non-empty list of valid summary types.

    Parameters
    ----------
    outputs : list of str
        Requested output types (tidy, glance, augment).

    Raises
    ------
    ValueError
        If outputs is empty or contains invalid entries.
    """
    if not outputs:
        raise ValueError("outputs must contain at least one entry")
    invalid = [o for o in outputs if o not in VALID_REGRESSION_OUTPUT_TYPES]
    if invalid:
        raise ValueError(
            f"Invalid output(s): {invalid}. "
            f"Must be one or more of: {sorted(VALID_REGRESSION_OUTPUT_TYPES)}"
        )
