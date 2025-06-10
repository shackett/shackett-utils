"""
Personal utility functions for blog posts and applications
"""

import warnings

# Filter out known dependency warnings
warnings.filterwarnings(
    "ignore",
    message="Transforming to str index",
    category=UserWarning,
    module="anndata._core.aligned_df"
)

warnings.filterwarnings(
    "ignore",
    message="np.find_common_type is deprecated",
    category=DeprecationWarning,
    module="pandas.core.algorithms"
)

__version__ = "0.1.0" 