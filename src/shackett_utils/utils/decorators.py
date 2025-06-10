"""
Utility decorators for the shackett_utils package.
"""

import functools
import io
import sys
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def suppress_stdout(func: Callable[P, T]) -> Callable[P, T]:
    """
    A decorator that suppresses stdout output from a function.
    Particularly useful for suppressing MOFA's verbose output during testing.
    
    Parameters
    ----------
    func : Callable
        The function to wrap
        
    Returns
    -------
    Callable
        The wrapped function that will execute with suppressed stdout
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Store the original stdout
        stdout = sys.stdout
        # Redirect stdout to a dummy stream
        sys.stdout = io.StringIO()
        try:
            # Run the function
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore stdout
            sys.stdout = stdout
    return wrapper 