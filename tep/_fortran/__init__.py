"""Fortran extension package for TEP."""
try:
    from .teprob import *
except ImportError as e:
    raise ImportError(
        f"Fortran extension not available: {e}\n"
        "Install with Fortran compiler or use backend='python'"
    ) from e
