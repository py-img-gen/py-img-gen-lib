import warnings


def suppress_warnings() -> None:
    r"""Suppress target warnings."""
    warnings.simplefilter(action="ignore", category=FutureWarning)
