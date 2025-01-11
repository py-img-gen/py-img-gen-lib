import warnings

from py_img_gen.utils import suppress_warnings


def test_suppress_warnings():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Execute the function to be tested
        suppress_warnings()

        # Raise a FutureWarning that is the target of suppression
        warnings.warn("This is a warning", FutureWarning)

        # Check that the warning was suppressed
        assert len(w) == 0
