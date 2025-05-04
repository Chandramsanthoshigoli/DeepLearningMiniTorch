from hypothesis import settings
from hypothesis.strategies import floats, integers

import minitorch

__all__ = ["small_ints", "med_ints", "small_floats", "assert_close"]

# Register profile for continuous integration runs without timeouts
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# Strategy Definitions
small_ints = integers(min_value=1, max_value=3)
med_ints = integers(min_value=1, max_value=20)

# Only valid floats (exclude NaN / inf)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)


def assert_close(a: float, b: float) -> None:
    """
    Asserts that two floats are close using minitorch's is_close function.
    """
    if not minitorch.operators.is_close(a, b):
        raise AssertionError(f"Failure: a={a:.6f} vs b={b:.6f}")
