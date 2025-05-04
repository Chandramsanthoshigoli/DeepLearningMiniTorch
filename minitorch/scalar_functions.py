from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and tracks scalar values.
    """

    @staticmethod
    def forward(ctx: Context, *inputs: ScalarLike) -> ScalarLike:
        raise NotImplementedError("Forward not implemented")

    @staticmethod
    def backward(ctx: Context, d_output: ScalarLike) -> Tuple[ScalarLike, ...]:
        raise NotImplementedError("Backward not implemented")


# Backward functions for composed functions

def log_back(x: float, d: float) -> float:
    return d / x


def inv_back(x: float, d: float) -> float:
    return -1 * d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0
