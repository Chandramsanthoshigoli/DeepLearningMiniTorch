from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import minitorch
from . import operators
from .autodiff import Context

if TYPE_CHECKING:
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
    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        ctx = Context(False)
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected float, got %s" % type(c)

        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return d_output, d_output


class Log(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return -d_output / (a * a)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -d_output


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        s = operators.sigmoid(a)
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (s,) = ctx.saved_values
        return d_output * s * (1 - s)


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        v = operators.exp(a)
        ctx.save_for_backward(v)
        return v

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (v,) = ctx.saved_values
        return d_output * v


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0
