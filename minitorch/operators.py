"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable, List

# ## Task 0.1


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    return -x


def lt(x: float, y: float) -> float:
    "$f(x) = 1.0$ if x is less than y else $0.0$"
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    "$f(x) = 1.0$ if x is equal to y else $0.0$"
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    "$f(x) = x$ if x is greater than y else $y$"
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    For stability:
    if x >= 0: 1 / (1 + exp(-x))
    else: exp(x) / (1 + exp(x))
    """
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """
    $f(x) = x$ if x > 0, else 0
    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f(x) = log(x)$, return $d \cdot f'(x) = d/x$"
    return d / x


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$, return $-d / x^2$"
    return -d / (x ** 2)


def relu_back(x: float, d: float) -> float:
    r"If $f(x) = relu(x)$, return $d$ if $x > 0$ else $0$"
    return d if x > 0 else 0


# ## Task 0.3


def map_fn(fn: Callable[[float], float]) -> Callable[[Iterable[float]], List[float]]:
    """
    Higher-order map. Applies `fn` to each element in a list.
    """
    def apply(my_list: Iterable[float]) -> List[float]:
        return [fn(x) for x in my_list]
    return apply


def negList(ls: Iterable[float]) -> List[float]:
    "Use `map_fn` and `neg` to negate each element in `ls`"
    return map_fn(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], List[float]]:
    """
    Higher-order zipWith (like map2). Applies `fn` to zipped elements of two lists.
    """
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    "Add elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """
    Higher-order reduce. Reduces a list using `fn` starting from `start`.
    """
    def apply(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return apply


def sum(ls: Iterable[float]) -> float:
    "Sum a list using `reduce` and `add`."
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Multiply elements of a list using `reduce` and `mul`."
    return reduce(mul, 1)(ls)
