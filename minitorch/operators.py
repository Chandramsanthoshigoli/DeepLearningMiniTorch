"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable


# ## Task 0.1
#
# Implementation of a prelude of elementary functions.

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
    "$f(x, y) = x < y$"
    return float(x < y)


def eq(x: float, y: float) -> float:
    "$f(x, y) = x == y$"
    return float(x == y)


def max(x: float, y: float) -> float:
    "$f(x, y) = max(x, y)$"
    return x if x > y else y


# ## Task 0.2
# Implement sigmoid and relu below

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    return x if x > 0 else 0.0


# ## Task 0.3
# Higher-order functions

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def inner(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return inner


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def inner(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return inner


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    def inner(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return inner
