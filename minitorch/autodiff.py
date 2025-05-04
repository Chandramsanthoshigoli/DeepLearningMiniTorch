from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


def central_difference(f: Any, *vals: float, arg: int = 0, epsilon: float = 1e-6) -> float:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_list = list(vals)
    vals_list[arg] += epsilon
    f_plus = f(*vals_list)
    vals_list[arg] -= 2 * epsilon
    f_minus = f(*vals_list)
    return (f_plus - f_minus) / (2 * epsilon)


class Variable(Protocol):
    derivative: Any

    def accumulate_derivative(self, x: Any) -> None:
        ...

    @property
    def unique_id(self) -> int:
        ...

    def is_leaf(self) -> bool:
        ...

    def is_constant(self) -> bool:
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    visited: set[int] = set()
    order: list[Variable] = []

    def dfs(v: Variable) -> None:
        if v.is_constant() or v.unique_id in visited:
            return
        visited.add(v.unique_id)
        for p in v.parents:
            dfs(p)
        order.append(v)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    variable.derivative = deriv
    topo = list(topological_sort(variable))

    for v in reversed(topo):
        if v.is_constant() or v.is_leaf():
            continue
        for parent, grad in v.chain_rule(v.derivative):
            if getattr(parent, "derivative", None) is None:
                parent.derivative = grad
            else:
                parent.derivative += grad


@dataclass
class Context:
    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        if not self.no_grad:
            self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
