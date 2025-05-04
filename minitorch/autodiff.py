from .autodiff import Context
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, List


from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation

def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the arg to differentiate against
        epsilon : the small shift for computing the finite difference

    Returns:
        Approximate derivative with respect to `arg`
    """
    vals1 = list(vals)
    vals2 = list(vals)
    vals1[arg] += epsilon
    vals2[arg] -= epsilon
    return (f(*vals1) - f(*vals2)) / (2 * epsilon)


# ## Task 1.3
# Variable class

class Variable:
    def __init__(self, data: float) -> None:
        self.data = data
        self.derivative = 0.0
        self.name = ""
        self.parents = []

    def history(self) -> Context:
        return Context()

    def chain_rule(self, out_derivative) -> List[Tuple["Variable", float]]:
        return [(p, out_derivative) for p in self.parents]
