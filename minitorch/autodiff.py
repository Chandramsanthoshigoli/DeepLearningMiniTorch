from typing import Any, Callable, Optional

class Variable:
    def __init__(self, data: Any, history: Optional[Any] = None):
        self.data = data
        self.history = history
        self._derivative = 0.0

    def derivative(self) -> float:
        return self._derivative

def central_difference(f: Callable[[Any], float], inputs: list[Any], arg: int, epsilon: float = 1e-6) -> float:
    x = inputs.copy()
    x[arg] += epsilon
    upper = f(x)
    x[arg] -= 2 * epsilon
    lower = f(x)
    return (upper - lower) / (2 * epsilon)
