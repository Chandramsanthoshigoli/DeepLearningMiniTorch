from typing import List
from minitorch.scalar import Scalar

class Parameter:
    def __init__(self, value: float):
        self.value = value

    def __mul__(self, other: Any) -> float:
        return self.value * other

class Module:
    def __init__(self):
        self.params = []

    def add_parameter(self, p: Parameter) -> None:
        self.params.append(p)

    def parameters(self) -> List[Parameter]:
        return self.params

def scalar_function(x: Scalar, p: Parameter) -> Scalar:
    return x * p.value
