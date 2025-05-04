from typing import Callable

def identity(x: float) -> float:
    return x

def sigmoid(x: float) -> float:
    import math
    return 1 / (1 + math.exp(-x))

def log(x: float) -> float:
    import math
    return math.log(x)

def inv(x: float) -> float:
    return 1.0 / x

def relu(x: float) -> float:
    return x if x > 0 else 0.0
