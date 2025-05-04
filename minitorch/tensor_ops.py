from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        def _map(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
            # Implementation would go here
            pass
        return _map

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]:
        def _zip(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
            # Implementation would go here
            pass
        return _zip

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float],
        start: float = 0.0,
    ) -> Callable[[Tensor, int], Tensor]:
        def _reduce(a: Tensor, dim: int) -> Tensor:
            # Implementation would go here
            pass
        return _reduce
