from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Type
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

class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]):
        def _map(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            tensor_map(fn)(*out.tuple(), *a.tuple())
            return out
        return _map

    @staticmethod
    def zip(fn: Callable[[float, float], float]):
        def _zip(a: Tensor, b: Tensor) -> Tensor:
            shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(shape)
            tensor_zip(fn)(*out.tuple(), *a.tuple(), *b.tuple())
            return out
        return _zip

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float = 0.0):
        def _reduce(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            tensor_reduce(fn)(*out.tuple(), *a.tuple(), dim)
            return out
        return _reduce

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


def tensor_map(fn: Callable[[float], float]):
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        for i in range(len(out)):
            out_index = [0] * len(out_shape)
            to_index(i, out_shape, out_index)
            in_index = broadcast_index(out_index, out_shape, in_shape)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])
    return _map


def tensor_zip(fn: Callable[[float, float], float]):
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for i in range(len(out)):
            out_index = [0] * len(out_shape)
            to_index(i, out_shape, out_index)
            a_index = broadcast_index(out_index, out_shape, a_shape)
            b_index = broadcast_index(out_index, out_shape, b_shape)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
    return _zip


def tensor_reduce(fn: Callable[[float, float], float]):
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = [0] * len(out_shape)
        a_index = [0] * len(a_shape)

        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            for j in range(len(a_shape)):
                a_index[j] = out_index[j]
            out_pos = index_to_position(out_index, out_strides)
            result = out[out_pos]

            for j in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index, a_strides)
                result = fn(result, a_storage[a_pos])
            out[out_pos] = result
    return _reduce


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


SimpleBackend = TensorBackend(TensorOps)
