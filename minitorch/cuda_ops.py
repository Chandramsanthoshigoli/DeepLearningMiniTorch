from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import TensorOps

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> Callable:
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float = 0.0) -> Callable:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(fn: Callable[[float], float]) -> Callable:
    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])
    return cuda.jit()(_map)


def tensor_zip(fn: Callable[[float, float], float]) -> Callable:
    def _zip(out, out_shape, out_strides, out_size,
             a_storage, a_shape, a_strides,
             b_storage, b_shape, b_strides):
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
    return cuda.jit()(_zip)


def tensor_reduce(fn: Callable[[float, float], float]) -> Callable:
    def _reduce(out, out_shape, out_strides, out_size,
                a_storage, a_shape, a_strides,
                reduce_dim, reduce_value):
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            a_index = out_index.copy()
            start = reduce_value

            for j in range(pos, a_shape[reduce_dim], BLOCK_DIM):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index, a_strides)
                start = fn(start, a_storage[a_pos])

            cache[pos] = start
            cuda.syncthreads()

            if pos == 0:
                result = cache[0]
                for j in range(1, BLOCK_DIM):
                    result = fn(result, cache[j])
                out_index = cuda.local.array(MAX_DIMS, numba.int32)
                to_index(out_pos, out_shape, out_index)
                out_pos_final = index_to_position(out_index, out_strides)
                out[out_pos_final] = result
    return cuda.jit()(_reduce)


def _tensor_matrix_multiply(out, out_shape, out_strides, out_size,
                             a_storage, a_shape, a_strides,
                             b_storage, b_shape, b_strides):
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row >= out_shape[1] or col >= out_shape[2]:
        return

    value = 0.0
    for k in range((a_shape[2] + BLOCK_DIM - 1) // BLOCK_DIM):
        a_i = row
        a_j = k * BLOCK_DIM + cuda.threadIdx.y
        if a_i < a_shape[1] and a_j < a_shape[2]:
            a_index = [batch, a_i, a_j]
            a_shared[cuda.threadIdx.x][cuda.threadIdx.y] = a_storage[index_to_position(a_index, a_strides)]
        else:
            a_shared[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

        b_i = k * BLOCK_DIM + cuda.threadIdx.x
        b_j = col
        if b_i < b_shape[1] and b_j < b_shape[2]:
            b_index = [batch, b_i, b_j]
            b_shared[cuda.threadIdx.x][cuda.threadIdx.y] = b_storage[index_to_position(b_index, b_strides)]
        else:
            b_shared[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

        cuda.syncthreads()
        for n in range(BLOCK_DIM):
            value += a_shared[cuda.threadIdx.x][n] * b_shared[n][cuda.threadIdx.y]
        cuda.syncthreads()

    out_index = [batch, row, col]
    if row < out_shape[1] and col < out_shape[2]:
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = value


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
