import sys
import math
import struct
from typing import Tuple

import numpy as np
import cv2
from numba import cuda, float32, int32

# Constants
MAX_ERROR = 1e30


@cuda.jit(device=True)
def transform_index(x: int, y: int, n: int, t: int) -> int:
    """
    Map local block coordinates (x,y) under one of 8 isometries.
    """
    if t == 0:
        return y * n + x
    if t == 1:
        return (n - 1 - y) * n + x
    if t == 2:
        return y * n + (n - 1 - x)
    if t == 3:
        return x * n + y
    if t == 4:
        return (n - 1 - x) * n + y
    if t == 5:
        return x * n + (n - 1 - y)
    if t == 6:
        return (n - 1 - y) * n + (n - 1 - x)
    if t == 7:
        return (n - 1 - x) * n + (n - 1 - y)
    return y * n + x


@cuda.jit
def compress_kernel(
    img: cuda.device_array, width: int, height: int, block_size: int,
    dom_x: cuda.device_array, dom_y: cuda.device_array, dom_count: int,
    out_dx: cuda.device_array, out_dy: cuda.device_array,
    out_t: cuda.device_array, out_a: cuda.device_array, out_b: cuda.device_array
):
    """
    For each range-block (one block per CUDA blockIdx.x), test all domain-blocks
    under each isometry, compute least-squares fit (a,b), pick minimal error.
    """
    range_idx = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x

    blocks_x = width // block_size
    ry = (range_idx // blocks_x) * block_size
    rx = (range_idx % blocks_x) * block_size

    # shared memory: [best_error, dx, dy, t, a, b]
    shmem = cuda.shared.array(6, dtype=float32)
    if thread_id == 0:
        shmem[0] = MAX_ERROR  # best_error
    cuda.syncthreads()

    domain_index = thread_id // 8
    transform_type = thread_id % 8

    if domain_index < dom_count:
        dx = dom_x[domain_index]
        dy = dom_y[domain_index]

        # accumulate sums for least-squares
        sum_d = float32(0.0)
        sum_r = float32(0.0)
        sum_dd = float32(0.0)
        sum_dr = float32(0.0)

        for yy in range(block_size):
            for xx in range(block_size):
                ti = transform_index(xx, yy, block_size, transform_type)
                dy_off = dy + (ti // block_size)
                dx_off = dx + (ti % block_size)
                d_val = img[dy_off, dx_off]
                r_val = img[ry + yy, rx + xx]
                sum_d += d_val
                sum_r += r_val
                sum_dd += d_val * d_val
                sum_dr += d_val * r_val

        denom = block_size * block_size * sum_dd - sum_d * sum_d
        a = (block_size * block_size * sum_dr - sum_d * sum_r) / denom if denom != 0 else 1.0
        b = (sum_r - a * sum_d) / (block_size * block_size)

        # compute prediction error
        err = float32(0.0)
        for yy in range(block_size):
            for xx in range(block_size):
                ti = transform_index(xx, yy, block_size, transform_type)
                dy_off = dy + (ti // block_size)
                dx_off = dx + (ti % block_size)
                dt = img[dy_off, dx_off]
                prediction = a * dt + b
                diff = img[ry + yy, rx + xx] - prediction
                err += diff * diff

        # atomic update of best
        cuda.atomic.min(shmem, 0, err)
        cuda.syncthreads()

        if err <= shmem[0]:
            # thread 0 writes shared params
            if thread_id == 0:
                shmem[1] = dx
                shmem[2] = dy
                shmem[3] = transform_type
                shmem[4] = a
                shmem[5] = b
            cuda.syncthreads()

            out_dx[range_idx] = int32(shmem[1])
            out_dy[range_idx] = int32(shmem[2])
            out_t[range_idx] = int32(shmem[3])
            out_a[range_idx] = shmem[4]
            out_b[range_idx] = shmem[5]


def load_grayscale_image(path: str) -> np.ndarray:
    """Load image as grayscale numpy array."""
    img = cv2.imread(
