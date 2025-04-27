import sys
import numpy as np
import cv2
from numba import cuda, float32, int32
import math
import struct

# device function: map (x,y) under 8 isometries
@cuda.jit(device=True)
def transform_index(x, y, N, t):
    if t == 0:
        return y * N + x
    elif t == 1:
        return (N - 1 - y) * N + x
    elif t == 2:
        return y * N + (N - 1 - x)
    elif t == 3:
        return x * N + y
    elif t == 4:
        return (N - 1 - x) * N + y
    elif t == 5:
        return x * N + (N - 1 - y)
    elif t == 6:
        return (N - 1 - y) * N + (N - 1 - x)
    elif t == 7:
        return (N - 1 - x) * N + (N - 1 - y)
    else:
        return y * N + x

# CUDA kernel
@cuda.jit
def compress_kernel(img, W, H, N, domX, domY, domCount, out_dx, out_dy, out_t, out_a, out_b):
    range_idx = cuda.blockIdx.x
    tx = cuda.threadIdx.x

    blocks_x = W // N
    ry = (range_idx // blocks_x) * N
    rx = (range_idx % blocks_x) * N

    # shared memory for best error + params
    smem = cuda.shared.array(1 + 5, dtype=float32)  # [bestErr, dx, dy, t, a, b]
    if tx == 0:
        smem[0] = 1e30  # bestErr
    cuda.syncthreads()

    di = tx // 8
    t  = tx % 8

    if di < domCount:
        dx = domX[di]
        dy = domY[di]

        # compute sums
        sumD = float32(0.0)
        sumR = float32(0.0)
        sumDD = float32(0.0)
        sumDR = float32(0.0)
        for yy in range(N):
            for xx in range(N):
                # domain pixel (with isometry)
                ti = transform_index(xx, yy, N, t)
                dy_off = dy + (ti // N)
                dx_off = dx + (ti % N)
                d = img[dy_off, dx_off]
                r = img[ry + yy, rx + xx]
                sumD += d
                sumR += r
                sumDD += d * d
                sumDR += d * r

        denom = N*N*sumDD - sumD*sumD
        a = (N*N*sumDR - sumD*sumR) / denom if denom != 0 else 1.0
        b = (sumR - a*sumD) / (N*N)

        # compute error
        err = float32(0.0)
        for yy in range(N):
            for xx in range(N):
                ti = transform_index(xx, yy, N, t)
                dy_off = dy + (ti // N)
                dx_off = dx + (ti % N)
                Dt = img[dy_off, dx_off]
                pred = a * Dt + b
                diff = img[ry+yy, rx+xx] - pred
                err += diff * diff

        # atomic compare & set best
        old = cuda.atomic.min(smem, 0, err)
        cuda.syncthreads()
        # if we are the winner, write params
        if err <= smem[0]:
            if tx == 0:
                smem[1] = dx
                smem[2] = dy
                smem[3] = t
                smem[4] = a
                smem[5] = b
            cuda.syncthreads()
            # store to global output
            out_dx[range_idx] = int32(smem[1])
            out_dy[range_idx] = int32(smem[2])
            out_t[range_idx]  = int32(smem[3])
            out_a[range_idx]  = smem[4]
            out_b[range_idx]  = smem[5]

def main():
    if len(sys.argv) < 4:
        print("Usage: python fractal_compress.py <input.jpg> <output.fcomp> <blockSize>")
        sys.exit(1)

    infile  = sys.argv[1]
    outfile = sys.argv[2]
    N       = int(sys.argv[3])

    img_bgr = cv2.imread(infile)
    if img_bgr is None:
        print("Cannot load image")
        sys.exit(1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    # prepare domain block positions
    domX = []
    domY = []
    for y in range(0, H - 2*N + 1, N):
        for x in range(0, W - 2*N + 1, N):
            domX.append(x)
            domY.append(y)
    domX = np.array(domX, dtype=np.int32)
    domY = np.array(domY, dtype=np.int32)
    domCount = domX.size

    # allocate outputs
    blocks_x = W // N
    blocks_y = H // N
    rangeCount = blocks_x * blocks_y

    out_dx = np.zeros(rangeCount, dtype=np.int32)
    out_dy = np.zeros(rangeCount, dtype=np.int32)
    out_t  = np.zeros(rangeCount, dtype=np.int32)
    out_a  = np.zeros(rangeCount, dtype=np.float32)
    out_b  = np.zeros(rangeCount, dtype=np.float32)

    # copy to device
    d_img   = cuda.to_device(img_gray)
    d_domX  = cuda.to_device(domX)
    d_domY  = cuda.to_device(domY)
    d_out_dx = cuda.to_device(out_dx)
    d_out_dy = cuda.to_device(out_dy)
    d_out_t  = cuda.to_device(out_t)
    d_out_a  = cuda.to_device(out_a)
    d_out_b  = cuda.to_device(out_b)

    threads_per_block = int(math.ceil(domCount*8/32)*32)
    blocks_per_grid = rangeCount

    shmem_size = (1 + 5) * 4  # floats

    compress_kernel[blocks_per_grid, threads_per_block, shmem_size](
        d_img, W, H, N, d_domX, d_domY, domCount,
        d_out_dx, d_out_dy, d_out_t, d_out_a, d_out_b
    )
    cuda.synchronize()

    # copy back
    d_out_dx.copy_to_host(out_dx)
    d_out_dy.copy_to_host(out_dy)
    d_out_t.copy_to_host(out_t)
    d_out_a.copy_to_host(out_a)
    d_out_b.copy_to_host(out_b)

    # write .fcomp
    with open(outfile, "wb") as f:
        f.write(struct.pack("i", W))
        f.write(struct.pack("i", H))
        f.write(struct.pack("i", N))
        for i in range(rangeCount):
            f.write(struct.pack("i", out_dx[i]))
            f.write(struct.pack("i", out_dy[i]))
            f.write(struct.pack("i", out_t[i]))
            f.write(struct.pack("f", out_a[i]))
            f.write(struct.pack("f", out_b[i]))

    print("Compression complete.")

if __name__ == "__main__":
    main()
