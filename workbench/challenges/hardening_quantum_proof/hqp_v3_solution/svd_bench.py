#!/usr/bin/env python3
"""Micro-benchmark: cuSOLVER SVD drivers on the matrix shapes the zip-up sweeps hit.

Run on the GPU you care about (Colab T4, Modal A100/H100, validator-class hardware):

    python svd_bench.py

Decides whether HQP_SVD_DRIVER=gesvdj is safe and profitable vs the default gesvd.
Shapes: (rows, cols) as produced by truncating an MPO site tensor of bond chi with
two physical legs (d^2=4) — e.g. chi=512 -> (2048, 1024).
"""
import time

import torch

SHAPES = [(512, 1024), (1024, 2048), (2048, 2048), (3200, 1600), (4096, 4096)]
REPS = 5


def bench(driver, A):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        for _ in range(REPS):
            torch.linalg.svd(A, full_matrices=False, driver=driver)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / REPS, None
    except Exception as e:
        torch.cuda.synchronize()
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")
    print(f"{'shape':>14} | {'gesvd (s)':>10} | {'gesvdj (s)':>10} | speedup")
    print("-" * 60)
    tot_g = tot_j = 0.0
    for m, n in SHAPES:
        A = torch.randn(m, n, dtype=torch.complex64, device="cuda")
        tg, eg = bench("gesvd", A)
        tj, ej = bench("gesvdj", A)
        del A
        torch.cuda.empty_cache()
        g = f"{tg:.4f}" if tg else f"FAIL"
        j = f"{tj:.4f}" if tj else (f"FAIL ({ej})" if ej else "n/a")
        s = f"{tg/tj:5.1f}x" if (tg and tj) else "-"
        print(f"{f'({m},{n})':>14} | {g:>10} | {j:>10} | {s}")
        if tg and tj:
            tot_g += tg
            tot_j += tj
    if tot_g and tot_j:
        print("-" * 60)
        print(f"TOTAL gesvd {tot_g:.3f}s vs gesvdj {tot_j:.3f}s -> {tot_g/tot_j:.2f}x")
        print("=> export HQP_SVD_DRIVER=gesvdj" if tot_g / tot_j >= 1.5
              else "=> keep HQP_SVD_DRIVER=gesvd")


if __name__ == "__main__":
    main()
