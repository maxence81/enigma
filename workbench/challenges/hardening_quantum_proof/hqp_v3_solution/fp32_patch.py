import os

import autoray
import quimb.tensor.decomp as _qd

# ------------------------------------------------------------------
# SVD driver for complex64 CUDA tensors.
#   "gesvd"  (default, proven): cuSOLVER QR-iteration. Stable on every shape seen so
#            far, but slow on small/medium matrices.
#   "gesvdj": cuSOLVER Jacobi. Usually 2-5x faster on the matrix sizes the zip-up
#            sweeps produce, but can fail to converge on some shapes -> we then fall
#            back to gesvd, then to a complex128 gesvd, so correctness is preserved.
# The final call chain is always safe; the knob only changes the fast path.
# Run svd_bench.py (in this bundle) on the target GPU to decide before flipping.
# ------------------------------------------------------------------
_SVD_DRIVER = os.environ.get("HQP_SVD_DRIVER", "gesvd")


def _sgn(x):
    xp = getattr(_qd, 'get_namespace', autoray.get_namespace)(x)
    ax = xp.abs(x)
    x0 = ax < 1e-12
    return (x + x0) / (ax + x0)


_qd.sgn = _sgn

try:
    import torch
    _svd = torch.linalg.svd

    def rsvd(A, full_matrices=False, **k):
        if A.dtype == torch.complex64:
            if A.is_cuda:
                driver = _SVD_DRIVER
                try:
                    return _svd(A, full_matrices=False, driver=driver)
                except Exception:
                    try:
                        return _svd(A, full_matrices=False, driver="gesvd")
                    except Exception:
                        U, s, Vh = _svd(A.to(torch.complex128), full_matrices=False, driver="gesvd")
                        return U.to(torch.complex64), s.to(torch.float32), Vh.to(torch.complex64)
            else:
                try:
                    return _svd(A, full_matrices=False)
                except Exception:
                    U, s, Vh = _svd(A.to(torch.complex128), full_matrices=False)
                    return U.to(torch.complex64), s.to(torch.float32), Vh.to(torch.complex64)
        if A.dtype == torch.complex128 and A.is_cuda:
            # c128 on GPU: cuSOLVER's default gesvdj can fail to converge on large bonds;
            # gesvd is the robust QR-based path (same reason the c64 branch uses it).
            try:
                return _svd(A, full_matrices=False, driver="gesvd")
            except Exception:
                return _svd(A, full_matrices=False)
        return _svd(A, full_matrices=False)

    _qr = torch.linalg.qr

    def rqr(A, *a, **k):
        if A.dtype == torch.complex64:
            try:
                return _qr(A, *a, **k)
            except Exception:
                Q, R = _qr(A.to(torch.complex128), *a, **k)
                return Q.to(torch.complex64), R.to(torch.complex64)
        return _qr(A, *a, **k)

    torch.linalg.svd = rsvd
    torch.linalg.qr = rqr
    autoray.register_function('torch', 'linalg.svd', rsvd)
    autoray.register_function('torch', 'linalg.qr', rqr)
except ImportError:
    pass

# ------------------------------------------------------------------
# Anti-collapse safeguard: enforce a minimum bond dimension floor
# (HQP_MIN_BOND, default 64) during SVD truncation to prevent the MPO
# from collapsing into an unentangled product state (bond 1) if all
# singular values drop below the absolute cutoff threshold.
# ------------------------------------------------------------------
_orig_trim = _qd._trim_and_renorm_svd_result


def _safe_trim_and_renorm_svd_result(
    U, s, VH, cutoff=-1.0, cutoff_mode=1, max_bond=-1, absorb=0, renorm=0, *args, **kwargs
):
    min_b = int(os.environ.get("HQP_MIN_BOND", "64"))
    if min_b > 1:
        s_len = autoray.do("shape", s)[0]
        if (cutoff > 0.0) or (renorm > 0):
            if cutoff_mode in (1, "abs"):
                n_chi = autoray.do("count_nonzero", s > cutoff)
            elif cutoff_mode in (2, "rel"):
                n_chi = autoray.do("count_nonzero", s > cutoff * s[0])
            elif cutoff_mode in (3, 4, 5, 6, "sum2", "rsum2", "sum1", "rsum1"):
                pow = 2 if cutoff_mode in (3, 4, "sum2", "rsum2") else 1
                sp = s**pow
                csp = autoray.do("cumsum", sp, 0)
                tot = csp[-1]
                if cutoff_mode in (4, 6, "rsum2", "rsum1"):
                    n_chi = autoray.do("count_nonzero", csp < (1 - cutoff) * tot) + 1
                else:
                    n_chi = autoray.do("count_nonzero", (tot - csp) > cutoff) + 1
            else:
                n_chi = autoray.do("count_nonzero", s > cutoff)
            n_chi = max(int(n_chi), 1)
        elif max_bond > 0:
            n_chi = max_bond
        else:
            n_chi = s_len
        # Safeguard: never drop below min_b (up to full available rank s_len)
        n_chi = max(n_chi, min(min_b, s_len))
        if max_bond > 0:
            n_chi = min(n_chi, max_bond)
        try:
            return _orig_trim(
                U, s, VH, cutoff=-1.0, cutoff_mode=cutoff_mode, max_bond=n_chi, absorb=absorb, renorm=renorm, *args, **kwargs
            )
        except TypeError:
            return _orig_trim(
                U, s, VH, cutoff=-1.0, cutoff_mode=cutoff_mode, max_bond=n_chi, absorb=absorb, renorm=renorm
            )
    try:
        return _orig_trim(
            U, s, VH, cutoff=cutoff, cutoff_mode=cutoff_mode, max_bond=max_bond, absorb=absorb, renorm=renorm, *args, **kwargs
        )
    except TypeError:
        return _orig_trim(
            U, s, VH, cutoff=cutoff, cutoff_mode=cutoff_mode, max_bond=max_bond, absorb=absorb, renorm=renorm
        )


_qd._trim_and_renorm_svd_result = _safe_trim_and_renorm_svd_result
