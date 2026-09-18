import os
import fp32_patch  # ensure safe SVD truncation and anti-collapse safeguards are active
from quimb.tensor import tensor_network_1d_compress, MatrixProductOperator, MatrixProductState, Circuit

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit

# Memory-bounded MPO·MPO compression. quimb's apply(contract=True) stacks the two
# operators then contracts each site of the FULL bond1*bond2 product via cotengra,
# which can pick a pathological-memory path and OOM once the state-MPO bond grows
# (observed: 21 GB kill at bond ~256). "zipup"/"dm" instead sweep and truncate
# bond-by-bond, so peak memory is bounded by max_bond, never the full product.
_COMPRESS_METHOD = os.environ.get("HQP_COMPRESS_METHOD", "zipup")

# ------------------------------------------------------------------
#  Constructors
# ------------------------------------------------------------------

def mpo_from_circuit(circ: Circuit):
    # add dummy rz to cover all sites
    for q in range(circ.N):
    #    circ.rz(0.0, q)
        circ.u3(0, 0, 0, q)
    tn_uni = circ.get_uni()

    # contract gates per site tag
    for st in list(tn_uni.site_tags):
        tn_uni ^= st

    # make sure bonds are simple 1D chain bonds
    tn_uni.fuse_multibonds_()  

    # cast as MatrixProductOperator
    mpo = tn_uni.view_as_(
        MatrixProductOperator,
        cyclic=False,
        L=circ.N,
    )

    mpo.ensure_bonds_exist()
    return mpo


# ------------------------------------------------------------------
#  MPO x MPO composition
# ------------------------------------------------------------------

def apply_mpo(mpo1: MatrixProductOperator, mpo2: MatrixProductOperator,
                side,
                max_bond=None,
                cutoff=0.0,
                contract=True,
                compress=True,
                **compress_opts):
    """Memory-bounded MPO·MPO product.

    Result is the same compressed product as quimb's apply(contract=True,
    compress=True), but computed by stacking the two operators LAZILY
    (contract=False -> no high-bond per-site contraction) and then compressing
    with a bounded 1D sweep (zip-up / density-matrix). Peak memory is bounded by
    max_bond rather than the full bond1*bond2 product, so it never OOMs on the
    21 GB cap as the state-MPO bond grows.
    """
    # Form the product MPO per-site (contract=True): each site contracts just the
    # two local tensors -> a clean 1-tensor/site MPO with bond bA*bB. This is
    # cheap (bounded by the local bonds, ~tens of MB here). Crucially we keep
    # compress=False so quimb does NOT run its default density-matrix compress,
    # which contracts the high-bond product against its conjugate via cotengra and
    # is what blew past the 21 GB cap.
    if side == "right":
        prod = mpo1.apply(mpo2, compress=False, contract=True)
        L = len(mpo1.sites)
    elif side == "left":
        prod = mpo2.apply(mpo1, compress=False, contract=True)
        L = len(mpo2.sites)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    if not compress:
        return prod

    # Compress the clean product MPO with a memory-bounded 1D sweep (zip-up):
    # a left-to-right SVD sweep whose peak memory is bounded by max_bond, never
    # the density matrix. zip-up on a proper 1-tensor/site MPO avoids the
    # permute-arrays mismatch that the lazy (2-tensor/site) structure triggers.
    #
    # Truncation mode (HQP_CUTOFF_MODE):
    #   "rel" (default, upstream): quimb's relative cutoff (cutoff_mode=1) —
    #       discards singular values below cutoff * s_max. Scale-free, but on a
    #       peaked circuit the peak-carrying sectors can sit far below s_max:
    #       they get trimmed first, which is what dissolved the d3 peak in the
    #       2026-09-07/08 runs (w1 ~ 2e-10, MPO core bond 34 at end).
    #   "abs": absolute cutoff (quimb cutoff_mode=4) — discards values below
    #       the raw threshold, protecting every sector with amplitude above the
    #       floor regardless of scale. Verified on quimb 1.11: mode 4 keeps
    #       40-58 sectors where mode 1 keeps all-or-nothing on flat spectra.
    # "abs" is experimental: larger bonds -> slower absorption; pair it with a
    # small cutoff (e.g. 1e-3) and a high max_bond (4096).
    cutoff_mode = os.environ.get("HQP_CUTOFF_MODE", "rel")
    if cutoff_mode in ("abs", "rel", "sum2", "rsum2"):
        cutoff_opts = {"cutoff_mode": cutoff_mode, "cutoff": cutoff}
    else:
        cutoff_opts = {"cutoff": cutoff}
    out = tensor_network_1d_compress(
        prod,
        max_bond=max_bond,
        method=_COMPRESS_METHOD,
        optimize="auto-hq",
        permute_arrays=False,   # skip cosmetic axis reorder (buggy bond-name lookup on these MPOs); indices carry the meaning
        inplace=True,
        **cutoff_opts,
    )
    if not isinstance(out, MatrixProductOperator):
        out = out.view_as_(MatrixProductOperator, cyclic=False, L=L)
    out.ensure_bonds_exist()
    return out



# ------------------------------------------------------------------
#  Applying circuits to MPO
# ------------------------------------------------------------------


def apply_circuit(mpo, circ, side, max_bond=None, cutoff=0.0, contract=True, compress=True, **compress_opts):
    return apply_mpo(mpo, mpo_from_circuit(circ), side=side, max_bond=max_bond, cutoff=cutoff, contract=contract, compress=compress, **compress_opts)


def apply_swaps(mpo: MatrixProductOperator, swaps_l, swaps_r, max_bond=None, cutoff=0.0, to_backend=None, inplace=False):
    N = len(mpo.sites)
    qc_swaps_l = QuantumCircuit(N)
    qc_swaps_r = QuantumCircuit(N)

    for q0, q1 in swaps_l:
        qc_swaps_l.swap(q0, q1)

    for q0, q1 in swaps_r:
        qc_swaps_r.swap(q0, q1)

    mpo_out = mpo if inplace else mpo.copy()

    if len(swaps_l) > 0:
        circ_l = quimb_circuit(qc_swaps_l.inverse().decompose("swap"), Circuit, to_backend=to_backend)
        mpo_out = apply_circuit(mpo_out, circ_l, side="right", max_bond=max_bond, cutoff=cutoff)
    
    if len(swaps_r) > 0:
        circ_r = quimb_circuit(qc_swaps_r.decompose("swap"), Circuit, to_backend=to_backend)
        mpo_out = apply_circuit(mpo_out, circ_r, side="left", max_bond=max_bond, cutoff=cutoff) 

    return mpo_out

