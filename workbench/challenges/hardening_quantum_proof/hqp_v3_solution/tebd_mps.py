"""TEBD matrix-product-state simulator for peaked circuits — torch/CUDA port.

Faithful port of the proven milestone-1 approach (custom MPS, mixed-canonical
SVD truncation, swap network for all-to-all CZ, canonical beam-search argmax) to
torch so it runs on the validator GPU via the CUDA libs torch already bundles —
no cupy/libcublas dependency. Algorithm validated against exact statevector.
"""
import os
import time
import torch


class MPS:
    def __init__(self, n, dtype, dev, perm=None):
        self.n = n
        self.dtype = dtype
        self.dev = dev
        z = torch.zeros((1, 2, 1), dtype=dtype, device=dev)
        z[0, 0, 0] = 1.0
        self.A = [z.clone() for _ in range(n)]
        self.center = 0
        if perm is None:
            self.pos = list(range(n))
            self.qubit_at = list(range(n))
        else:
            assert sorted(perm) == list(range(n))
            self.pos = list(perm)
            self.qubit_at = [0] * n
            for q, p in enumerate(perm):
                self.qubit_at[p] = q

    def _move_right(self):
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl * d, Dr)
        Q, R = torch.linalg.qr(M)
        k = Q.shape[1]
        self.A[p] = Q.reshape(Dl, d, k)
        self.A[p + 1] = torch.tensordot(R, self.A[p + 1], dims=([1], [0]))
        self.center = p + 1

    def _move_left(self):
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl, d * Dr)
        Q, R = torch.linalg.qr(M.conj().T)
        Qr = Q.conj().T
        L = R.conj().T
        k = Qr.shape[0]
        self.A[p] = Qr.reshape(k, d, Dr)
        self.A[p - 1] = torch.tensordot(self.A[p - 1], L, dims=([2], [0]))
        self.center = p - 1

    def move_center_to(self, target):
        while self.center < target:
            self._move_right()
        while self.center > target:
            self._move_left()

    def apply_two_site(self, p, G4, chi, cutoff):
        A1, A2 = self.A[p], self.A[p + 1]
        Dl = A1.shape[0]; Dr = A2.shape[2]
        theta = torch.tensordot(A1, A2, dims=([2], [0]))           # (Dl,2,2,Dr)
        theta = torch.einsum("IJij,aijc->aIJc", G4, theta)
        M = theta.reshape(Dl * 2, 2 * Dr)
        if M.is_cuda:
            driver = os.environ.get("HQP_SVD_DRIVER", "gesvd")
            try:
                U, s, Vh = torch.linalg.svd(M, full_matrices=False, driver=driver)
            except Exception:
                U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        else:
            U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        k = min(chi, M.shape[0], M.shape[1])
        U = U[:, :k]; s = s[:k]; Vh = Vh[:k, :]
        s = s / torch.linalg.vector_norm(s)
        self.A[p] = U.reshape(Dl, 2, k)
        self.A[p + 1] = (s[:, None] * Vh).reshape(k, 2, Dr)
        self.center = p + 1
        gov = getattr(self, "governor", None)
        if gov is not None:
            gov.pace_after_svd()

    def apply_1q(self, q, U2):
        p = self.pos[q]
        self.A[p] = torch.einsum("ij,ajb->aib", U2, self.A[p])

    def _swap_adjacent(self, p, chi, cutoff, SWAP4):
        self.move_center_to(p)
        self.apply_two_site(p, SWAP4, chi, cutoff)
        qa, qb = self.qubit_at[p], self.qubit_at[p + 1]
        self.qubit_at[p], self.qubit_at[p + 1] = qb, qa
        self.pos[qa], self.pos[qb] = p + 1, p

    def apply_2q(self, q1, q2, G4, chi, cutoff, SWAP4):
        p1, p2 = self.pos[q1], self.pos[q2]
        if p1 > p2:
            p1, p2 = p2, p1
            q1, q2 = q2, q1
            G4 = G4.permute(1, 0, 3, 2)
        while p2 > p1 + 1:
            self._swap_adjacent(p2 - 1, chi, cutoff, SWAP4)
            p2 -= 1
        self.move_center_to(p1)
        self.apply_two_site(p1, G4, chi, cutoff)

    def max_bond(self):
        return max(a.shape[2] for a in self.A)


def _gates(qc):
    import numpy as np
    index = {qb: i for i, qb in enumerate(qc.qubits)}
    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [index[qb] for qb in inst.qubits]
        yield np.asarray(op.to_matrix()), qs


def evolve(qc, chi, cutoff=1e-10, dtype=None, dev=None, perm=None, log=print):
    dev = dev or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype or torch.complex128  # double precision (accuracy-first)
    SWAP = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=dtype, device=dev).reshape(2, 2, 2, 2)
    mps = MPS(qc.num_qubits, dtype, dev, perm=perm)
    t0 = time.time()
    ng = 0
    for mat, qs in _gates(qc):
        M = torch.as_tensor(mat, dtype=dtype, device=dev)
        if len(qs) == 1:
            mps.apply_1q(qs[0], M)
        elif len(qs) == 2:
            mps.apply_2q(qs[0], qs[1], M.reshape(2, 2, 2, 2), chi, cutoff, SWAP)
        else:
            raise ValueError(f"{len(qs)}-qubit gate unsupported")
        ng += 1
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    log(f"    [tebd_mps] {ng} gates, evolve={dt:.1f}s reached_chi={mps.max_bond()}")
    return mps, dt


def topk(mps, beam=512, k=8):
    """Canonical beam search. Returns LOGICAL-order [(bits, prob), ...] top-k."""
    mps.move_center_to(0)
    A = mps.A
    ones = torch.ones((1, 1), dtype=A[0].dtype, device=A[0].device)
    beams = [("", ones)]
    for a in A:
        cand = []
        for bits, vec in beams:
            for b in (0, 1):
                nv = vec @ a[:, b, :]
                w = float((nv.conj() * nv).real.sum().item())
                cand.append((w, bits + str(b), nv))
        cand.sort(key=lambda t: t[0], reverse=True)
        beams = [(bits, nv) for (_, bits, nv) in cand[:beam]]
    res = []
    for bits, nv in beams:
        logical = ["0"] * mps.n
        for chain_pos, ch in enumerate(bits):
            logical[mps.qubit_at[chain_pos]] = ch
        res.append(("".join(logical), float((nv.conj() * nv).real.sum().item())))
    res.sort(key=lambda t: t[1], reverse=True)
    return res[:k]


def topk_reverse(mps, beam=512, k=8):
    """Reverse canonical beam search (site n-1 -> 0). Returns LOGICAL-order [(bits, prob), ...] top-k."""
    n = mps.n
    mps.move_center_to(n - 1)
    A = mps.A
    ones = torch.ones((1, 1), dtype=A[0].dtype, device=A[0].device)
    beams = [([], ones)]
    for p in range(n - 1, -1, -1):
        a = A[p]
        cand = []
        for bits_rev, vec in beams:
            for b in (0, 1):
                nv = a[:, b, :] @ vec
                w = float((nv.conj() * nv).real.sum().item())
                cand.append((w, bits_rev + [b], nv))
        cand.sort(key=lambda t: t[0], reverse=True)
        beams = [(bits_rev, nv) for (_, bits_rev, nv) in cand[:beam]]

    res = []
    for bits_rev, nv in beams:
        chain_bits = list(reversed(bits_rev))
        logical = ["0"] * n
        for chain_pos, b in enumerate(chain_bits):
            logical[mps.qubit_at[chain_pos]] = str(b)
        res.append(("".join(logical), float((nv.conj() * nv).real.sum().item())))
    res.sort(key=lambda t: t[1], reverse=True)
    return res[:k]


def eval_bitstrings_batched(mps, bitstrings, batch_size=2048):
    """Evaluate exact MPS probabilities |<x|psi>|^2 for a list of logical bitstrings in batches on GPU.
    Returns a numpy array of float probabilities.
    """
    import numpy as np
    n = mps.n
    dev = mps.A[0].device
    A_perm = [a.permute(1, 0, 2) for a in mps.A]
    M = len(bitstrings)
    if M == 0:
        return np.array([])

    chain_bits_all = np.zeros((M, n), dtype=np.int64)
    for m, s in enumerate(bitstrings):
        for p in range(n):
            q = mps.qubit_at[p]
            chain_bits_all[m, p] = int(s[q])

    chain_bits_tensor = torch.tensor(chain_bits_all, dtype=torch.long, device=dev)
    all_probs = []

    for start_idx in range(0, M, batch_size):
        end_idx = min(start_idx + batch_size, M)
        b_batch = chain_bits_tensor[start_idx:end_idx]
        b0 = b_batch[:, 0]
        V = A_perm[0][b0]
        for p in range(1, n):
            bp = b_batch[:, p]
            Mp = A_perm[p][bp]
            V = torch.bmm(V, Mp)
        amps = V.squeeze(1).squeeze(1)
        probs = (amps.conj() * amps).real
        all_probs.append(probs.detach().cpu().numpy())

    return np.concatenate(all_probs)


def topk_bidirectional(mps, beam=512, k=16):
    """Runs forward and reverse canonical beam searches, combines unique candidates,
    evaluates exact amplitudes on GPU, and sorts them.
    Also returns consensus mask and disagreeing qubit indices.
    """
    fwd_cands = topk(mps, beam=beam, k=k)
    rev_cands = topk_reverse(mps, beam=beam, k=k)

    cands_dict = {}
    for bits, _ in fwd_cands:
        cands_dict[bits] = "fwd"
    for bits, _ in rev_cands:
        if bits in cands_dict:
            cands_dict[bits] = "both"
        else:
            cands_dict[bits] = "rev"

    unique_bits = list(cands_dict.keys())
    probs = eval_bitstrings_batched(mps, unique_bits)

    combined = []
    for bits, p in zip(unique_bits, probs):
        combined.append((bits, float(p), cands_dict[bits]))
    combined.sort(key=lambda t: t[1], reverse=True)

    top_fwd = fwd_cands[0][0]
    top_rev = rev_cands[0][0]
    disagree_indices = [i for i in range(mps.n) if top_fwd[i] != top_rev[i]]

    return combined[:k], disagree_indices, top_fwd, top_rev


def refine_subspace(mps, base_bitstring, uncertain_indices, max_bits=14):
    """Exhaustively evaluates all 2^len(uncertain_indices) bit combinations on GPU.
    Returns (best_bitstring, best_prob).
    """
    import numpy as np
    if not uncertain_indices:
        p = float(eval_bitstrings_batched(mps, [base_bitstring])[0])
        return base_bitstring, p

    indices = list(uncertain_indices)[:max_bits]
    num_combos = 1 << len(indices)
    chars = list(base_bitstring)
    combos = []
    for mask in range(num_combos):
        cand = list(chars)
        for bit_idx, pos in enumerate(indices):
            cand[pos] = "1" if (mask & (1 << bit_idx)) else "0"
        combos.append("".join(cand))

    probs = eval_bitstrings_batched(mps, combos)
    best_idx = int(np.argmax(probs))
    return combos[best_idx], float(probs[best_idx])


def local_search_neighborhood(mps, seed_bitstring, max_rounds=5, flip_order=2, log=print):
    """Hill climbing coordinate descent on the exact MPS amplitude surface."""
    import numpy as np
    n = mps.n
    current = seed_bitstring
    curr_prob = float(eval_bitstrings_batched(mps, [current])[0])
    for r in range(max_rounds):
        neighbors = []
        chars = list(current)
        for i in range(n):
            flipped = list(chars)
            flipped[i] = "1" if chars[i] == "0" else "0"
            neighbors.append("".join(flipped))
        if flip_order >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    flipped = list(chars)
                    flipped[i] = "1" if chars[i] == "0" else "0"
                    flipped[j] = "1" if chars[j] == "0" else "0"
                    neighbors.append("".join(flipped))
        probs = eval_bitstrings_batched(mps, neighbors)
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        if best_prob > curr_prob * (1 + 1e-6):
            current = neighbors[best_idx]
            curr_prob = best_prob
        else:
            break
    return current, curr_prob


def save_mps(mps, filepath):
    state = {
        "n": mps.n,
        "center": mps.center,
        "pos": list(mps.pos),
        "qubit_at": list(mps.qubit_at),
        "A": [a.detach().cpu() for a in mps.A],
    }
    torch.save(state, filepath)


def load_mps(filepath, dev=None, dtype=None):
    dev = dev or ("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(filepath, map_location=dev)
    mps = MPS(state["n"], dtype or state["A"][0].dtype, dev, perm=state["pos"])
    mps.center = state["center"]
    mps.qubit_at = state["qubit_at"]
    mps.A = [a.to(device=dev, dtype=dtype or a.dtype) for a in state["A"]]
    return mps

