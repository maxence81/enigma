from quimb.tensor import MatrixProductOperator, MatrixProductState, Circuit, CircuitMPS, tensor_network_1d_compress

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit

from circuit_mpo import apply_circuit, apply_swaps, mpo_from_circuit

from utils import iter_layers, merge_layers, elem_counts, merge_gates, get_tn_info

import numpy as np
import time

import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    level=logging.INFO
)

# ------------------------------------------------------------------
#  Rewiring
# ------------------------------------------------------------------
import os
_os = os
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from qiskit.transpiler import CouplingMap

_SABRE_TRIALS = int(_os.environ.get("HQP_SABRE_TRIALS", "200"))


def layer_weight(qc):
    """Cheap proxy for how much absorbing this layer will grow the core MPO:
    entangling (2q+) gates dominate, 1q gates contribute marginally."""
    n2 = sum(1 for inst in qc.data if len(inst.qubits) > 1)
    return n2 + 0.1 * len(qc.data)


def rewire_layers(ls, perm, seed=None):
    nq = len(perm)
    qc = merge_layers(ls)
    qc = QuantumCircuit(nq).compose(qc, qubits=np.argsort(perm))

    qc = ElidePermutations()(qc)
    ss = SabreSwap(coupling_map=CouplingMap.from_line(ls[0].num_qubits), heuristic='decay', trials=_SABRE_TRIALS, seed=seed)
    qc = ss(qc)

    return list(iter_layers(qc))



# ------------------------------------------------------------------
#  Unswapping
# ------------------------------------------------------------------

def get_bond_sizes(mpo: MatrixProductOperator):
    return np.array([mpo.bond_size(ii,ii+1) for ii in range(len(mpo.sites) - 1)])


def swap_perm(perm, swaps):
    for q0, q1 in swaps:
        (perm[q0], perm[q1]) = (perm[q1], perm[q0])
    return perm


def get_good_swaps(mpo, qubit_pairs, how, max_bond, cutoff, to_backend=None, equal=False):
    current_bonds = get_bond_sizes(mpo)
    #log_print("    [debug](select)(bond sizes before) -> ", current_bonds.tolist())

    swaps_l = qubit_pairs if how in ("left", "both") else []
    swaps_r = qubit_pairs if how in ("right", "both") else []

    mpo_tmp = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend)
    new_bonds = get_bond_sizes(mpo_tmp)
    del mpo_tmp
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if equal is None:
        new_bonds = new_bonds + (np.random.rand(*new_bonds.shape)-0.5)
        improved = np.nonzero(new_bonds < current_bonds)[0]
    elif equal:
        improved = np.nonzero(new_bonds <= current_bonds)[0]
    else:
        improved = np.nonzero(new_bonds < current_bonds)[0]

    return improved


def unswap(mpo: MatrixProductOperator, hows=("left", "right", "both"), max_bond=2048, cutoff=0.0001, max_its=None, equal=False, to_backend=None, t0=0, deadline=None):
    if max_its is None:
        max_its = int(_os.environ.get("HQP_US_MAX_ITS", "3"))
    num_qubits = len(mpo.sites)
    all_pairs = [(i, i+1) for i in range(num_qubits-1)]

    perm_left = list(range(len(mpo.sites)))
    perm_right = list(range(len(mpo.sites)))

    logging.info("    [start unswap] -> " + str(get_tn_info(mpo)))
    num_improvements = 1
    start_counts = 1
    end_counts = 0
    ii = 0

    stats_data = []
    while num_improvements > 0 and ii < max_its and start_counts != end_counts:
        if deadline is not None and time.time() > deadline:
            logging.info("    [unswap deadline reached] -> stop")
            break
        num_improvements = 0
        start_counts = elem_counts(mpo)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        for how in hows:
            for parity in [0, 1]:
                # Estimate which qubit pairs to swap
                new_swap_ids = get_good_swaps(mpo, qubit_pairs=all_pairs[parity::2], how=how, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend, equal=equal)
                new_swaps = [all_pairs[i] for i in new_swap_ids if i % 2 == parity]

                # Apply the selected swaps
                swaps_l = new_swaps if how in ("left", "both") else []
                swaps_r = new_swaps if how in ("right", "both") else []
                mpo = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                # Update the permutations
                if how in ("left", "both"):
                    perm_left = swap_perm(perm_left, new_swaps)
                if how in ("right", "both"):
                    perm_right = swap_perm(perm_right, new_swaps)
    
                # Track how many new swaps were applied
                num_improvements += len(new_swap_ids)
                stats_data.append({"time": time.perf_counter()-t0, "stage": "unswapping", "iteration": ii, "side": how, "parity": parity, "new_swaps": len(new_swap_ids), "total_swaps": num_improvements, **get_tn_info(mpo)})
                logging.info(f"    [{ii} | {how} | {parity}](new_swaps: {len(new_swap_ids)} | total: {num_improvements}) -> " + str(get_tn_info(mpo)))

        end_counts = elem_counts(mpo)
        # Relative improvement check: stop if elements didn't drop by at least 1%
        # to prevent unproductive chatter/ping-pong between marginal swap pairs.
        if end_counts >= start_counts * 0.99:
            logging.info(f"    [unswap converged early: {start_counts} -> {end_counts}]")
            break
        ii += 1
    logging.info(f"    [end unswap] -> " + str(get_tn_info(mpo)))

    return mpo, (perm_left, perm_right), stats_data


# ------------------------------------------------------------------
#  MPO Cancellation + Unswapping
# ------------------------------------------------------------------

def mpo_compress_unswap(circuit: QuantumCircuit, max_bond=8192, cutoff=0.001, unswap_threshold=1e6, early_stopping_gates=100, center_ratio=0.5, equal=False, flip_freq=None, max_its=None, to_backend=None, seed=None, hows=("both", "left", "right"), mpo_core=None, deadline=None):
    if max_its is None:
        max_its = int(_os.environ.get("HQP_US_MAX_ITS", "3"))
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    t0 = time.perf_counter()

    # Split circuit into left and right
    if type(center_ratio) is float:
        C = int(len(circuit) * center_ratio)
    elif type(center_ratio) is int:
        C = center_ratio
    circuit_left = merge_gates(circuit[:C], circuit.num_qubits).inverse()
    circuit_right = merge_gates(circuit[C:], circuit.num_qubits)
    if "measure" not in circuit_right.count_ops():
        circuit_right.measure_all()
    if "measure" not in circuit_left.count_ops():
        circuit_left.measure_all()

    layers_left = list(iter_layers(circuit_left))
    layers_right = list(iter_layers(circuit_right))


    T_U = circuit.count_ops().get("unitary", 0)
    T_UL = circuit_left.count_ops().get("unitary", 0)
    T_UR = circuit_right.count_ops().get("unitary", 0)

    logging.info(f"Total unitaries: {T_U} = {T_UL} (left) + {T_UR} (right)")

    # Rewire layers
    layers_left = rewire_layers(layers_left, np.arange(circuit.num_qubits, dtype=int), seed=seed)
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]

    layers_right = rewire_layers(layers_right, np.arange(circuit.num_qubits, dtype=int), seed=seed)
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    # Start the MPO and counters
    ii_left = 0
    ii_right = 0
    do_left = False
    if mpo_core is None:
        mpo_core = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    logging.info("[start compressing] -> " + str(get_tn_info(mpo_core)))


    total_u_consumed = 0
    current_u_consumed = 0
    total_u_consumed_left = 0
    total_u_consumed_right = 0

    stats_data = []

    # Greedy absorption (HQP_GREEDY_ABSORB=1, default): the reference recipe computes
    # BOTH the left and right candidate absorptions at every step (two full
    # MPO-product + zip-up sweep passes over all sites) only to keep the smaller one.
    # We instead absorb the lighter side first and accept it immediately when its size
    # is comfortably below the unswap threshold; the second trial is paid only near the
    # threshold, where the comparison actually matters. ~2x fewer compression sweeps.
    # Set HQP_GREEDY_ABSORB=0 to restore the exact reference both-sides behaviour.
    greedy_absorb = flip_freq is None and _os.environ.get("HQP_GREEDY_ABSORB", "1") == "1"
    greedy_soft = float(_os.environ.get("HQP_GREEDY_SOFT", "1.0"))
    unswap_cycle_count = 0

    # Start loop
    while ii_left < len(layers_left) or ii_right < len(layers_right):
        if deadline is not None and time.time() > deadline:
            logging.info(f"[deadline reached: {T_U - total_u_consumed} unitaries left] -> stop absorbing, extract best-so-far")
            break

        mpo_left = None
        counts_left = 1e20
        mpo_right = None
        counts_right = 1e20

        def _absorb(is_left):
            nonlocal mpo_left, counts_left, mpo_right, counts_right
            if is_left:
                mpo_left = apply_circuit(mpo_core, q2c(layers_left[ii_left].inverse()), side="right", max_bond=max_bond, cutoff=cutoff)
                counts_left = elem_counts(mpo_left)
            else:
                mpo_right = apply_circuit(mpo_core, q2c(layers_right[ii_right]), side="left", max_bond=max_bond, cutoff=cutoff)
                counts_right = elem_counts(mpo_right)

        if greedy_absorb:
            has_l = ii_left < len(layers_left)
            has_r = ii_right < len(layers_right)
            if has_l and has_r:
                u_l = dict(layers_left[ii_left].count_ops()).get('unitary', 0)
                u_r = dict(layers_right[ii_right].count_ops()).get('unitary', 0)
                if u_l > 0 and u_r == 0:
                    first = "L"
                elif u_r > 0 and u_l == 0:
                    first = "R"
                else:
                    first = "L" if total_u_consumed_left <= total_u_consumed_right else "R"
            else:
                first = "L" if has_l else "R"
            second = "R" if first == "L" else "L"
            try:
                _absorb(first == "L")
            except KeyboardInterrupt:
                break
            c_first = counts_left if first == "L" else counts_right
            if c_first > greedy_soft * unswap_threshold:
                # Near (or over) the unswap budget: evaluate the other side too and
                # keep the smaller one, exactly like the reference both-sides trial.
                if (second == "L" and ii_left < len(layers_left)) or (second == "R" and ii_right < len(layers_right)):
                    try:
                        _absorb(second == "L")
                    except KeyboardInterrupt:
                        break
        else:
            # Reference behaviour: try both sides, keep the smaller one.
            if ii_left < len(layers_left):
                try:
                    mpo_left = apply_circuit(mpo_core, q2c(layers_left[ii_left].inverse()), side="right", max_bond=max_bond, cutoff=cutoff)
                except KeyboardInterrupt:
                    break
                counts_left = elem_counts(mpo_left)
            if ii_right < len(layers_right):
                try:
                    mpo_right = apply_circuit(mpo_core, q2c(layers_right[ii_right]), side="left", max_bond=max_bond, cutoff=cutoff)
                except KeyboardInterrupt:
                    break
                counts_right = elem_counts(mpo_right)

        if flip_freq is None:
            ops_l = dict(layers_left[ii_left].count_ops()) if ii_left < len(layers_left) else {}
            ops_r = dict(layers_right[ii_right].count_ops()) if ii_right < len(layers_right) else {}
            u_l = ops_l.get('unitary', 0)
            u_r = ops_r.get('unitary', 0)
            if u_l > 0 and u_r == 0 and counts_left < 1.5 * counts_right:
                do_left = True
            elif u_r > 0 and u_l == 0 and counts_right < 1.5 * counts_left:
                do_left = False
            else:
                do_left = counts_left < counts_right
        else:
            if mpo_left is None:
                do_left = False
            elif mpo_right is None:
                do_left = True
            elif (ii_right + ii_left) % flip_freq == 0:
                do_left = not do_left

        # CRITICAL ANTI-LIVELOCK INVARIANT:
        # If current_u_consumed < min_u, unswapping is prohibited to avoid cancelling identity SWAPs.
        # Once min_u is satisfied, unswap is triggered whenever:
        # 1. current_u_consumed >= unswap_interval (periodic cadence, default 20), OR
        # 2. total tensor elements exceed unswap_threshold.
        can_unswap = (current_u_consumed >= int(_os.environ.get("HQP_MIN_U_CONSUMED", "5")))
        unswap_interval = int(_os.environ.get("HQP_UNSWAP_INTERVAL", "20"))
        should_unswap = can_unswap and (
            current_u_consumed >= unswap_interval or
            [counts_right, counts_left][int(do_left)] >= unswap_threshold
        )
        if not should_unswap:                
            if do_left:
                mpo_core = mpo_left
                # Update counts
                new_ops = dict(layers_left[ii_left].count_ops())
                new_us = new_ops.get('unitary', 0)
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_left += new_us

                # Log
                side_chosen = "L"
                ii_left += 1
            else:
                mpo_core = mpo_right
                # Update counts
                new_ops = dict(layers_right[ii_right].count_ops())
                new_us = new_ops.get('unitary', 0)  
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_right += new_us
            
                # Log
                side_chosen = "R"
                ii_right += 1            
            
            mpo_left = None
            mpo_right = None
            
            logging.info((f"[{ii_right}R/{len(layers_right)}]" if side_chosen == "R" else f"[{ii_left}L/{len(layers_left)}]") + 
                         f"(swap: {new_swaps}, u: {new_us} | c_u: {current_u_consumed} | t_u_l: {total_u_consumed_left}/{T_UL} | t_u_r: {total_u_consumed_right}/{T_UR} | t_u: {total_u_consumed}/{T_U}) -> " +
                         str(get_tn_info(mpo_core)))
            stats_data.append({"time": time.perf_counter() - t0, "stage": "absorbing", "absorb_side": "left", 
                                "it_left": ii_left, "it_right": ii_right, "layers_left": len(layers_left), "layers_right": len(layers_right),
                                "u_consumed_total_left": total_u_consumed_left, "u_consumed_total_right": total_u_consumed_right, "u_consumed_total": total_u_consumed,
                                "swap_consumed": new_swaps, "u_consumed": new_us, "u_consumed_after_unswap": current_u_consumed, 
                                **get_tn_info(mpo_core)})
        
        # Unswap if both sides go over the size budget
        else: 
            unswap_cycle_count += 1
            mpo_left = None
            mpo_right = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # Apply unswapping
            try:
                mpo_core, (new_perm_left, new_perm_right), new_unswap_stats = unswap(mpo_core, hows=hows, max_bond=max_bond, cutoff=cutoff, max_its=max_its, equal=equal, to_backend=to_backend, t0=t0, deadline=deadline)
                stats_data += new_unswap_stats
            except KeyboardInterrupt:
                break        
            # Dynamic rewire seed to prevent deterministic routing livelocks
            rewire_seed = (seed + unswap_cycle_count * 1007) if seed is not None else None
            # Rewire left circuit
            if ii_left < len(layers_left):
                layers_left = rewire_layers(layers_left[(ii_left):] + init_meas, new_perm_left, seed=rewire_seed)
                init_meas = layers_left[-2:]
                layers_left = layers_left[:-2]
            else:
                layers_left = []
            
            # Rewire right circuit
            if ii_right < len(layers_right):
                layers_right = rewire_layers(layers_right[(ii_right):] + final_meas, new_perm_right, seed=rewire_seed)
                final_meas = layers_right[-2:]
                layers_right = layers_right[:-2]
            else:
                layers_right = []
            
            ii_left = 0
            ii_right = 0
            current_u_consumed = 0

            # Stop early if there are few gates left
            if (T_U - total_u_consumed) <= early_stopping_gates:
                break
    
    # Remove any leftover layers
    layers_left = layers_left[(ii_left):] if ii_left < len(layers_left) else []
    layers_left += init_meas
    layers_right = layers_right[(ii_right):] if ii_right < len(layers_right) else []
    layers_right += final_meas

    logging.info(f"[end compressing](left: {len(layers_left)}, right: {len(layers_right)}) -> " + str(get_tn_info(mpo_core)))

    return mpo_core, layers_left, layers_right, stats_data


def mpo_to_mps(mpo_core, layers_left, layers_right, max_bond=4096, cutoff=0.001, to_backend=None):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    cutoff_mode = os.environ.get("HQP_CUTOFF_MODE", "abs")
    compress_opts = {"cutoff_mode": cutoff_mode} if cutoff_mode in ("abs", "rel", "sum2", "rsum2") else {}

    # Use the compressed MPO to get the MPS by applying it to |0> state
    final_mps = quimb_circuit(
        QuantumCircuit(len(mpo_core.sites)),
        quimb_circuit_class=CircuitMPS,
        to_backend=to_backend,
    ).psi

    # First take the leftover front layers
    layers_left = list(iter_layers(merge_layers(layers_left).inverse())) if len(layers_left) > 0 else []
    
    for ii_left in range(len(layers_left)):
        l_left = layers_left[ii_left]
        new_ops = dict(l_left.count_ops())
        layer_mpo = mpo_from_circuit(q2c(l_left))
        final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff, **compress_opts)
        logging.info(f"[Left {ii_left} / {len(layers_left)}] -> " + str(get_tn_info(final_mps)))

    logging.info("[Left MPS] -> " + str(get_tn_info(final_mps)))

    # Then apply the compressed MPO to the layers using memory-bounded zipup sweep
    prod = mpo_core.apply(final_mps, contract=False)
    final_mps = tensor_network_1d_compress(
        prod,
        max_bond=max_bond,
        cutoff=cutoff,
        method="zipup",
        **compress_opts,
    )
    if not isinstance(final_mps, MatrixProductState):
        final_mps = final_mps.view_as_(MatrixProductState, cyclic=False, L=len(mpo_core.sites))
    for t in final_mps:
        d = t.data
        m = d.abs().max() if hasattr(d, "abs") else np.abs(d).max()
        if m > 0 and not np.isnan(float(m)):
            t.modify(data=d / m)
    logging.info("[Left MPS + Core MPO] -> " + str(get_tn_info(final_mps)))

    # Then iterate through final layers if there are any
    final_meas = []
    for ii_right in range(len(layers_right)):
        l_right = layers_right[ii_right]
        new_ops = dict(l_right.count_ops())
        if "barrier" in new_ops or "measure" in new_ops:
            final_meas.append(l_right)
        else:
            layer_mpo = mpo_from_circuit(q2c(l_right))
            final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff, **compress_opts)
            logging.info(f"[Front MPS + Core MPO + Right {ii_right} / {len(layers_right)}] -> " + str(get_tn_info(final_mps)))
    
    for t in final_mps:
        d = t.data
        m = d.abs().max() if hasattr(d, "abs") else np.abs(d).max()
        if m > 0 and not np.isnan(float(m)):
            t.modify(data=d / m)
    logging.info(f"[Front MPS + Core MPO + Right MPS] -> " + str(get_tn_info(final_mps)))

    # Extract final permutation from measurements
    final_perm = [g.qubits[0]._index for g in final_meas[-1]]

    # Return MPS and final perm
    return final_mps, final_perm

