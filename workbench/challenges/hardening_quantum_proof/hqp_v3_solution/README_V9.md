# HQP difficulty 3 — submission v9

This bundle solves every supplied circuit from its QASM input. It contains no
known peaked states, filename shortcuts, or structural-fingerprint answers.

## Production path

- Circuits with at most 30 qubits use an exact statevector.
- Larger circuits use the level-2 winning MPO Unswap architecture.
- The difficulty-3 profile uses complex64 tensors, absolute truncation at
  `0.0002`, bond cap 512, and one bilateral Unswap pass every 20 absorbed
  unitaries.
- Canonical beam search extracts the argmax; sampling is not used.
- The four-hour profile runs one deterministic ordering. A candidate below the
  configured weight floor is rejected; a candidate above it can be returned as
  best-effort when its top-1 margin is not independently decisive.
- The 14,100-second internal budget reserves 900 seconds for final MPO-to-MPS
  extraction, beam search, and the stdout result protocol.

The Dockerfile defines the production settings explicitly. Every setting can be
overridden with its corresponding `HQP_*` environment variable for benchmarking.

## Validation status

The output protocol and exact small-circuit path are locally testable without a
large GPU. A complete blind difficulty-3 GPU run is still required before an
on-chain submission; the `0.0002` profile must not be described as validated
until that run finishes with distance de Hamming zero.
