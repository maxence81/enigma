#!/usr/bin/env python3
"""Build the deterministic HQP v9 submission archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


FILES = (
    "Dockerfile",
    "README_V9.md",
    "circuit_mpo.py",
    "extract.py",
    "fp32_patch.py",
    "hardening_quantum_proof.py",
    "svd_bench.py",
    "tebd_mps.py",
    "unswap.py",
    "utils.py",
    "enigma_challenges/__init__.py",
    "enigma_challenges/breaking_rsa.py",
    "enigma_challenges/hardening_quantum_proof.py",
    "enigma_challenges/mock_challenge.py",
    "enigma_challenges/solution_output.py",
)
TIMESTAMP = (2026, 9, 19, 0, 0, 0)
EXPECTED_SHA256 = "2a0dbf4443beeaf8f9f253bb71fdf01b3e54093970bc495899b3684956259794"


def main() -> None:
    source = Path(__file__).resolve().parent
    output = Path.cwd() / "hqp_submission_v9.zip"

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_name in FILES:
            info = ZipInfo(relative_name, TIMESTAMP)
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            mode = 0o755 if relative_name == "hardening_quantum_proof.py" else 0o644
            info.external_attr = mode << 16
            archive.writestr(
                info,
                (source / relative_name).read_bytes(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        output.unlink(missing_ok=True)
        raise SystemExit(f"unexpected archive SHA-256: {digest}")
    print(f"{output}: {digest}")


if __name__ == "__main__":
    main()
