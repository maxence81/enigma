# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Challenge interface for the mock (plumbing test) challenge.

The mock challenge uses Ed25519 signatures to verify that only authorized
parties can produce valid solutions. The solver signs a time-based payload
with a private key; the validator verifies the signature against a known
public key and checks that the timestamp is within the validity window.

The private key is never included in source code or Docker images. It lives
on the miner's machine as an environment variable.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import time
from typing import *

from . import Challenge, Serde

_logger = logging.getLogger(__name__)

# Timestamp windows for accepting mock solutions. These are the authoritative
# values used by the live validator (see mock_solution.py). We use them here
# too so that MockChallenge.verify (workbench) and the online validator agree
# and there is no duplicated check logic.
MOCK_TIMESTAMP_MAX_AGE_SECONDS = 3600
MOCK_TIMESTAMP_MAX_FUTURE_SKEW_SECONDS = 60


@dataclass
class Problem(Serde):
    """
    Problem instance for the mock challenge.

    The mock challenge has no computational problem to solve. The problem
    is simply a container for the difficulty label.

    Attributes:
        difficulty (int): Difficulty label (unused, present for interface
            consistency).
    """
    difficulty: int


@dataclass
class Solution(Serde):
    """
    Solution to the mock challenge.

    The solution carries an Ed25519 signature over a JSON payload containing
    a timestamp. The validator verifies the signature and checks that the
    timestamp falls within the validity window.

    Attributes:
        status (str): Final solution status.
        signature (str): Hex-encoded Ed25519 signature over ``payload``.
        payload (str): JSON string containing at least ``{"ts": <unix_epoch>}``.
    """
    status: str
    signature: Optional[str]
    payload: Optional[str]


@dataclass
class Verif(Serde):
    """
    Verification data for the mock challenge.

    Attributes:
        public_key_hex (str): Hex-encoded Ed25519 public key.
    """
    public_key_hex: str


def _validate_mock_timestamp_and_signature(
    solution: Solution, verif: Verif
) -> tuple[bool, str | None]:
    """Pure core logic for Ed25519 sig + timestamp validity.

    Single implementation shared between MockChallenge.verify (workbench/offline)
    and the validator's mock_solution.run (online) to eliminate duplication and
    divergence.
    """
    if solution.signature is None or solution.payload is None:
        return False, "signature or payload is None"

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        return False, "cryptography package required for mock challenge verification"

    try:
        payload_data = json.loads(solution.payload)
        ts = int(payload_data["ts"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return False, f"Invalid mock payload JSON: {e}"

    now = time.time()
    skew_seconds = ts - now
    if skew_seconds > MOCK_TIMESTAMP_MAX_FUTURE_SKEW_SECONDS:
        return False, (
            f"Mock payload timestamp is too far in the future "
            f"({skew_seconds:.0f}s > {MOCK_TIMESTAMP_MAX_FUTURE_SKEW_SECONDS}s skew allowed)"
        )

    age_seconds = now - ts
    if age_seconds > MOCK_TIMESTAMP_MAX_AGE_SECONDS:
        return False, (
            f"Mock payload timestamp older than one hour "
            f"({age_seconds:.0f}s > {MOCK_TIMESTAMP_MAX_AGE_SECONDS}s)"
        )

    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(verif.public_key_hex)
        )
        public_key.verify(
            bytes.fromhex(solution.signature),
            solution.payload.encode("utf-8"),
        )
    except Exception as e:
        return False, f"Invalid mock signature verification: {e}"

    return True, None


def validate_mock_solution(solution: Solution, verif: Verif) -> tuple[bool, str | None]:
    """Public entrypoint for validating a mock Solution against its Verif.

    This is the single source of truth for "is this solution correct?" for the
    mock challenge. Used by both the platform validator and the offline
    workbench.
    """
    return _validate_mock_timestamp_and_signature(solution, verif)


@dataclass
class MockChallenge(Challenge[Problem, Solution, Verif]):
    """
    Mock challenge for testing the validator/miner pipeline.

    This challenge verifies that a solution was produced by a party holding
    the Ed25519 private key corresponding to ``public_key_hex``, and that
    the signed timestamp is recent (within ``MOCK_TIMESTAMP_MAX_AGE_SECONDS``).

    Attributes:
        difficulty (int): Difficulty label.
        public_key_hex (str): Hex-encoded Ed25519 public key.
    """
    difficulty: int
    public_key_hex: str

    def generate(self, seed: int) -> tuple[Problem, Verif]:
        problem = Problem(difficulty=self.difficulty)
        verif = Verif(public_key_hex=self.public_key_hex)
        return problem, verif

    def verify(self, prob: Problem, sol: Solution, verif: Verif) -> bool:
        success, reason = validate_mock_solution(sol, verif)
        if success:
            _logger.info("Verification SUCCESS: valid signature, timestamp OK")
        else:
            _logger.info(f"Verification FAILURE: {reason}")
        return success


def sign_mock_payload(private_key_hex: str) -> Solution:
    """
    Sign a time-based payload with an Ed25519 private key.

    This is the "solver" for the mock challenge. Call this on the miner
    side (never inside the Docker container).

    Args:
        private_key_hex: Hex-encoded 32-byte Ed25519 private key.

    Returns:
        A Solution with the signature and payload filled in.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    payload = json.dumps({"ts": int(time.time()), "challenge": "mock"})
    private_key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(private_key_hex)
    )
    signature = private_key.sign(payload.encode("utf-8"))

    return Solution(
        status="success",
        signature=signature.hex(),
        payload=payload,
    )


def generate_keypair() -> tuple[str, str]:
    """
    Generate a new Ed25519 keypair.

    Returns:
        (private_key_hex, public_key_hex) -- both as hex strings.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat, PrivateFormat, NoEncryption,
    )

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    public_bytes = private_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    return private_bytes.hex(), public_bytes.hex()
