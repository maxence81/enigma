# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Challenge interface for Hardening Quantum Proof (peaked circuits).

Given a quantum circuit in OpenQASM format, find the computational basis state
with the highest measurement probability -- the "peaked state."
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import *

from . import Serde

_logger = logging.getLogger(__name__)

# Validator read-only mount (see qbittensor.validator.solution.constants).
CHALLENGE_INPUT_JSON_PATH: str = "/challenge_input/challenge_input.json"


@dataclass
class Problem(Serde):
    """
    Problem instance for the Hardening Quantum Proof challenge.

    The miner receives only the difficulty and the path to the QASM file.
    No circuit metadata is exposed -- the miner must simulate the circuit
    to find the peaked state.

    Attributes:
        difficulty: Difficulty level (non-negative integer).
        qasm_file: Path to the OpenQASM circuit file.
    """
    difficulty: int
    qasm_file: str


@dataclass
class Solution(Serde):
    """
    Solution to a Hardening Quantum Proof problem.

    Attributes:
        status: Final solution status ("success", "failed", "timeout").
        peaked_state: The peaked bitstring (only '0's and '1's), or None.
    """
    status: str
    peaked_state: Optional[str]


@dataclass
class Verif(Serde):
    """
    Verification data for a Hardening Quantum Proof problem.

    Attributes:
        peaked_state: The known peaked bitstring.
    """
    peaked_state: str


def validate_hqp_solution(
    solution: Solution, verif: Verif, require_success_status: bool = True
) -> tuple[bool, str | None]:
    """Single source of truth for Hardening Quantum Proof solution correctness.

    Shared between the workbench verifier and the live validator to prevent
    divergence.

    Accepts exact match or reversed bit order (different qubit numbering
    conventions).

    Returns (success, failure_reason or None).
    """
    if require_success_status and getattr(solution, "status", None) != "success":
        return False, f"Solution status is '{getattr(solution, 'status', None)}', not 'success'"

    submitted = solution.peaked_state
    if submitted is None:
        return False, "Solution is missing peaked_state"

    expected = verif.peaked_state

    if submitted == expected:
        return True, None

    if submitted == expected[::-1]:
        return True, None

    return False, f"Incorrect peaked_state: got '{submitted}'"


def load_solver_input(argv: list[str]) -> tuple[str, Problem]:
    """Load ``(challenge_id, Problem)`` for an HQP solver entrypoint.

    Supports two input modes:

    - **Workbench / CLI:** ``argv`` is ``[script, challenge_id, problem_json]``.
    - **Validator Docker:** no CLI args; read ``CHALLENGE_INPUT_JSON_PATH`` from
      the read-only ``/challenge_input`` mount written by ``hqp_setup``.
    """
    if len(argv) == 3:
        return argv[1].strip(), Problem.from_json(argv[2].strip())

    input_path = Path(CHALLENGE_INPUT_JSON_PATH)
    if not input_path.is_file():
        raise FileNotFoundError(
            "HQP input not found: pass '<challenge_id> <JSON Problem>' as CLI args "
            f"or provide {CHALLENGE_INPUT_JSON_PATH} on the challenge input mount."
        )

    data = json.loads(input_path.read_text(encoding="utf-8"))
    challenge_id = str(data.get("challenge_id") or "unknown")
    problem = Problem(
        difficulty=int(data["difficulty"]),
        qasm_file=str(data["qasm_file"]),
    )
    return challenge_id, problem
